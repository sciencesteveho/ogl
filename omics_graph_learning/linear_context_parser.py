#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Parse local genomic data to nodes and attributes"""


from itertools import repeat
from multiprocessing import Pool
import os
import pickle
import subprocess
from subprocess import PIPE
from subprocess import Popen
from typing import Any, Dict, List, Optional, Tuple, Union

import pybedtools  # type: ignore
from pybedtools.featurefuncs import extend_fields  # type: ignore

from config_handlers import ExperimentConfig
from config_handlers import TissueConfig
from constants import ATTRIBUTES
from positional_encoding import PositionalEncoding
from utils import dir_check_make
from utils import genes_from_gencode
from utils import time_decorator


class LinearContextParser:
    """Object that parses local genomic da
    ta into graph edges

    Attributes:
        experiment_config: Dataclass containing the experiment configuration.
        tissue_config: Dataclass containing the tissue configuration.
        bedfiles: dictionary containing each local genomic data file

    Methods
    ----------
    _make_directories:
        Prepare necessary directories.
    _prepare_local_features:
        Creates a dict of local context datatypes and their bedtools objects.
    _slop_sort:
        Slop each line of a bedfile to get all features within a window.
    _bed_intersect:
        Function to intersect a slopped bed entry with all other node types.
    _aggregate_attributes:
        For each node of a node_type get their overlap with gene windows then
        aggregate total nucleotides, gc content, and all other attributes.
    _generate_edges:
        Unix concatenate and sort each edge file.
    _save_node_attributes:
        Save attributes for all node entries.
    parse_context_data:
        Parse local genomic data into graph edges.

    # Helpers
        ATTRIBUTES -- list of node attribute types
        DIRECT -- list of datatypes that only get direct overlaps, no slop
        ONEHOT_NODETYPE -- dictionary of node type one-hot vectors

    Examples:
    --------
    >>> localparseObject = LinearContextParser(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
        bedfiles=adjusted_bedfiles,
    )

    >>> localparseObject.parse_context_data()
    """

    # list helpers
    DIRECT = ["tads", "loops"]
    NODE_FEATS = ["start", "end", "size"] + ATTRIBUTES

    # var helpers - for CPU cores
    ATTRIBUTE_PROCESSES = 16

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        tissue_config: TissueConfig,
        bedfiles: List[str],
    ):
        """Initialize the class"""
        self.bedfiles = bedfiles
        self.blacklist = experiment_config.blacklist
        self.chromfile = experiment_config.chromfile
        self.experiment_name = experiment_config.experiment_name
        self.fasta = experiment_config.fasta
        self.feat_window = experiment_config.feat_window
        self.nodes = experiment_config.nodes
        self.working_directory = experiment_config.working_directory
        self.positional_encoding = experiment_config.positional_encoding

        self.resources = tissue_config.resources
        self.gencode = tissue_config.local["gencode"]
        self.tissue = tissue_config.resources["tissue"]
        self.tss = tissue_config.local["tss"]

        self.node_processes = len(self.nodes) + 1
        self._set_directories()
        self._prepare_bed_references()

        # make directories
        self._make_directories()

    def _set_directories(self) -> None:
        """Set directories from yaml"""
        self.tissue_dir = self.working_directory / self.tissue
        self.local_dir = self.tissue_dir / "local"
        self.parse_dir = self.tissue_dir / "parsing"
        self.attribute_dir = self.parse_dir / "attributes"

        self.edge_dir = self.parse_dir / "edges"
        self.intermediate_sorted = self.parse_dir / "intermediate" / "sorted"

    def _prepare_bed_references(
        self,
    ) -> None:
        """Prepares genesymbol to gencode conversion and blacklist as pybedtools
        objects.
        """
        self.genesymbol_to_gencode = genes_from_gencode(
            pybedtools.BedTool(self.local_dir / self.gencode)  # type: ignore
        )
        self.blacklist = pybedtools.BedTool(self.blacklist).sort().saveas()  # type: ignore

    def _make_directories(self) -> None:
        """Directories for parsing genomic bedfiles into graph edges and nodes"""
        dir_check_make(self.parse_dir)

        for directory in [
            "edges",
            "attributes",
            "intermediate/slopped",
            "intermediate/sorted",
        ]:
            dir_check_make(self.parse_dir / directory)

        for attribute in ATTRIBUTES:
            dir_check_make(self.attribute_dir / attribute)

    @time_decorator(print_args=True)
    def _prepare_local_features(
        self,
        bed: str,
    ) -> Tuple[str, pybedtools.BedTool]:
        """
        Creates a dict of local context datatypes and their bedtools objects.
        Renames features if necessary. Intersects each bed to get only features
        that do not intersect ENCODE blacklist regions.
        """

        def rename_feat_chr_start(feature: Any) -> str:
            """Add chr, start to feature name
            Cpgislands add prefix to feature names  # enhancers,
            Histones add an additional column
            """
            simple_rename = [
                "cpgislands",
                "crms",
            ]
            if prefix in simple_rename:
                feature = extend_fields(feature, 4)
                feature[3] = f"{feature[0]}_{feature[1]}_{prefix}"
            else:
                feature[3] = f"{feature[0]}_{feature[1]}_{feature[3]}"
            return feature

        # prepare data as pybedtools objects and intersect -v against blacklist
        # beds = {}
        local_bed = pybedtools.BedTool(self.local_dir / bed).sort()
        ab = self._remove_blacklist_and_alt_configs(
            bed=local_bed, blacklist=self.blacklist
        )

        # rename features if necessary and only keep coords
        prefix = bed.split("_")[0].lower()
        if prefix in self.nodes and prefix != "gencode":
            result = ab.each(rename_feat_chr_start).cut([0, 1, 2, 3]).saveas()
            prepared_bed = pybedtools.BedTool(str(result), from_string=True)
        else:
            prepared_bed = ab.cut([0, 1, 2, 3])

        return prefix, prepared_bed

    @time_decorator(print_args=True)
    def _slop_sort(
        self,
        bedinstance: Dict[str, pybedtools.BedTool],
        chromfile: str,
        feat_window: int,
    ) -> Tuple[Dict[str, pybedtools.BedTool], Dict[str, pybedtools.BedTool]]:
        """Slop each line of a bedfile to get all features within a window

        Args:
            bedinstance // a region-filtered genomic bedfile
            chromfile // textfile with sizes of each chromosome in hg38

        Returns:
            bedinstance_sorted -- sorted bed
            bedinstance_slopped -- bed slopped by amount in feat_window
        """
        bedinstance_slopped, bedinstance_sorted = {}, {}
        for key, value in bedinstance.items():
            bedinstance_sorted[key] = bedinstance[key].sort()
            if key not in ATTRIBUTES + self.DIRECT:
                nodes = bedinstance[key].slop(g=chromfile, b=feat_window).sort()
                newstrings = [
                    str(line_1).split("\n")[0] + "\t" + str(line_2)
                    for line_1, line_2 in zip(nodes, value)
                ]
                bedinstance_slopped[key] = pybedtools.BedTool(
                    "".join(newstrings), from_string=True
                ).sort()
        return bedinstance_sorted, bedinstance_slopped

    @time_decorator(print_args=True)
    def _bed_intersect(
        self,
        node_type: str,
        all_files: str,
    ) -> None:
        """Function to intersect a slopped bed entry with all other node types.
        Each bed is slopped then intersected twice. First, it is intersected
        with every other node type. Then, the intersected bed is filtered to
        only keep edges within the gene region."""
        print(f"starting combinations {node_type}")

        def _unix_intersect(node_type: str, type: Optional[str] = None) -> None:
            """Intersect and cut relevant columns"""
            if type == "direct":
                folder = "sorted"
                cut_cmd = ""
            else:
                folder = "slopped"
                cut_cmd = " | cut -f5,6,7,8,9,10,11,12"

            final_cmd = f"bedtools intersect \
                -wa \
                -wb \
                -sorted \
                -a {self.parse_dir}/intermediate/{folder}/{node_type}.bed \
                -b {all_files}"

            with open(self.edge_dir / f"{node_type}.bed", "w") as outfile:
                subprocess.run(final_cmd + cut_cmd, stdout=outfile, shell=True)
            outfile.close()

        def _filter_duplicate_bed_entries(
            bedfile: pybedtools.bedtool.BedTool,
        ) -> pybedtools.bedtool.BedTool:
            """Filters a bedfile by removing entries that are identical"""
            return bedfile.filter(
                lambda x: [x[0], x[1], x[2], x[3]] != [x[4], x[5], x[6], x[7]]
            ).saveas()

        def _add_distance(feature: Any) -> str:
            """Add distance as [8]th field to each overlap interval"""
            feature = extend_fields(feature, 9)
            feature[8] = max(int(feature[1]), int(feature[5])) - min(
                int(feature[2]), int(feature[5])
            )
            return feature

        def _insulate_regions(deduped_edges: str, loopfile: str) -> None:
            """Insulate regions by intersection with chr loops and making sure
            both sides overlap"""

        if node_type in self.DIRECT:
            _unix_intersect(node_type, type="direct")
            _filter_duplicate_bed_entries(
                pybedtools.BedTool(self.edge_dir / f"{node_type}.bed")
            ).sort().saveas(self.edge_dir / f"{node_type}_dupes_removed")
        else:
            _unix_intersect(node_type)
            _filter_duplicate_bed_entries(
                pybedtools.BedTool(self.edge_dir / f"{node_type}.bed")
            ).each(_add_distance).saveas().sort().saveas(
                self.edge_dir / f"{node_type}_dupes_removed"
            )

    @time_decorator(print_args=True)
    def _aggregate_attributes(self, node_type: str) -> None:
        """For each node of a node_type get their overlap with gene windows then
        aggregate total nucleotides, gc content, and all other attributes."""

        def add_size(feature: Any) -> str:
            """Add size as a field to the bedtool object"""
            feature = extend_fields(feature, 5)
            feature[4] = feature.end - feature.start
            return feature

        def sum_gc(feature: Any) -> str:
            """Sum GC content for each feature"""
            feature[13] = int(feature[8]) + int(feature[9])
            return feature

        ref_file = (
            pybedtools.BedTool(self.intermediate_sorted / f"{node_type}.bed")
            .filter(lambda x: "alt" not in x[0])
            .each(add_size)
            .sort()
            .saveas()
        )

        for attribute in ATTRIBUTES:
            save_file = (
                self.attribute_dir / attribute / f"{node_type}_{attribute}_percentage"
            )
            print(f"{attribute} for {node_type}")
            if attribute == "gc":
                ref_file.nucleotide_content(fi=self.fasta).each(sum_gc).sort().groupby(
                    g=[1, 2, 3, 4], c=[5, 14], o=["sum"]
                ).saveas(save_file)
            elif attribute == "recombination":
                ref_file.intersect(
                    pybedtools.BedTool(self.intermediate_sorted / f"{attribute}.bed"),
                    wao=True,
                    sorted=True,
                ).groupby(g=[1, 2, 3, 4], c=[5, 9], o=["sum", "mean"]).sort().saveas(
                    save_file
                )
            else:
                ref_file.intersect(
                    pybedtools.BedTool(self.intermediate_sorted / f"{attribute}.bed"),
                    wao=True,
                    sorted=True,
                ).groupby(g=[1, 2, 3, 4], c=[5, 10], o=["sum"]).sort().saveas(save_file)

    @time_decorator(print_args=True)
    def _generate_edges(self) -> None:
        """Unix concatenate and sort each edge file with parallelization options."""

        def _chk_file_and_run(file: str, cmd: str) -> None:
            """Check that a file does not exist before calling subprocess"""
            if not os.path.isfile(file) or os.path.getsize(file) == 0:
                subprocess.run(cmd, stdout=None, shell=True)

        cmds = {
            "cat_cmd": [
                f"cat {self.parse_dir}/edges/*_dupes_removed* >",
                f"{self.parse_dir}/edges/all_concat.bed",
            ],
            "sort_cmd": [
                f"LC_ALL=C sort --parallel=32 -S 80% -k10,10 {self.parse_dir}/edges/all_concat.bed |",
                "uniq >" f"{self.parse_dir}/edges/all_concat_sorted.bed",
            ],
        }

        for cmd in cmds:
            _chk_file_and_run(
                cmds[cmd][1],
                cmds[cmd][0] + cmds[cmd][1],
            )

    @time_decorator(print_args=True)
    def _save_node_attributes(
        self,
        node: str,
    ) -> None:
        """Save attributes for all node entries to create node attributes during
        graph construction."""
        # initialize position encoding
        if self.positional_encoding:
            positional_encoding = PositionalEncoding(
                chromfile=self.chromfile,
                binsize=50000,
            )

        stored_attributed: Dict[str, Dict[str, Union[str, float, Dict[str, float]]]] = (
            {}
        )
        bedfile_lines = {}
        for attribute in ATTRIBUTES:
            filename = self.attribute_dir / attribute / f"{node}_{attribute}_percentage"
            with open(filename, "r") as file:
                lines = [tuple(line.rstrip().split("\t")) for line in file]
                bedfile_lines[attribute] = set(lines)
            for line in bedfile_lines[attribute]:
                if attribute == "gc":
                    stored_attributed[f"{line[3]}_{self.tissue}"] = {
                        "chr": line[0].replace("chr", ""),
                    }
                    stored_attributed[f"{line[3]}_{self.tissue}"] = {
                        "coordinates": {"start": float(line[1]), "end": float(line[2])},
                        "size": float(line[4]),
                        "gc": float(line[5]),
                    }
                    if self.positional_encoding:
                        encoding = positional_encoding(
                            chromosome=line[0], start=line[1], end=line[2]
                        )
                        stored_attributed[f"{line[3]}_{self.tissue}"][
                            "positional_encoding"
                        ] = encoding
                else:
                    try:
                        stored_attributed[f"{line[3]}_{self.tissue}"][attribute] = (
                            float(line[5])
                        )
                    except ValueError:
                        stored_attributed[f"{line[3]}_{self.tissue}"][attribute] = 0

        with open(self.attribute_dir / f"{node}_reference.pkl", "wb") as output:
            pickle.dump(stored_attributed, output)

    @time_decorator(print_args=True)
    def parse_context_data(self) -> None:
        """Parse local genomic data into graph edges."""

        @time_decorator(print_args=True)
        def _save_intermediate(
            bed_dictionary: Dict[str, pybedtools.bedtool.BedTool], folder: str
        ) -> None:
            """Save region specific bedfiles"""
            for key in bed_dictionary:
                file = self.parse_dir / "intermediate" / f"{folder}/{key}.bed"
                if not os.path.exists(file):
                    bed_dictionary[key].saveas(file)

        @time_decorator(print_args=True)
        def _pre_concatenate_all_files(all_files: str) -> None:
            """Pre-concatenate via unix commands to save time"""
            if not os.path.exists(all_files) or os.stat(all_files).st_size == 0:
                cat_cmd = ["cat"] + [
                    self.intermediate_sorted / f"{x}.bed" for x in bedinstance_slopped
                ]
                sort_cmd = "sort -k1,1 -k2,2n"
                concat = Popen(cat_cmd, stdout=PIPE)
                with open(all_files, "w") as outfile:
                    subprocess.run(
                        sort_cmd, stdin=concat.stdout, stdout=outfile, shell=True
                    )
                outfile.close()

        # process windows and renaming
        pool = Pool(processes=self.node_processes)
        bedinstance_flat = dict(
            pool.starmap(
                self._prepare_local_features, [(bed,) for bed in self.bedfiles]
            )
        )
        pool.close()  # re-open and close pool after every multi-process

        # sort and extend windows according to FEAT_WINDOWS
        bedinstance_sorted, bedinstance_slopped = self._slop_sort(
            bedinstance=bedinstance_flat,
            chromfile=self.chromfile,
            feat_window=self.feat_window,
        )

        # save intermediate files
        _save_intermediate(bedinstance_sorted, folder="sorted")
        _save_intermediate(bedinstance_slopped, folder="slopped")

        # pre-concatenate to save time
        all_files = self.intermediate_sorted / "all_files_concatenated.bed"
        _pre_concatenate_all_files(all_files)

        # perform intersects across all feature types - one process per nodetype
        pool = Pool(processes=self.node_processes)
        pool.starmap(self._bed_intersect, zip(self.nodes, repeat(all_files)))
        pool.close()

        # get size and all attributes - one process per nodetype
        pool = Pool(processes=self.ATTRIBUTE_PROCESSES)
        pool.map(self._aggregate_attributes, ["basenodes"] + self.nodes)
        pool.close()

        # parse edges into individual files
        self._generate_edges()

        # save node attributes as reference for later - one process per nodetype
        pool = Pool(processes=self.ATTRIBUTE_PROCESSES)
        pool.map(self._save_node_attributes, ["basenodes"] + self.nodes)
        pool.close()

        # cleanup: remove intermediate files
        retain = ["all_concat_sorted.bed"]
        for item in os.listdir(self.edge_dir):
            if item not in retain:
                os.remove(self.edge_dir / item)

    def _remove_blacklist_and_alt_configs(
        self,
        bed: pybedtools.bedtool.BedTool,
        blacklist: pybedtools.bedtool.BedTool,
    ) -> pybedtools.bedtool.BedTool:
        """Remove blacklist and alternate chromosomes from bedfile."""
        return self._remove_alt_configs(bed.intersect(blacklist, sorted=True, v=True))

    @staticmethod
    def _remove_alt_configs(
        bed: pybedtools.bedtool.BedTool,
    ) -> pybedtools.bedtool.BedTool:
        """Remove alternate chromosomes from bedfile."""
        return bed.filter(lambda x: "_" not in x[0]).saveas()
