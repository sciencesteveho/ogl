#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] fix params for cores
# - [ ] try and refactor yamls and init


"""Parse local genomic data to nodes and attributes"""

from itertools import repeat
from multiprocessing import Pool
import os
import pickle
import subprocess
from subprocess import PIPE
from subprocess import Popen
from typing import Dict, List, Optional, Tuple

import pybedtools
from pybedtools.featurefuncs import extend_fields

from utils import ATTRIBUTES
from utils import GeneralUtils
from utils import GenomeDataUtils
from utils import time_decorator


class LocalContextParser:
    """Object that parses local genomic data into graph edges

    Args:
        bedfiles // dictionary containing each local genomic data    type as bedtool
            obj
        windows // bedtool object of windows +/- 250k of protein coding genes
        params // configuration vals from yaml

    Methods
    ----------
    _make_directories:
        prepare necessary directories

    # Helpers
        ATTRIBUTES -- list of node attribute types
        DIRECT -- list of datatypes that only get direct overlaps, no slop
        ONEHOT_NODETYPE -- dictionary of node type one-hot vectors
    """

    # list helpers
    DIRECT = ["tads"]
    NODE_FEATS = ["start", "end", "size"] + ATTRIBUTES

    # var helpers - for CPU cores
    ATTRIBUTE_PROCESSES = 64

    def __init__(
        self,
        experiment_name: str,
        nodes: List[str],
        working_directory: str,
        bedfiles: List[str],
        params: Dict[str, Dict[str, str]],
    ):
        """Initialize the class"""
        self.bedfiles = bedfiles
        self.experiment_name = experiment_name
        self.nodes = nodes
        self.working_directory = working_directory
        self.node_processes = len(nodes) + 1  # 12

        self.resources = params["resources"]
        self.gencode = params["local"]["gencode"]

        self.tissue = self.resources["tissue"]
        self.chromfile = self.resources["chromfile"]
        self.fasta = self.resources["fasta"]

        self.root_dir = params["dirs"]["root_dir"]
        self.tissue_dir = (
            f"{self.working_directory}/{self.experiment_name}/{self.tissue}"
        )
        self.local_dir = f"{self.tissue_dir}/local"
        self.parse_dir = f"{self.tissue_dir}/parsing"
        self.attribute_dir = f"{self.parse_dir}/attributes"

        genes = f"{self.tissue_dir}/tpm_filtered_genes.bed"
        gene_windows = f"{self.tissue_dir}/tpm_filtered_gene_regions.bed"

        # prepare list of genes passing tpm filter
        if not (os.path.exists(genes) and os.stat(genes).st_size > 0):
            self._prepare_tpm_filtered_genes(
                genes=genes,
                gene_windows=gene_windows,
                base_nodes=f"{self.tissue_dir}/local/basenodes_hg38.txt",
                gct=f"{params['resources']['tpm']}",
            )

        # prepare references
        self.gencode_ref = pybedtools.BedTool(genes)
        self.gene_windows = pybedtools.BedTool(gene_windows)
        self.genesymbol_to_gencode = GenomeDataUtils.genes_from_gencode(
            pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        )

        # make directories
        self._make_directories()

    def _prepare_tpm_filtered_genes(
        self,
        genes: str,
        gene_windows: str,
        base_nodes: str,
        gct: str,
    ) -> None:
        """Prepare tpm filtered genes and gene windows"""
        filtered_genes = GenomeDataUtils._tpm_filter_gene_windows(
            gencode=f"{self.root_dir}/shared_data/local/{self.gencode}",
            tissue=self.tissue,
            tpm_file=gct,
            chromfile=self.chromfile,
        )

        windows = pybedtools.BedTool(base_nodes).slop(g=self.chromfile, b=25000).sort()
        filtered_genes.saveas(genes)
        windows.saveas(gene_windows)

    def _make_directories(self) -> None:
        """Directories for parsing genomic bedfiles into graph edges and nodes"""
        GeneralUtils.dir_check_make(self.parse_dir)

        for directory in [
            "edges",
            "attributes",
            "intermediate/slopped",
            "intermediate/sorted",
        ]:
            GeneralUtils.dir_check_make(f"{self.parse_dir}/{directory}")

        for attribute in ATTRIBUTES:
            GeneralUtils.dir_check_make(f"{self.attribute_dir}/{attribute}")

    @time_decorator(print_args=True)
    def _region_specific_features_dict(
        self,
        bed: str,
    ) -> List[Dict[str, pybedtools.bedtool.BedTool]]:
        """
        Creates a dict of local context datatypes and their bedtools objects.
        Renames features if necessary.
        """

        def rename_feat_chr_start(feature: str) -> str:
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

        # prepare data as pybedtools objects
        bed_dict = {}
        prefix = bed.split("_")[0].lower()
        a = self.gene_windows
        b = pybedtools.BedTool(f"{self.tissue_dir}/local/{bed}").sort()
        ab = b.intersect(a, sorted=True, u=True)

        # take specific windows and format each file
        if prefix in self.nodes and prefix != "gencode":
            result = ab.each(rename_feat_chr_start).cut([0, 1, 2, 3]).saveas()
            bed_dict[prefix] = pybedtools.BedTool(str(result), from_string=True)
        else:
            bed_dict[prefix] = ab.cut([0, 1, 2, 3])

        return bed_dict

    @time_decorator(print_args=True)
    def _slop_sort(
        self,
        bedinstance: Dict[str, str],
        chromfile: str,
        feat_window: int = 2000,
    ) -> Tuple[
        Dict[str, pybedtools.bedtool.BedTool], Dict[str, pybedtools.bedtool.BedTool]
    ]:
        """Slop each line of a bedfile to get all features within a window

        Args:
            bedinstance // a region-filtered genomic bedfile
            chromfile // textfile with sizes of each chromosome in hg38

        Returns:
            bedinstance_sorted -- sorted bed
            bedinstance_slopped -- bed slopped by amount in feat_window
        """
        bedinstance_slopped, bedinstance_sorted = {}, {}
        for key in bedinstance.keys():
            bedinstance_sorted[key] = bedinstance[key].sort()
            if key in ATTRIBUTES + self.DIRECT:
                pass
            else:
                nodes = bedinstance[key].slop(g=chromfile, b=feat_window).sort()
                newstrings = []
                for line_1, line_2 in zip(nodes, bedinstance[key]):
                    newstrings.append(str(line_1).split("\n")[0] + "\t" + str(line_2))
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
        only keep edges within the gene region.

        Additionally, there is an option to insulate regions and only create
        interactions if they exist within the same chromatin loop.

        Args:
            node_type // _description_
            all_files // _description_

        Raises:
            AssertionError: _description_
        """
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

            with open(f"{self.parse_dir}/edges/{node_type}.bed", "w") as outfile:
                subprocess.run(final_cmd + cut_cmd, stdout=outfile, shell=True)
            outfile.close()

        def _filter_duplicate_bed_entries(
            bedfile: pybedtools.bedtool.BedTool,
        ) -> pybedtools.bedtool.BedTool:
            """Filters a bedfile by removing entries that are identical"""
            return bedfile.filter(
                lambda x: [x[0], x[1], x[2], x[3]] != [x[4], x[5], x[6], x[7]]
            ).saveas()

        def _add_distance(feature: str) -> str:
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
                pybedtools.BedTool(f"{self.parse_dir}/edges/{node_type}.bed")
            ).sort().saveas(f"{self.parse_dir}/edges/{node_type}_dupes_removed")
        else:
            _unix_intersect(node_type)
            _filter_duplicate_bed_entries(
                pybedtools.BedTool(f"{self.parse_dir}/edges/{node_type}.bed")
            ).each(_add_distance).saveas().sort().saveas(
                f"{self.parse_dir}/edges/{node_type}_dupes_removed"
            )

    @time_decorator(print_args=True)
    def _aggregate_attributes(self, node_type: str) -> None:
        """For each node of a node_type get their overlap with gene windows then
        aggregate total nucleotides, gc content, and all other attributes

        Args:
            node_type // node datatype in self.nodes
        """

        def add_size(feature: str) -> str:
            """ """
            feature = extend_fields(feature, 5)
            feature[4] = feature.end - feature.start
            return feature

        def sum_gc(feature: str) -> str:
            """ """
            feature[13] = int(feature[8]) + int(feature[9])
            return feature

        ref_file = pybedtools.BedTool(
            f"{self.parse_dir}/intermediate/sorted/{node_type}.bed"
        )
        ref_file = (
            ref_file.filter(lambda x: "alt" not in x[0]).each(add_size).sort().saveas()
        )

        for attribute in ATTRIBUTES:
            save_file = (
                f"{self.attribute_dir}/{attribute}/{node_type}_{attribute}_percentage"
            )
            print(f"{attribute} for {node_type}")
            if attribute == "gc":
                ref_file.nucleotide_content(fi=self.fasta).each(sum_gc).sort().groupby(
                    g=[1, 2, 3, 4], c=[5, 14], o=["sum"]
                ).saveas(save_file)
            elif attribute == "recombination":
                ref_file.intersect(
                    f"{self.parse_dir}/intermediate/sorted/{attribute}.bed",
                    wao=True,
                    sorted=True,
                ).groupby(g=[1, 2, 3, 4], c=[5, 9], o=["sum", "mean"]).sort().saveas(
                    save_file
                )
            else:
                ref_file.intersect(
                    f"{self.parse_dir}/intermediate/sorted/{attribute}.bed",
                    wao=True,
                    sorted=True,
                ).groupby(g=[1, 2, 3, 4], c=[5, 10], o=["sum"]).sort().saveas(save_file)

    @time_decorator(print_args=True)
    def _generate_edges(self) -> None:
        """Unix concatenate and sort each edge file"""

        def _chk_file_and_run(file: str, cmd: str) -> None:
            """Check that a file does not exist before calling subprocess"""
            if os.path.isfile(file) and os.path.getsize(file) != 0:
                pass
            else:
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
        """
        Save attributes for all node entries. Used during graph construction for
        gene_nodes that fall outside of the gene window and for some gene_nodes
        from interaction data
        """

        attr_dict, attr_dict_nochr, set_dict = (
            {},
            {},
            {},
        )  # dict[gene] = [chr, start, end, size, gc]
        for attribute in ATTRIBUTES:
            filename = (
                f"{self.parse_dir}/attributes/{attribute}/{node}_{attribute}_percentage"
            )
            with open(filename, "r") as file:
                lines = [tuple(line.rstrip().split("\t")) for line in file]
                set_dict[attribute] = set(lines)
            for line in set_dict[attribute]:
                if attribute == "gc":
                    attr_dict[f"{line[3]}_{self.tissue}"] = {
                        "chr": line[0].replace("chr", ""),
                    }
                    for dictionary in [attr_dict, attr_dict_nochr]:
                        dictionary[f"{line[3]}_{self.tissue}"] = {
                            "start": float(line[1]),
                            "end": float(line[2]),
                            "size": float(line[4]),
                            "gc": float(line[5]),
                        }
                else:
                    try:
                        for dictionary in [attr_dict, attr_dict_nochr]:
                            dictionary[f"{line[3]}_{self.tissue}"][attribute] = float(
                                line[5]
                            )
                    except ValueError:
                        for dictionary in [attr_dict, attr_dict_nochr]:
                            dictionary[f"{line[3]}_{self.tissue}"][attribute] = 0

        with open(f"{self.parse_dir}/attributes/{node}_reference.pkl", "wb") as output:
            pickle.dump(attr_dict, output)

        # with open(
        #     f"{self.parse_dir}/attributes/{node}_reference_nochr.pkl", "wb"
        # ) as output:
        #     pickle.dump(attr_dict_nochr, output)

    @time_decorator(print_args=True)
    def parse_context_data(self) -> None:
        """_summary_

        Args:
            a // _description_
            b // _description_

        Raises:
            AssertionError: _description_

        Returns:
            c -- _description_
        """

        @time_decorator(print_args=True)
        def _save_intermediate(
            bed_dictionary: Dict[str, pybedtools.bedtool.BedTool], folder: str
        ) -> None:
            """Save region specific bedfiles"""
            for key in bed_dictionary:
                file = f"{self.parse_dir}/intermediate/{folder}/{key}.bed"
                if not os.path.exists(file):
                    bed_dictionary[key].saveas(file)

        @time_decorator(print_args=True)
        def _pre_concatenate_all_files(all_files: str) -> None:
            """Lorem Ipsum"""
            if not os.path.exists(all_files) or os.stat(all_files).st_size == 0:
                cat_cmd = ["cat"] + [
                    f"{self.parse_dir}/intermediate/sorted/" + x + ".bed"
                    for x in bedinstance_slopped
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
        bedinstance = pool.map(
            self._region_specific_features_dict, [bed for bed in self.bedfiles]
        )
        pool.close()  # re-open and close pool after every multi-process

        # convert back to dictionary
        bedinstance = {
            key.casefold(): value
            for element in bedinstance
            for key, value in element.items()
        }

        # sort and extend windows according to FEAT_WINDOWS
        bedinstance_sorted, bedinstance_slopped = self._slop_sort(
            bedinstance=bedinstance,
            chromfile=self.chromfile,
            feat_window=2000,
        )

        # save intermediate files
        _save_intermediate(bedinstance_sorted, folder="sorted")
        _save_intermediate(bedinstance_slopped, folder="slopped")

        # pre-concatenate to save time
        all_files = f"{self.parse_dir}/intermediate/sorted/all_files_concatenated.bed"
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
        for item in os.listdir(f"{self.parse_dir}/edges"):
            if item not in retain:
                os.remove(f"{self.parse_dir}/edges/{item}")
