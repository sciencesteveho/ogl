#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import argparse
import csv
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

from utils import _listdir_isfile_wrapper
from utils import _tpm_filter_gene_windows
from utils import ATTRIBUTES
from utils import dir_check_make
from utils import genes_from_gencode
from utils import NODES
from utils import parse_yaml
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
        FEAT_WINDOWS -- dictionary of each nodetype: overlap windows
        NODES -- list of nodetypes
        ONEHOT_NODETYPE -- dictionary of node type one-hot vectors
    """

    # list helpers
    DIRECT = ["tads"]
    NODE_FEATS = ["start", "end", "size"] + ATTRIBUTES

    # var helpers - for CPU cores
    NODE_CORES = len(NODES) + 1  # 12
    ATTRIBUTE_CORES = len(ATTRIBUTES)  # 3

    def __init__(
        self,
        bedfiles: List[str],
        params: Dict[str, Dict[str, str]],
    ):
        """Initialize the class"""
        self.bedfiles = bedfiles
        self.resources = params["resources"]
        self.tissue_specific = params["tissue_specific"]
        self.gencode = params["local"]["gencode"]

        self.tissue = self.resources["tissue"]
        self.chromfile = self.resources["chromfile"]
        self.fasta = self.resources["fasta"]

        self.root_dir = params["dirs"]["root_dir"]
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
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
            )

        # prepare references
        self.gencode_ref = pybedtools.BedTool(genes)
        self.gene_windows = pybedtools.BedTool(gene_windows)
        self.genesymbol_to_gencode = genes_from_gencode(
            pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        )

        # make directories
        self._make_directories()

    def _prepare_tpm_filtered_genes(
        self, genes: str, gene_windows: str, base_nodes: str
    ) -> None:
        """Prepare tpm filtered genes and gene windows"""
        filtered_genes = _tpm_filter_gene_windows(
            gencode=f"{self.root_dir}/shared_data/local/{self.gencode}",
            tissue=self.tissue,
            tpm_file=self.resources["tpm"],
            chromfile=self.chromfile,
            slop=False,
        )

        windows = pybedtools.BedTool(base_nodes).slop(g=self.chromfile, b=25000).sort()
        filtered_genes.saveas(genes)
        windows.saveas(gene_windows)

    def _make_directories(self) -> None:
        """Directories for parsing genomic bedfiles into graph edges and nodes"""
        dir_check_make(self.parse_dir)

        for directory in [
            "edges/genes",
            "attributes",
            "intermediate/slopped",
            "intermediate/sorted",
        ]:
            dir_check_make(f"{self.parse_dir}/{directory}")

        for attribute in ATTRIBUTES:
            dir_check_make(f"{self.attribute_dir}/{attribute}")

    @time_decorator(print_args=True)
    def _region_specific_features_dict(
        self, bed: str
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
        b = pybedtools.BedTool(f"{self.root_dir}/{self.tissue}/local/{bed}").sort()
        ab = b.intersect(a, sorted=True, u=True)

        # take specific windows and format each file
        if prefix in NODES and prefix != "gencode":
            result = ab.each(rename_feat_chr_start).cut([0, 1, 2, 3]).saveas()
            bed_dict[prefix] = pybedtools.BedTool(str(result), from_string=True)
        else:
            bed_dict[prefix] = ab.cut([0, 1, 2, 3])

        return bed_dict

    @time_decorator(print_args=True)
    def _slop_sort(
        self, bedinstance: Dict[str, str], chromfile: str, feat_window: int = 2000
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
    def _bed_intersect(self, node_type: str, all_files: str) -> None:
        """Function to intersect a slopped bed entry with all other node types.
        Each bed is slopped then intersected twice. First, it is intersected
        with every other node type. Then, the intersected bed is filtered to
        only keep edges within the gene region.

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


def place_holder_function():
    """_summary_ of function"""
    pass


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
