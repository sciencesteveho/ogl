#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import argparse
import csv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pybedtools

from utils import (
    genes_from_gencode,
    parse_yaml,
    time_decorator,
)


class EdgeParser:
    """Object to construct tensor based graphs from parsed bedfiles

    The baseline graph structure is build from the following in order:
        Curated protein-protein interactions from the integrated interactions
        database V 2021-05
        TF-gene circuits from Marbach et al.
        TF-gene interactions from TFMarker. We keep "TF" and "I Marker" type relationships
        Enhancer-gene networks from FENRIR
        Enhancer-enhancer networks from FENRIR

        Alternative polyadenylation targets from APAatlas

    Args:
        params: configuration vals from yaml

    Methods
    ----------
    _genes_from_gencode:
        Lorem
    _base_graph:
        Lorem
    _iid_ppi:
        Lorem
    _mirna_targets:
        Lorem
    _tf_markers:
        Lorem
    _marchbach_regulatory_circuits:
        Lorem
    _enhancer_index:
        Lorem
    _format_enhancer:
        Lorem
    _process_graph_edges:
        Lorem
    """

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
    ):
        """Initialize the class"""
        self.gencode = params["shared"]["gencode"]
        self.interaction_files = params["interaction"]
        self.tissue = params["resources"]["tissue"]
        self.tissue_name = params["resources"]["tissue_name"]
        self.marker_name = params["resources"]["marker_name"]
        self.ppi_tissue = params["resources"]["ppi_tissue"]
        self.tissue_specific = params["tissue_specific"]

        self.root_dir = params["dirs"]["root_dir"]
        self.shared_dir = f"{self.root_dir}/shared_data"
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
        self.parse_dir = f"{self.tissue_dir}/parsing"
        self.interaction_dir = f"{self.tissue_dir}/interaction"
        self.shared_interaction_dir = f"{self.shared_dir}/interaction"

        self.gencode_ref = pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        self.genesymbol_to_gencode = genes_from_gencode(gencode_ref=self.gencode_ref)
        self.gencode_attr_ref =  self._blind_read_file(
            f"{self.tissue_dir}/local/gencode_v26_node_attr.bed"
        )
        self.mirna_ref = self._blind_read_file(
            f"{self.interaction_dir}/{params['interaction']['mirdip']}"
        )
        self.enhancer_ref = self._blind_read_file(
            f"{self.tissue_dir}/local/enhancers_lifted.bed"
        )
        self.e_indexes = self._enhancer_index(
            e_index=f"{self.shared_interaction_dir}/enhancer_indexes.txt",
            e_index_unlifted=f"{self.shared_interaction_dir}/enhancer_indexes_unlifted.txt",
        )

    def _blind_read_file(self, file: str) -> List[str]:
        """Blindly reads a file into csv reader and stores file as a list of
        lists
        """
        return [line for line in csv.reader(open(file, newline=""), delimiter="\t")]

def place_holder_function():
    """_summary_ of function"""
    pass


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()