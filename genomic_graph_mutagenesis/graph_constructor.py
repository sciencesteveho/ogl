#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 
#

"""Parse edges from interaction-type omics data"""

import argparse
import csv
import os
from itertools import repeat
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pybedtools

from utils import (
    dir_check_make,
    genes_from_gencode,
    parse_yaml,
    time_decorator,
)


class GraphConstructor:
    """Object to construct tensor based graphs from parsed edges

    Args:
        params: configuration vals from yaml

    Methods
    ----------
    _genes_from_gencode:
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
        self.graph_dir = f"{self.parse_dir}/graphs"
        dir_check_make(self.graph_dir)

        self.gencode_ref = pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        self.genesymbol_to_gencode = genes_from_gencode(gencode_ref=self.gencode_ref)
        self.gencode_attr_ref =  self._blind_read_file(
            f"{self.tissue_dir}/local/gencode_v26_node_attr.bed"
        )
        self.mirna_ref = self._blind_read_file(
            f"{self.interaction_dir}/{self.tissue}_mirdip"
        )
        self.enhancer_ref = self._blind_read_file(
            f"{self.tissue_dir}/local/enhancers_lifted.bed"
        )
        self.e_indexes = self._enhancer_index(
            e_index=f"{self.shared_interaction_dir}/enhancer_indexes.txt",
            e_index_unlifted=f"{self.shared_interaction_dir}/enhancer_indexes_unlifted.txt",
        )


                

def main() -> None:
    """Pipeline to generate individual graphs"""
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )

    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     help="Path to .yaml file with filenames"
    # )

    # args = parser.parse_args()
    # params = parse_yaml(args.config)
    
    # # instantiate object
    # edgeparserObject = EdgeParser(
    #     params=params,
    #     )

    # # run pipeline!
    # edgeparserObject.parse_edges()


if __name__ == '__main__':
    main()