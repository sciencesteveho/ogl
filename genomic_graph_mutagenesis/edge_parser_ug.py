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
import os
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

    PROTEIN_TISSUES = [
        "amygdala",
        "bone",
        "bone marrow",
        "brain",
        "heart",
        "hypothalamus",
        "kidney",
        "liver",
        "lung",
        "lymph nodes",
        "mammary gland",
        "ovary",
        "pancreas",
        "pituitary gland",
        "placenta",
        "prostate",
        "salivary gland",
        "skeletal muscle",
        "small intestine",
        "spleen",
        "stomach",
        "testes",
        "uterus",
        "synovial macrophages",
        "chondrocytes",
        "synovial membrane",
        "articular cartilage",
    ]

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
    ):
        """Initialize the class"""
        self.tissue_name = "universalgenome"

        self.gencode = params["shared"]["gencode"]
        self.interaction_files = params["interaction"]
        self.tissue_specific = params["tissue_specific"]

        self.root_dir = params["dirs"]["root_dir"]
        self.circuit_dir = params["dirs"]["circuit_dir"]
        self.shared_dir = f"{self.root_dir}/shared_data"
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
        self.parse_dir = f"{self.tissue_dir}/parsing"
        self.interaction_dir = f"{self.tissue_dir}/interaction"
        self.shared_interaction_dir = f"{self.shared_dir}/interaction"

        self.gencode_ref = pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        self.genesymbol_to_gencode = genes_from_gencode(gencode_ref=self.gencode_ref)
        self.gencode_attr_ref = self._blind_read_file(
            f"{self.tissue_dir}/local/gencode_v26_node_attr.bed"
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

    @time_decorator(print_args=True)
    def _iid_ppi(
        self,
        interaction_file: str,
        tissue: str,
    ) -> List[Tuple[str, str, float, str]]:
        """Protein-protein interactions from the Integrated Interactions
        Database v 2021-05"""
        df = pd.read_csv(interaction_file, delimiter="\t")
        df = df[["symbol1", "symbol2", "evidence_type", "n_methods", tissue]]
        t_spec_filtered = df[
            (df[tissue] > 0)
            & (df["n_methods"] >= 4)
            & (df["evidence_type"].str.contains("exp"))
        ]
        edges = list(
            zip(
                *map(t_spec_filtered.get, ["symbol1", "symbol2"]),
                repeat(-1),
                repeat("ppi"),
            )
        )
        return [
            (
                f"{self.genesymbol_to_gencode[edge[0]]}",
                f"{self.genesymbol_to_gencode[edge[1]]}",
                edge[2],
                edge[3],
            )
            for edge in edges
            if edge[0] in self.genesymbol_to_gencode.keys()
            and edge[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _mirna_targets(
        self,
        target_list: str,
    ) -> List[Tuple[str, str]]:
        """Filters all miRNA -> target interactions from miRTarBase and only
        keeps the miRNAs that are active in the given tissue from mirDIP.
        """
        return [
            (
                line[0],
                self.genesymbol_to_gencode[line[1]],
                -1,
                "mirna",
            )
            for line in csv.reader(open(target_list, newline=""), delimiter="\t")
            if line[1] in self.genesymbol_to_gencode.keys()
        ]
        
    @time_decorator(print_args=True)
    def _tf_markers(self, interaction_file: str) -> List[Tuple[str, str]]:
        tf_keep = ["TF", "I Marker", "TFMarker"]
        tf_markers = []
        with open(interaction_file, newline="") as file:
            file_reader = csv.reader(file, delimiter="\t")
            next(file_reader)
            for line in file_reader:
                if line[2] in tf_keep:
                    try:
                        if ";" in line[10]:
                            genes = line[10].split(";")
                            for gene in genes:
                                if line[2] == 'I Marker':
                                    tf_markers.append((gene, line[1]))
                                else:
                                    tf_markers.append((line[1], gene))
                        else:
                            if line[2] == 'I Marker':
                                tf_markers.append((line[10], line[1]))
                            else:
                                tf_markers.append((line[1], line[10]))
                    except IndexError: 
                        pass 

        return [
            (
                f'{self.genesymbol_to_gencode[tup[0]]}_tf',
                self.genesymbol_to_gencode[tup[1]],
                -1,
                "tf_marker",
            )
            for tup in tf_markers
            if tup[0] in self.genesymbol_to_gencode.keys()
            and tup[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _marbach_regulatory_circuits(
        self,
        interaction_file: str,
        score_filter: int,
        ) -> List[Tuple[str, str, float, str]]:
        """Regulatory circuits from Marbach et al., Nature Methods, 2016. Each
        network is in the following format:
            col_1   TF
            col_2   Target gene
            col_3   Edge weight 
        """
        tf_g, scores = [], []
        with open(interaction_file, newline = '') as file:
            file_reader = csv.reader(file, delimiter='\t')
            for line in file_reader:
                scores.append(float(line[2]))
                if line[0] in self.genesymbol_to_gencode.keys() and line[1] in self.genesymbol_to_gencode.keys():
                    tf_g.append((line[0], line[1], float(line[2])))
        
        cutoff = np.percentile(scores, score_filter)

        return [
            (f'{self.genesymbol_to_gencode[line[0]]}_tf',
            self.genesymbol_to_gencode[line[1]],
            line[2],
            'circuits',)
            for line in tf_g
            if line[2] >= cutoff
            ]
        
    def _load_tss(self, tss_path: str) -> pybedtools.BedTool:
        """Load TSS file and ignore any TSS that do not have a gene target.
        Returns:
            pybedtools.BedTool - TSS w/ target genes
        """
        tss = pybedtools.BedTool(tss_path)
        return tss.filter(lambda x: x[3].split("_")[3] != "").saveas()
    
    @time_decorator(print_args=True)
    def _process_graph_edges(self) -> None:
        """_summary_ of function"""
        ppi_edges = [
            self.iid_ppi(
                f"{self.interaction_dir}/{self.interaction_files['ppis']}", tissue
            )
            for tissue in self.PROTEIN_TISSUES
        ]
        mirna_targets = self._mirna_targets(target_list=f"{self.interaction_dir}/{self.interaction_files['mirnatargets']}")
        tf_markers = self._tf_markers(
            interaction_file=f"{self.interaction_dir}/{self.interaction_files['tf_marker']}",
        )
        circuit_edges = [self._marbach_regulatory_circuits(interaction_file=f"{self.circuit_dir}/{file}", score_filter=80,) for file in os.listdir(self.circuit_dir)]
        

def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
