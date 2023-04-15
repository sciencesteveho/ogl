#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 
#

"""Parse edges from interaction-type omics data"""

import argparse
import csv
from itertools import repeat
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

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
    _fenrir_enhancer_enhancer:
        Lorem
    _fenrir_enhancer_gene:
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
            f"{self.interaction_dir}/{self.tissue}_mirdip"
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
    
    @time_decorator(print_args=True)
    def _iid_ppi(
        self,
        interaction_file: str,
        tissue: str,
        ) -> List[Tuple[str, str, float, str]]:
        """Protein-protein interactions from the Integrated Interactions
        Database v 2021-05"""
        df = pd.read_csv(interaction_file, delimiter='\t')
        df = df[['symbol1', 'symbol2', 'evidence_type', 'n_methods', tissue]]
        t_spec_filtered = df[
            (df[tissue] > 0)
            & (df['n_methods'] >= 3)
            & (df['evidence_type'].str.contains('exp'))
            ]
        edges = list(
                zip(*map(t_spec_filtered.get, ['symbol1', 'symbol2']), repeat(-1), repeat('ppi'))
                )
        return [
            (
                f'{self.genesymbol_to_gencode[edge[0]]}',
                f'{self.genesymbol_to_gencode[edge[1]]}',
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
        tissue_active_mirnas: str
        ) -> List[Tuple[str, str]]:
        """Filters all miRNA -> target interactions from miRTarBase and only
        keeps the miRNAs that are active in the given tissue from mirDIP.
        """
        active_mirna = [
            line[3]
            for line in csv.reader(open(tissue_active_mirnas, newline=""), delimiter="\t")
        ]

        return [
            (
                line[0],
                self.genesymbol_to_gencode[line[1]],
                -1,
                "mirna",
            )
            for line in csv.reader(open(target_list, newline=""), delimiter="\t")
            if line[0] in active_mirna and line[1] in self.genesymbol_to_gencode.keys()
        ]
    
    @time_decorator(print_args=True)
    def _tf_markers(self, interaction_file: str) -> List[Tuple[str, str]]:
        tf_keep = ["TF", "I Marker", "TFMarker"]
        tf_markers = []
        with open(interaction_file, newline="") as file:
            file_reader = csv.reader(file, delimiter="\t")
            next(file_reader)
            for line in file_reader:
                if line[2] in tf_keep and line[5] == self.marker_name:
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

    def _enhancer_index(
        self,
        e_index: str, 
        e_index_unlifted: str
        ) -> Dict[str, str]:
        """Returns a dict to map enhancers from hg19 to hg38"""
        def text_to_dict(txt, idx1, idx2):
            with open(txt) as file:
                file_reader = csv.reader(file, delimiter='\t')
                return {
                    line[idx1]:line[idx2]
                    for line in file_reader
                }
        e_dict = text_to_dict(e_index, 1, 0)
        e_dict_unlifted = text_to_dict(e_index_unlifted, 0, 1)
        e_dict_unfiltered = {
            enhancer:e_dict[e_dict_unlifted[enhancer]]
            for enhancer in e_dict_unlifted
            if e_dict_unlifted[enhancer] in e_dict.keys()
            }
        return {
            k:v for k,v in e_dict_unfiltered.items()
            if 'alt' not in v
            }

    def _format_enhancer(
        self,
        input: str,
        index: int,
        ) -> str:
        return f"{input.replace(':', '-').split('-')[index]}"

    @time_decorator(print_args=True)
    def _fenrir_enhancer_enhancer(
        self,
        interaction_file: str,
        score_filter: int,
        ) -> List[Tuple[str, str, float, str]]:
        """Convert each enhancer-enhancer link to hg38 and return a formatted
        tuple."""
        e_e_liftover, scores = [], []
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            for line in file_reader:
                scores.append(int(line[2]))
                if line[0] in self.e_indexes.keys() and line[1] in self.e_indexes.keys():
                    e_e_liftover.append((self.e_indexes[line[0]], self.e_indexes[line[1]], line[2]))

        cutoff = np.percentile(scores, score_filter)
        return [
            (f"enhancer_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
            f"enhancer_{self._format_enhancer(line[1], 0)}_{self._format_enhancer(line[1], 1)}",
            -1,
            'enhancer-enhancer',)
            for line in e_e_liftover
            if int(line[2]) >= cutoff 
        ]

    @time_decorator(print_args=True)
    def _fenrir_enhancer_gene(
        self,
        interaction_file: str,
        score_filter: int,
        ) -> List[Tuple[str, str, float, str]]:
        """Convert each enhancer-gene link to hg38 and ensemble ID, return a
        formatted tuple.
        """
        e_g_liftover, scores = [], []
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            for line in file_reader:
                scores.append(int(line[3]))
                if line[0] in self.e_indexes.keys() and line[2] in self.genesymbol_to_gencode.keys():
                    e_g_liftover.append((self.e_indexes[line[0]], self.genesymbol_to_gencode[line[2]], line[3]))

        cutoff = np.percentile(scores, score_filter)
        return [
            (f"enhancer_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
            line[1],
            -1,
            'enhancer-gene')
            for line in e_g_liftover
            if int(line[2]) >= cutoff
        ]

    @time_decorator(print_args=True)
    def _process_graph_edges(self) -> None:
        """Retrieve all interaction edges and saves them to a text file.
        Edges will be loaded from the text file for subsequent runs to save
        processing time.
        
        Returns:
            A list of all edges
        """
        ppi_edges = self._iid_ppi(
            interaction_file=f"{self.interaction_dir}/{self.interaction_files['ppis']}",
            tissue=self.ppi_tissue,
        )
        mirna_targets = self._mirna_targets(
            target_list=f"{self.interaction_dir}/{self.interaction_files['mirnatargets']}",
            tissue_active_mirnas=f"{self.interaction_dir}/{self.interaction_files['mirdip']}",
        )
        tf_markers = self._tf_markers(
            interaction_file=f"{self.interaction_dir}/{self.interaction_files['tf_marker']}",
        )
        e_e_edges = self._fenrir_enhancer_enhancer(
            f"{self.interaction_dir}" f"/{self.tissue_specific['enhancers_e_e']}",
            score_filter=30,
        )
        e_g_edges = self._fenrir_enhancer_gene(
            f"{self.interaction_dir}" f"/{self.tissue_specific['enhancers_e_g']}",
            score_filter=70,
        )
        circuit_edges = self._marbach_regulatory_circuits(
            f"{self.interaction_dir}" f"/{self.interaction_files['circuits']}",
            score_filter=80
        )

        self.edges = (
            ppi_edges + mirna_targets + tf_markers + e_e_edges + e_g_edges + circuit_edges
        )
        
        gencode_nodes = (
            [tup[0] for tup in ppi_edges]
            + [tup[1] for tup in ppi_edges]
            + [tup[1] for tup in mirna_targets]
            + [tup[0] for tup in tf_markers]
            + [tup[1] for tup in tf_markers]
            + [tup[1] for tup in e_g_edges]
            + [tup[0] for tup in circuit_edges]
            + [tup[1] for tup in circuit_edges]
        )

        enhancers = (
            [tup[0] for tup in e_e_edges]
            + [tup[1] for tup in e_e_edges]
            + [tup[0] for tup in e_g_edges]
        )

        mirnas = [tup[0] for tup in mirna_targets]

        return set(gencode_nodes), set(enhancers), set(mirnas)
    
    @time_decorator(print_args=False)
    def _add_node_coordinates(
        self,
        nodes,
        node_ref,
    ) -> None:
        """_summary_

        Args:
            nodes:
            node_ref:
        """
        return [
            line[0:4] for line in node_ref if line[3] in set(nodes)
        ]

    @time_decorator(print_args=True)
    def parse_edges(self) -> None:
        """Constructs tissue-specific interaction base graph"""

        # retrieve interaction-based edges
        gencode_nodes, enhancers, mirnas = self._process_graph_edges()

        # add coordinates to nodes in parallel
        pool = Pool(processes=3)
        nodes_for_attr = pool.starmap(
            self._add_node_coordinates,
            list(
                zip(
                    [gencode_nodes, enhancers, mirnas],
                    [self.gencode_attr_ref, self.enhancer_ref, self.mirna_ref],
                )
            ),
        )
        pool.close()
        nodes_for_attr = sum(nodes_for_attr, [])  # flatten list of lists

        # add coordinates to edges
        full_edges = []
        nodes_with_coords = {node[3]: node[0:3] for node in nodes_for_attr}
        for edge in self.edges:
            if edge[0] in nodes_with_coords and edge[1] in nodes_with_coords:
                full_edges.append(
                    [edge[0]]
                    + nodes_with_coords[edge[0]]
                    + [edge[1]]
                    + nodes_with_coords[edge[1]]
                    + [edge[2], edge[3]]
                )

        # write edges to file
        all_interaction_file = f"{self.interaction_dir}/interaction_edges.txt"
        with open(all_interaction_file, "w+") as output:
            csv.writer(output, delimiter="\t").writerows(self.edges)
        
        # write nodes to file
        with open(f"{self.tissue_dir}/local/basenodes_hg38.txt", 'w+') as output:
            csv.writer(output, delimiter="\t").writerows(nodes_for_attr)
        
        # write edges with coordinates to file
        with open(f"{self.interaction_dir}/full_edges.txt", 'w+') as output:
            csv.writer(output, delimiter="\t").writerows(full_edges)

def main() -> None:
    """Pipeline to generate individual graphs"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to .yaml file with filenames"
    )

    args = parser.parse_args()
    params = parse_yaml(args.config)
    
    # instantiate object
    edgeparserObject = EdgeParser(
        params=params,
        )

    # run pipeline!
    edgeparserObject.parse_edges()


if __name__ == '__main__':
    main()