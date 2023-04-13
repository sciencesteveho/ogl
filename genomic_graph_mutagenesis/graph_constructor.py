#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] remove hardcoded files and add in the proper area, either the yaml or the processing fxn
#

"""Create graphs from parsed edges

This script will create one graph per tissue from the parsed edges. All edges
are added before being filtered to only keep edges that can traverse back to a
base node. Attributes are then added for each node.
"""

import argparse
import csv
from itertools import repeat
from multiprocessing import Pool
import pickle
from typing import Any, Dict, List, Tuple

import networkx as nx
import pybedtools

from utils import dir_check_make, parse_yaml, time_decorator



class GraphConstructor:
    """Object to construct tensor based graphs from parsed edges

    Args:
        params: configuration vals from xwxwyaml

    Methods
    ----------
    _genes_from_gencode:
        Lorem

    # start end size gc cnv cpg ctcf dnase h3k27ac h3k27me3 h3k36me3 h3k4me1 h3k4me3 h3k9me3 indels line ltr microsatellites phastcons polr2a rbpbindingsites recombination repg1b repg2 reps1 reps2 reps3 reps4 rnarepeat simplerepeats sine snp polyadenylation
    """

    NODES = [
        "chromatinloops",
        "cpgislands",
        "ctcfccre",
        "enhancers",
        "gencode",
        "histones",
        "promoters",
        "superenhancers",
        "tads",
        "tfbindingclusters",
        "tss",
    ]

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
        ):
        """Initialize the class"""
        self.tissue = params["resources"]["tissue"]

        self.root_dir = params["dirs"]["root_dir"]
        self.shared_dir = f"{self.root_dir}/shared_data"
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
        self.parse_dir = f"{self.tissue_dir}/parsing"
        self.interaction_dir = f"{self.tissue_dir}/interaction"
        self.graph_dir = f"{self.root_dir}/graphs/{self.tissue}"
        dir_check_make(self.graph_dir)

        self.genes = self._filtered_genes(f"{self.tissue_dir}/tpm_filtered_genes.bed")

    def _filtered_genes(self, gencode_ref: str) -> List[str]:
        """Get genes from gencode ref that pass TPM filter"""
        a = pybedtools.BedTool(gencode_ref)
        return [
            tup[3] for tup in a
        ]
    
    def _base_graph(self, edges: List[str]):
        """Create a graph from list of edges"""
        G = nx.Graph()
        G.add_edges_from((tup[0], tup[1]) for tup in edges)
        return G 
    
    @time_decorator(print_args=True)
    def _get_edges(
        self,
        edge_file: str,
        edge_type: str,
        ) -> List[str]:
        """Get edges from file"""
        if edge_type == "base":
            return [
                (tup[0], tup[1]) for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
            ]
        if edge_type == "local":
            return [
                (tup[3], tup[7]) for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
            ]
        if edge_type not in ("base", "local"):
            raise ValueError("Edge type must be 'base' or 'local'")
        
    @time_decorator(print_args=False)
    def _prepare_reference_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Add polyadenylation to gencode ref dict used to fill in attributes.
        Base_node attr are hard coded in as the first type to load. There are
        duplicate keys in preprocessing but they have the same attributes so
        they'll overrwrite without issue."""
        ref = pickle.load(open(f'{self.parse_dir}/attributes/base_nodes_reference.pkl', 'rb'))
        for node in self.NODES:
            ref_for_concat = pickle.load(
                open(f'{self.parse_dir}/attributes/{node}_reference.pkl', 'rb')
            )
            ref.update(ref_for_concat)
        return ref
    
    @time_decorator(print_args=False)
    def _add_attributes(
        self, 
        graph: nx.Graph,
        attr_ref: Dict[str, Dict[str, Any]],
        ) -> nx.Graph:
        """Add attributes to graph nodes"""
        nx.set_node_attributes(graph, attr_ref)
        return graph
    
    @time_decorator(print_args=True)
    def _n_ego_graph(
        self,
        graph: nx.Graph,
        max_nodes: int,
        node: str,
        radius: int,
        ) -> nx.Graph:
        """Get n-ego graph centered around a gene (node)"""
        # get n-ego graph
        n_ego_graph = nx.ego_graph(
            graph=graph,
            n=node,
            radius=radius,
            undirected=True,
        )

        # if n_ego_graph is too big, reduce radius until n_ego_graph has nodes < max_nodes
        while n_ego_graph.number_of_nodes() > max_nodes:    
            radius -= 1
            n_ego_graph = nx.ego_graph(
                graph=graph,
                n=node,
                radius=radius,
            )

        return n_ego_graph
    
    def process_graphs(self) -> None:
        """_summary_
        """
        # get edges
        base_edges = self._get_edges(
            edge_file=f"{self.interaction_dir}/interaction_edges.txt",
            edge_type="base",
        )

        local_context_edges = self._get_edges(
            edge_file=f"{self.parse_dir}/edges/all_concat_sorted.bed",
            edge_type="local",
        )

        # create graph
        graph = self._base_graph(edges=base_edges)

        # add local context edges
        graph.add_edges_from((tup[0], tup[1]) for tup in local_context_edges)

        # get attribute reference dictionary
        ref = self._prepare_reference_attributes()

        # add attributes
        graph = self._add_attributes(
            graph=graph,
            attr_ref=ref,
        )

        # get n-ego graphs in parallel
        # set params
        max_nodes = 1000
        radius = 2

        pool = Pool(processes=32)
        pool.starmap(
            self._n_ego_graph,
            zip(repeat(graph),
                repeat(max_nodes),
                self.genes,
                repeat(radius),
            )
        )
        pool.close()

        # save individual graph AND ego graph


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
    graphconstructorObj = GraphConstructor(params=params)

    # process graphs
    graphconstructorObj.process_graphs()


if __name__ == '__main__':
    main()