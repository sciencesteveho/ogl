#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 
#

"""Create graphs from parsed edges

This script will create one graph per tissue from the parsed edges. All edges
are added before being filtered to only keep edges that can traverse back to a
base node. Attributes are then added for each node.
"""

import csv
from typing import Any, Dict, List, Tuple

import networkx as nx
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

    ATTRIBUTES = [
        "cnv",
        "cpg",
        "ctcf",
        "dnase",
        "gc",
        "h3k27ac",
        "h3k27me3",
        "h3k36me3",
        "h3k4me1",
        "h3k4me3",
        "h3k9me3",
        "indels",
        "line",
        "ltr",
        "microsatellites",
        "phastcons",
        "polr2a",
        "rbpbindingsites",
        "recombination",
        "repg1b",
        "repg2",
        "reps1",
        "reps2",
        "reps3",
        "reps4",
        "rnarepeat",
        "simplerepeats",
        "sine",
        "snp",
    ]

    NODES = [
        "chromatinloops",
        "cpgislands",
        "ctcfccre",
        "enhancers",
        "gencode",
        "histones",
        "polyasites",
        "promoters",
        "superenhancers",
        "tads",
        "tfbindingclusters",
        "tss",
    ]

    NODE_FEATS = ["start", "end", "size", "gc"] + ATTRIBUTES

    ONEHOT_EDGETYPE = {
        'local': [1,0,0,0,0],
        'enhancer-enhancer': [0,1,0,0,0],
        'enhancer-gene': [0,0,1,0,0],
        'circuits': [0,0,0,1,0],
        'ppi': [0,0,0,0,1],
    }

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
        ):
        """Initialize the class"""
        self.tissue_name = params["resources"]["tissue_name"]

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

    def _base_graph(self, edges: List[str]):
        """Create a graph from list of edges"""
        G = nx.Graph()
        G.add_edges_from((tup[0], tup[1]) for tup in edges)
        return G 
    
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
        
    def _base_node_traversals(self, base_graph: nx.Graph) -> List[str]:
        """Get edges that can traverse back to a base node"""
        edges = []
        for edge in base_graph.edges:
            if edge[0] in self.NODES:
                edges.append(edge)
            elif edge[1] in self.NODES:
                edges.append(edge)
        return edges
    
    def _n_ego_graph(self, base_graph: nx.Graph, n: int) -> nx.Graph:
        """Get n-ego graph"""
        # get base node edges
        base_node_edges = self._base_node_traversals(base_graph=base_graph)
        # get n-ego graph
        n_ego_graph = nx.ego_graph(base_graph, n=n, center=False)
        # add base node edges
        n_ego_graph.add_edges_from(base_node_edges)
        return n_ego_graph
    
    def _gene_subgraphs(self, base_graph: nx.Graph) -> List[nx.Graph]:
        """Get gene subgraphs"""
        # get gene nodes
        gene_nodes = [node for node in base_graph.nodes if node in self.genesymbol_to_gencode]
        # get gene subgraphs
        gene_subgraphs = [base_graph.subgraph(c).copy() for c in nx.connected_components(base_graph) if c & set(gene_nodes)]
        return gene_subgraphs

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
        base_graph = self._base_graph(edges=base_edges)
        base_graph.add_edges_from((tup[0], tup[1]) for tup in local_context_edges)

        # add attributes
        self._add_attributes(base_graph=base_graph)
        # write graph
        nx.write_gml(base_graph, f"{self.graph_dir}/{self.tissue_name}_graph.gml")


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