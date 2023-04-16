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
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pybedtools

from utils import dir_check_make, NODES, parse_yaml, time_decorator


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
        return [tup[3] for tup in a]

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
        add_tissue: bool = False,
    ) -> List[str]:
        """Get edges from file"""
        if edge_type == "base":
            if add_tissue:
                return [
                    (f"{tup[0]}_{self.tissue}", f"{tup[1]}_{self.tissue}")
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
            else:
                return [
                    (tup[0], tup[1])
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
        if edge_type == "local":
            if add_tissue:
                return [
                    (f"{tup[3]}_{self.tissue}", f"{tup[7]}_{self.tissue}")
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
            else:
                return [
                    (tup[3], tup[7])
                    for tup in csv.reader(open(edge_file, "r"), delimiter="\t")
                ]
        if edge_type not in ("base", "local"):
            raise ValueError("Edge type must be 'base' or 'local'")

    @time_decorator(print_args=False)
    def _prepare_reference_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Add polyadenylation to gencode ref dict used to fill in attributes.
        Base_node attr are hard coded in as the first type to load. There are
        duplicate keys in preprocessing but they have the same attributes so
        they'll overrwrite without issue.

        Returns:
            Dict[str, Dict[str, Any]]: nested dict of attributes for each node
        """
        ref = pickle.load(
            open(f"{self.parse_dir}/attributes/basenodes_reference.pkl", "rb")
        )
        for node in NODES:
            ref_for_concat = pickle.load(
                open(f"{self.parse_dir}/attributes/{node}_reference.pkl", "rb")
            )
            ref.update(ref_for_concat)

        for key in ref:
            if key in self.genes:
                ref[key]["is_gene"] = 1
                ref[key]["is_tf"] = 0
            elif "_tf" in key:
                ref[key]["is_gene"] = 0
                ref[key]["is_tf"] = 1
            else:
                ref[key]["is_gene"] = 0
                ref[key]["is_tf"] = 0

        return ref

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

    @time_decorator(print_args=True)
    def _nx_to_tensors(self, graph: nx.Graph) -> None:
        """Save graphs as np tensors, additionally saves a dictionary to map
        nodes to new integer labels

        Args:
            graph (nx.Graph)
        """
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        edges = nx.to_edgelist(graph)
        nodes = sorted(graph.nodes)

        with open(f"{self.graph_dir}/{self.tissue}_full_graph.pkl", "wb") as output:
            pickle.dump(
                {
                    "edge_index": np.array(
                        [[edge[0] for edge in edges], [edge[1] for edge in edges]]
                    ),
                    "node_feat": np.array(
                        [[val for val in graph.nodes[node].values()] for node in nodes]
                    ),
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "avg_edges": graph.number_of_edges() / graph.number_of_nodes(),
                },
                output,
            )

    def process_graphs(self) -> None:
        """_summary_"""
        # get edges
        base_edges = self._get_edges(
            edge_file=f"{self.interaction_dir}/interaction_edges.txt",
            edge_type="base",
            add_tissue=True,
        )

        local_context_edges = self._get_edges(
            edge_file=f"{self.parse_dir}/edges/all_concat_sorted.bed",
            edge_type="local",
            add_tissue=True,
        )

        # create graph
        graph = self._base_graph(edges=base_edges)

        # add local context edges
        graph.add_edges_from((tup[0], tup[1]) for tup in local_context_edges)

        # get attribute reference dictionary
        ref = self._prepare_reference_attributes()

        # save nx graph
        nx.write_gml(graph, f"{self.graph_dir}/{self.tissue}_full_graph.gml")

        # add attributes
        nx.set_node_attributes(graph, ref)

        # save dictionary of node to integer labels
        with open(f"{self.graph_dir}/{self.tissue}_gene_idxs.pkl", "wb") as output:
            pickle.dump(
                {node: idx for idx, node in enumerate(sorted(graph.nodes))},
                output,
            )

        # save individual graph
        self._nx_to_tensors(graph=graph)


def main() -> None:
    """Pipeline to generate individual graphs"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, help="Path to .yaml file with filenames")

    args = parser.parse_args()
    params = parse_yaml(args.config)

    # instantiate object
    graphconstructorObj = GraphConstructor(params=params)

    # process graphs
    graphconstructorObj.process_graphs()


if __name__ == "__main__":
    main()
