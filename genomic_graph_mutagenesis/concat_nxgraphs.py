#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Create concatenated NX graph for all tissues"""

import argparse
import pickle

import networkx as nx
import numpy as np

from utils import _concat_nx_graphs


def main() -> None:
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g',
        '--graph_type',
        type=str,
        help='Graph type to use (full or base)',
    )
    args = parser.parse_args()

    graph_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs"
    graph = _concat_nx_graphs(
        tissue_list=[
            "hippocampus",
            "left_ventricle",
            "liver",
            "lung",
            "mammary",
            "pancreas",
            "skeletal_muscle",
        ],
        graph_dir=graph_dir,
        graph_type=args.graph_type,
    )

    nodes = sorted(graph.nodes)

    # save indexes before renaming to integers
    with open(f"{graph_dir}/all_tissue_{args.graph_type}_graph_idxs.pkl", "wb") as output:
        pickle.dump(
            {node: idx for idx, node in enumerate(nodes)},
            output,
        )

    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    edges = nx.to_edgelist(graph)
    edge_index = np.array([[edge[0] for edge in edges], [edge[1] for edge in edges]])
    node_feat = np.array([[val for val in graph.nodes[node].values()] for node in nodes])

    with open(f"{graph_dir}/all_tissue_{args.graph_type}_graph.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": edge_index,
                "node_feat": node_feat,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "avg_edges": graph.number_of_edges() / graph.number_of_nodes(),
            },
            output,
        )

if __name__ == "__main__":
        main()