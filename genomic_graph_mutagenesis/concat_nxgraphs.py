#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Create concatenated NX graph for all tissues"""

import argparse
import pickle

import networkx as nx
import numpy as np

from utils import TISSUES


def _concat_nx_graphs(tissue_list, graph_dir, graph_type):
    """_summary_

    Args:
        tissue_list (str): _description_
    """
    graph_list = []
    for tissue in tissue_list:
        graph_list.append(
            nx.read_gml(f"{graph_dir}/{tissue}/{tissue}_{graph_type}_graph.gml")
        )

    return nx.compose_all(graph_list)


def main(graph_dir: str) -> None:
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--graph_type",
        type=str,
        help="Graph type to use (full or base)",
    )
    args = parser.parse_args()

    graph = _concat_nx_graphs(
        tissue_list=TISSUES,
        graph_dir=graph_dir,
        graph_type=args.graph_type,
    )

    # save indexes before renaming to integers
    with open(
        f"{graph_dir}/all_tissue_{args.graph_type}_graph_idxs.pkl", "wb"
    ) as output:
        pickle.dump(
            {node: idx for idx, node in enumerate(sorted(graph.nodes))},
            output,
        )

    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    edges = nx.to_edgelist(graph)
    nodes = sorted(graph.nodes)

    with open(f"{graph_dir}/all_tissue_{args.graph_type}_graph.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": np.array(
                    [[edge[0] for edge in edges], [edge[1] for edge in edges]]
                ),
                "node_feat": np.array(
                    [[val for val in graph.nodes[node].values()] for node in nodes]
                ),
                "edge_feat": np.array([edge[2]["edge_type"] for edge in edges]),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "avg_edges": graph.number_of_edges() / graph.number_of_nodes(),
            },
            output,
        )


if __name__ == "__main__":
    main(graph_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs")
