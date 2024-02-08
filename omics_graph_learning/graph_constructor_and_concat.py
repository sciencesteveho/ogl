#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] remove hardcoded files and add in the proper area, either the yaml or the processing fxn
# - [ ] add module to create local global nodes for genes

"""Create graphs from parsed edges

This script will create one graph per tissue from the parsed edges. All edges
are added before being filtered to only keep edges that can traverse back to a
base node. Attributes are then added for each node.
"""

import argparse
import csv
import pickle
from typing import Any, Dict, Generator, List

import networkx as nx
import numpy as np

import utils

NODES = [
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
]


def _base_graph(edges: List[str]):
    """Create a graph from list of edges"""
    graph = nx.Graph()
    for tup in edges:
        graph.add_edges_from([(tup[0], tup[1], {"edge_type": tup[2]})])
    return graph


@utils.time_decorator(print_args=True)
def _get_edges(
    edge_file: str,
    edge_type: str,
    add_tissue: bool = False,
    tissue: str = "",
) -> Generator[str, str, str]:
    """Get edges from file"""
    reader = csv.reader(open(edge_file, "r"), delimiter="\t")
    for tup in reader:
        if edge_type == "base":
            if add_tissue:
                yield (f"{tup[0]}_{tissue}", f"{tup[1]}_{tissue}", tup[3])
            else:
                yield (tup[0], tup[1], tup[3])
        elif edge_type == "local":
            if add_tissue:
                yield (f"{tup[3]}_{tissue}", f"{tup[7]}_{tissue}", "local")
            else:
                yield (tup[3], tup[7], "local")
        else:
            raise ValueError("Edge type must be 'base' or 'local'")


@utils.time_decorator(print_args=False)
def _prepare_reference_attributes(reference_dir: str) -> Dict[str, Dict[str, Any]]:
    """Base_node attr are hard coded in as the first type to load. There are
    duplicate keys in preprocessing but they have the same attributes so
    they'll overrwrite without issue.

    Returns:
        Dict[str, Dict[str, Any]]: nested dict of attributes for each node
    """
    ref = pickle.load(open(f"{reference_dir}/basenodes_reference.pkl", "rb"))
    for node in nodes:
        ref_for_concat = pickle.load(
            open(f"{reference_dir}/{node}_reference.pkl", "rb")
        )
        ref.update(ref_for_concat)

    for key in ref:
        if "_tf" in key:
            ref[key]["is_gene"] = 0
            ref[key]["is_tf"] = 1
        elif "ENSG" in key:
            ref[key]["is_gene"] = 1
            ref[key]["is_tf"] = 0
        else:
            ref[key]["is_gene"] = 0
            ref[key]["is_tf"] = 0
    return ref


@utils.time_decorator(print_args=True)
def graph_constructor(
    tissue: str,
    root_dir: str,
    nodes: List[str],
    graph_type: str,
) -> nx.Graph:
    """_summary_

    Args:
        tissue (str): _description_
        root_dir (str): _description_
        graph_dir (str): _description_
        interaction_dir (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        nx.Graph: _description_
    """
    # housekeeping
    interaction_dir = f"{root_dir}/{tissue}/interaction"
    parse_dir = f"{root_dir}/{tissue}/parsing"

    # get edges
    base_edges = _get_edges(
        edge_file=f"{interaction_dir}/interaction_edges.txt",
        edge_type="base",
        add_tissue=True,
    )

    local_context_edges = _get_edges(
        edge_file=f"{parse_dir}/edges/all_concat_sorted.bed",
        edge_type="local",
        add_tissue=True,
    )

    # get attribute reference dictionary
    ref = _prepare_reference_attributes(reference_dir=f"{parse_dir}/attributes/")

    # create graphs
    if graph_type == "full":
        graph = _base_graph(edges=base_edges)
        for tup in local_context_edges:  # add local context edges to full graph
            graph.add_edges_from([(tup[0], tup[1], {"edge_type": tup[2]})])
        nx.set_node_attributes(graph, ref)
        return graph
    else:
        graph = _base_graph(edges=base_edges)
        nx.set_node_attributes(graph, ref)
        return graph


@utils.time_decorator(print_args=True)
def _nx_to_tensors(
    prefix: str,
    graph_dir: str,
    graph: nx.Graph,
    graph_type: str,
) -> None:
    """Save graphs as np tensors, additionally saves a dictionary to map
    nodes to new integer labels

    Args:
        graph (nx.Graph)
    """
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    edges = nx.to_edgelist(graph)
    nodes = sorted(graph.nodes)

    with open(f"{graph_dir}/{prefix}_{graph_type}_graph.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": np.array(
                    [[edge[0] for edge in edges], [edge[1] for edge in edges]]
                ),
                "node_feat": np.array(
                    [list(graph.nodes[node].values()) for node in nodes]
                ),
                "edge_feat": [edge[2]["edge_type"] for edge in edges],
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "avg_edges": graph.number_of_edges() / graph.number_of_nodes(),
            },
            output,
            protocol=4,
        )


def main() -> None:
    """Pipeline to generate individual graphs"""
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_type",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    args = parser.parse_args()
    params = utils.parse_yaml(args.experiment_config)

    # set up variables for params to improve readability
    nodes = params["nodes"] + NODES if params["nodes"] is not None else NODES
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]

    # create primary graph directory
    root_dir = f"{working_directory}/{experiment_name}"
    graph_dir = f"{root_dir}/graphs"
    utils.dir_check_make(graph_dir)

    # instantiate objects and process graphs
    for idx, tissue in enumerate(params["tissues"]):
        if idx == 0:
            graph = graph_constructor(
                tissue=tissue,
                nodes=nodes,
                root_dir=root_dir,
                graph_type=args.graph_type,
            )
        else:
            current_graph = graph_constructor(
                tissue=tissue,
                nodes=nodes,
                root_dir=root_dir,
                graph_type=args.graph_type,
            )
            graph = nx.compose(graph, current_graph)

    # save indexes before renaming to integers
    with open(
        f"{graph_dir}/{experiment_name}_{args.graph_type}_graph_idxs.pkl", "wb"
    ) as output:
        pickle.dump(
            {node: idx for idx, node in enumerate(sorted(graph.nodes))},
            output,
        )

    # save idxs and write to tensors
    _nx_to_tensors(
        prefix=experiment_name,
        graph_dir=graph_dir,
        graph=graph,
        graph_type=args.graph_type,
    )


if __name__ == "__main__":
    main()
