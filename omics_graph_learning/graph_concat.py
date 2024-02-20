#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] remove hardcoded files and add in the proper area, either the yaml or the processing fxn
# - [ ] add module to create local global nodes for genes

"""Concatenate multiple graphs into a single graph and save their respective
indexes"""

import argparse
import pathlib
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np

import utils


def _open_graph_and_idxs(prefix: str) -> Tuple[Dict, Dict]:
    """Open graph and idxs from file."""
    with open(f"{prefix}.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(f"{prefix}_idxs.pkl", "rb") as f:
        idxs = pickle.load(f)
    return graph, idxs


def _reindex_idxs(idxs: Dict[str, int], new_start_idx: int) -> Dict[str, int]:
    """Reindex nodes to integers by adding the start_idx to each node."""
    return {key: value + new_start_idx for key, value in idxs.items()}


def _reindex_edges(edges: np.ndarray, new_start_idx: int) -> np.ndarray:
    """Reindex edges to integers by adding the start_idx to each node."""
    edges += new_start_idx
    return edges


def concatenate_graphs(
    pre_prefix: pathlib.PosixPath,
    tissues: List[str],
) -> None:
    """Concatenate multiple graphs into a single graph. The first tissue in the
    list is used as the base and all subsequent graphs are reindexed then
    concatenated.

    Args:
        pre_prefix (str): The prefix of the graph file names.
        tissues (List[str]): The list of tissue names.

    Returns:
        None
    """
    prefix = f"{pre_prefix}_{tissues[0]}"
    concat_graph, concat_idxs = _open_graph_and_idxs(prefix)
    concat_edges = concat_graph["edge_index"]
    concat_edge_feat = concat_graph["edge_feat"]
    concat_node_feat = concat_graph["node_feat"]
    concat_num_nodes = concat_graph["num_nodes"]
    concat_num_edges = concat_graph["num_edges"]

    for tissue in tissues[1:]:
        prefix = f"{pre_prefix}_{tissue}"
        graph, idxs = _open_graph_and_idxs(prefix)
        end_idx = max(concat_idxs.values()) + 1
        concat_idxs.update(_reindex_idxs(idxs=idxs, new_start_idx=end_idx))
        concat_edges = np.hstack(
            (
                concat_edges,
                _reindex_edges(edges=graph["edge_index"], new_start_idx=end_idx),
            )
        )
        concat_edge_feat = np.concatenate((concat_edge_feat, graph["edge_feat"]))
        concat_node_feat = np.concatenate((concat_node_feat, graph["node_feat"]))
        concat_num_nodes += graph["num_nodes"]
        concat_num_edges += graph["num_edges"]

    with open(f"{pre_prefix}.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": concat_edges,
                "node_feat": concat_node_feat,
                "edge_feat": concat_edge_feat,
                "num_nodes": concat_num_nodes,
                "num_edges": concat_num_edges,
                "avg_edges": concat_num_edges / concat_num_nodes,
            },
            output,
            protocol=4,
        )

    with open(f"{pre_prefix}_idxs.pkl", "wb") as output:
        pickle.dump(concat_idxs, output, protocol=4)


def main() -> None:
    """Pipeline to concatenate tissue graphs"""
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

    # set up variables
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]
    working_dir = pathlib.Path(working_directory)
    graph_dir = working_dir / experiment_name / "graphs"
    pre_prefix = graph_dir / f"{experiment_name}_{args.graph_type}_graph"

    # concat all graphs! and save to file
    concatenate_graphs(pre_prefix=pre_prefix, tissues=params["tissues"])


if __name__ == "__main__":
    main()
