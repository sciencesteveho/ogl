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


def _reindex_idxs(
    idxs: Dict[str, int],
    new_start_idx: int,
) -> Dict[str, int]:
    """Reindex nodes to integers by adding the start_idx to each node. Adds 1,
    since lists are 0-indexed."""
    # reindex nodes
    for key, values in idxs.items():
        idxs[key] = values + new_start_idx + 1
    return idxs


def _reindex_edges(edges: np.ndarray, new_start_idx: int) -> np.ndarray:
    """Reindex edges to integers by adding the start_idx to each node."""
    return np.array([edges[0] + new_start_idx, edges[1] + new_start_idx])


def _renindex_graph(
    graph: Dict[str, Union[float, int, List[str], np.array]],
) -> Dict[str, Union[float, int, List[str], np.array]]:
    """Reindex nodes to integers by adding the start_idx to each node."""


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
    root_dir = f"{working_directory}/{experiment_name}"
    graph_dir = f"{root_dir}/graphs"

    pre_prefix = f"{graph_dir}/{experiment_name}_{args.graph_type}_graph"
    for idx, tissue in enumerate(params["tissues"]):
        prefix = f"{pre_prefix}_{tissue}"
        graph, idxs = _open_graph_and_idxs(prefix)
        if idx == 0:
            end_idx = _get_idx_max_min(idxs)[0]
            concat_idxs = idxs
            concat_edges = graph["edge_index"]
            concat_edge_feat = graph["edge_feat"]
            concat_node_feat = graph["node_feat"]
            concat_num_nodes = graph["num_nodes"]
            concat_num_edges = graph["num_edges"]
        else:
            graph, idxs = _open_graph_and_idxs(prefix)
            concat_idxs = {**concat_idxs, **_reindex_idxs(idxs, end_idx)}
            concat_edges = np.hstack(
                (concat_edges, _reindex_edges(graph["edge_index"], end_idx))
            )
            concat_edge_feat = np.concatenate((concat_edge_feat, graph["edge_feat"]))
            concat_node_feat = np.concatenate((concat_node_feat, graph["node_feat"]))
            concat_num_nodes += graph["num_nodes"]
            concat_num_edges += graph["num_edges"]

    with open(f"{prefix}.pkl", "wb") as output:
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


if __name__ == "__main__":
    main()


# graph, idxs = _open_graph_and_idxs(
#     "regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_gte2_full_graph_aorta"
# )

# end_idx = max(idxs.values())
# concat_idxs = idxs
# concat_edges = graph["edge_index"]
# concat_edge_feat = graph["edge_feat"]
# concat_node_feat = graph["node_feat"]
# concat_num_nodes = graph["num_nodes"]
# concat_num_edges = graph["num_edges"]

# graph, idxs = _open_graph_and_idxs(
#     "regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_gte2_full_graph_liver"
# )

# # idxs["ENSG00000067840.12_liver"]

# concat_idxs = {**concat_idxs, **_reindex_idxs(idxs=idxs, new_start_idx=end_idx)}
# concat_edges = np.hstack((concat_edges, _reindex_edges(graph["edge_index"], end_idx)))

# len(concat_edges[0])
# len(concat_edges[1])
# concat_edge_feat = np.concatenate((concat_edge_feat, graph["edge_feat"]))
# concat_node_feat = np.concatenate((concat_node_feat, graph["node_feat"]))
# concat_num_nodes += graph["num_nodes"]
# concat_num_edges += graph["num_edges"]
