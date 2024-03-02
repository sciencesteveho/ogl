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

import pathlib
import pickle
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import utils


def _set_directories(
    working_directory: str, experiment_name: str, tissue: str
) -> Tuple[pathlib.PosixPath, pathlib.PosixPath, pathlib.PosixPath]:
    """Create directories for the experiment"""
    root_dir = pathlib.Path(f"{working_directory}/{experiment_name}")
    graph_dir = root_dir / "graphs"
    interaction_dir = root_dir / tissue / "interaction"
    parse_dir = root_dir / tissue / "parsing"
    utils.dir_check_make(graph_dir)
    return graph_dir, interaction_dir, parse_dir


@utils.time_decorator(print_args=True)
def _get_edges(
    edge_file: str,
    local: bool = False,
    add_tissue: bool = False,
    tissue: str = "",
) -> pd.DataFrame:
    """Get edges from file"""
    df = pd.read_csv(edge_file, sep="\t", header=None)
    suffix = f"_{tissue}" if add_tissue else ""
    if local:
        df = df.drop(columns=[0, 1, 2, 4, 5, 6, 8]).rename(columns={3: 0, 7: 1})
        df[2] = "local"
    df[0] += suffix
    df[1] += suffix

    return df


def _base_graph(edges: pd.DataFrame):
    """Create a graph from list of edges"""
    return nx.from_pandas_edgelist(
        edges,
        source=0,
        target=1,
        edge_attr=2,
    )


@utils.time_decorator(print_args=False)
def _prepare_reference_attributes(
    reference_dir: str, nodes: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Base_node attr are hard coded in as the first type to load. There are
    duplicate keys in preprocessing but they have the same attributes so
    they'll overwrite without issue.

    Returns:
        Dict[str, Dict[str, Any]]: nested dict of attributes for each node
    """
    with open(f"{reference_dir}/basenodes_reference.pkl", "rb") as file:
        ref = pickle.load(file)

    for node in nodes:
        with open(f"{reference_dir}/{node}_reference.pkl", "rb") as file:
            ref_for_concat = pickle.load(file)
            ref.update(ref_for_concat)

    ref = {
        key: {
            **value,
            "is_gene": int("_tf" not in key and "ENSG" in key),
            "is_tf": int("_tf" in key),
        }
        for key, value in ref.items()
    }
    return ref


@utils.time_decorator(print_args=True)
def graph_constructor(
    tissue: str,
    interaction_dir: pathlib.PosixPath,
    parse_dir: pathlib.PosixPath,
    graph_type: str,
    nodes: List[str],
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
    # create base graph
    if graph_type == "full":
        graph = _base_graph(
            edges=pd.concat(
                [
                    _get_edges(
                        edge_file=interaction_dir / "interaction_edges.txt",
                        add_tissue=True,
                        tissue=tissue,
                    ),
                    _get_edges(
                        edge_file=parse_dir / "edges" / "all_concat_sorted.bed",
                        local=True,
                        add_tissue=True,
                        tissue=tissue,
                    ),
                ],
            )
        )
    else:
        graph = _base_graph(
            edges=_get_edges(
                edge_file=interaction_dir / "interaction_edges.txt",
                add_tissue=True,
                tissue=tissue,
            )
        )

    # add node attributes
    ref = _prepare_reference_attributes(
        reference_dir=parse_dir / "attributes",
        nodes=nodes,
    )
    nx.set_node_attributes(graph, ref)

    return graph


@utils.time_decorator(print_args=True)
def _nx_to_tensors(
    graph: nx.Graph,
    graph_dir: pathlib.PosixPath,
    graph_type: str,
    prefix: str,
    rename: Dict[str, int],
    tissue: str,
) -> None:
    """Save graphs as np tensors, additionally saves a dictionary to map
    nodes to new integer labels

    Args:
        graph (nx.Graph)
    """
    graph = nx.relabel_nodes(graph, mapping=rename)  # manually rename nodes to idx
    edges = np.array(
        [
            [edge[0], edge[1]]
            for edge in nx.to_edgelist(graph, nodelist=list(rename.values()))
        ]
    ).T
    node_features = np.array(
        [np.array(graph.nodes[node].values()) for node in rename.values()]
    )
    edge_features = [graph[u][v][2] for u, v in edges.T]

    with open(graph_dir / f"{prefix}_{graph_type}_graph_{tissue}.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": edges,
                "node_feat": node_features,
                "edge_feat": edge_features,
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "avg_edges": graph.number_of_edges() / graph.number_of_nodes(),
            },
            output,
            protocol=4,
        )


def make_tissue_graph(
    nodes: List[str],
    experiment_name: str,
    working_directory: str,
    graph_type: str,
    tissue: str,
) -> None:
    """Pipeline to generate individual graphs"""

    # working_directory = (
    #     "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing"
    # )
    # tissue = "aorta"
    # experiment_name = "regulatory_only_hic_gte2"

    # get directories
    graph_dir, interaction_dir, parse_dir = _set_directories(
        working_directory=working_directory,
        experiment_name=experiment_name,
        tissue=tissue,
    )

    # instantiate objects and process graphs
    graph = graph_constructor(
        interaction_dir=interaction_dir,
        parse_dir=parse_dir,
        graph_type=graph_type,
        tissue=tissue,
        nodes=nodes,
    )

    # save indexes before renaming to integers
    rename = {node: idx for idx, node in enumerate(sorted(graph.nodes))}
    with open(
        graph_dir / f"{experiment_name}_{graph_type}_graph_{tissue}_idxs.pkl", "wb"
    ) as output:
        pickle.dump(rename, output)

    # save idxs and write to tensors
    _nx_to_tensors(
        graph=graph,
        graph_dir=graph_dir,
        graph_type=graph_type,
        prefix=experiment_name,
        rename=rename,
        tissue=tissue,
    )
