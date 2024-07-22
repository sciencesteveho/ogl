#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""This script will create one graph per tissue from the parsed edges. All edges
are added before being filtered to only keep edges that can traverse back to a
base node. Attributes are then added for each node as node features and parsed
as a series of numpy arrays."""


from pathlib import Path
import pickle
from typing import Any, Dict, List

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd

from utils import dir_check_make
from utils import time_decorator


def _base_graph(edges: pd.DataFrame):
    """Create a graph from list of edges"""
    return nx.from_pandas_edgelist(
        edges,
        source=0,
        target=1,
        edge_attr=2,
    )


def _remove_blacklist_nodes(
    graph: nx.Graph,
) -> nx.Graph:
    """Remove nodes from graph if they have no attributes. These nodes have no
    attributes because they overlap the encode blacklist."""
    blacklist = [node for node, attrs in graph.nodes(data=True) if len(attrs) == 0]
    graph.remove_nodes_from(blacklist)
    print(
        f"Removed {len(blacklist)} nodes from graph because they overlapped blacklist / have no attributes."
    )
    return graph


def _prune_nodes_without_gene_connections(
    graph: nx.Graph,
    gene_nodes: List[str],
) -> nx.Graph:
    """Remove nodes from an undirected graph if they are not in the same
    connected component as any node in gene_nodes."""
    connected_to_gene = set()
    for component in nx.connected_components(graph):
        if any(gene_node in component for gene_node in gene_nodes):
            connected_to_gene.update(component)

    nodes_to_remove = set(graph) - connected_to_gene
    graph.remove_nodes_from(nodes_to_remove)
    print(
        f"Removed {len(nodes_to_remove)} nodes from graph that are not connected to genes."
    )
    return graph


@time_decorator(print_args=False)
def _get_edges(
    edge_file: str,
    local: bool = False,
    add_tissue: bool = False,
    tissue: str = "",
) -> pd.DataFrame:
    """Get edges from edge file."""
    if local:
        df = pd.read_csv(edge_file, sep="\t", header=None, usecols=[3, 7]).rename(
            columns={3: 0, 7: 1}
        )
        df[2] = "local"
    else:
        df = pd.read_csv(edge_file, sep="\t", header=None)
    suffix = f"_{tissue}" if add_tissue else ""
    df[0] += suffix
    df[1] += suffix

    return df


def _add_tf_or_gene_onehot(
    node: str,
) -> Dict[str, int]:
    """Add one hot encoding for TFs and genes"""
    return {
        "is_gene": int("_tf" not in node and "ENSG" in node),
        "is_tf": int("_tf" in node),
    }


@time_decorator(print_args=False)
def _prepare_reference_attributes(
    reference_dir: str, nodes: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Base_node attr are hard coded in as the first type to load. There are
    duplicate keys in preprocessing but they have the same attributes so they'll
    overwrite without issue. The hardcoded removed_keys are either chr, which we
    don't want the model or learn, or they contain sub dictionaries. Coordinates
    are not a learning feature and positional encodings are added at a different
    step of model training.
    """
    with open(f"{reference_dir}/basenodes_reference.pkl", "rb") as file:
        ref = pickle.load(file)

    for node in nodes:
        with open(f"{reference_dir}/{node}_reference.pkl", "rb") as file:
            ref_for_concat = pickle.load(file)
            ref.update(ref_for_concat)

    return {
        key: {
            **value,
            **_add_tf_or_gene_onehot(key),
        }
        for key, value in ref.items()
    }


@time_decorator(print_args=False)
def graph_constructor(
    tissue: str,
    interaction_dir: Path,
    parse_dir: Path,
    graph_type: str,
    nodes: List[str],
    genes: List[str],
) -> nx.Graph:
    """Create a graph from parsed edges and from local context edges before
    combining. Then add attributes ot each node by using the reference
    dictionaries. The function then removes any nodes within the blacklist
    regions.

    Args:
        tissue (str): name of the tissue
        interaction_dir (Path): directory containing interaction edges
        parse_dir (Path): directory containing parsed edges
        graph_type (str): type of graph to create
        nodes (List[str]): the nodetypes to use in the graph beyond the default
        types included (see config handler)
        genes (List[str]): list of nodes representing genes, for pruning other
        nodes that do not eventually hop to a gene
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

    print(
        f"Base graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    # prune nodes without gene connections
    graph = _prune_nodes_without_gene_connections(graph=graph, gene_nodes=genes)

    # add node attributes
    ref = _prepare_reference_attributes(
        reference_dir=parse_dir / "attributes",
        nodes=nodes,
    )
    nx.set_node_attributes(graph, ref)
    print(
        f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    # remove nodes without attributes
    graph = _remove_blacklist_nodes(graph)
    return graph


@time_decorator(print_args=False)
def _nx_to_tensors(
    graph: nx.Graph,
    graph_dir: Path,
    graph_type: str,
    prefix: str,
    rename: Dict[str, int],
    tissue: str,
) -> None:
    """Save graphs as np tensors, additionally saves a dictionary to map
    nodes to new integer labels."""
    graph = nx.relabel_nodes(graph, mapping=rename)  # manually rename nodes to idx
    coordinates, positional_encodings, node_features = [], [], []

    # loop through nodes to make separate arrays
    for i in range(len(graph.nodes)):
        node_data = graph.nodes[i]
        coordinates.append(list(node_data["coordinates"].values()))
        positional_encodings.append(node_data["positional_encoding"].flatten())
        node_features.append(
            [
                value
                for key, value in node_data.items()
                if key not in ["coordinates", "positional_encoding"]
            ]
        )

    edges = np.array(
        [
            [edge[0], edge[1]]
            for edge in nx.to_edgelist(graph, nodelist=list(rename.values()))
        ]
    ).T
    edge_features = [graph[u][v][2] for u, v in edges.T]

    with open(graph_dir / f"{prefix}_{graph_type}_graph_{tissue}.pkl", "wb") as output:
        pickle.dump(
            {
                "edge_index": edges,
                "node_feat": np.array(node_features),
                "node_positional_encoding": np.array(positional_encodings),
                "node_coordinates": np.array(coordinates),
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
    working_directory: Path,
    graph_type: str,
    tissue: str,
    target_genes: List[str],
) -> None:
    """Pipeline to generate per-sample graphs."""

    # set directories
    graph_dir = working_directory / "graphs"
    interaction_dir = working_directory / tissue / "interaction"
    parse_dir = working_directory / tissue / "parsing"
    dir_check_make(graph_dir)

    # instantiate objects and process graphs
    graph = graph_constructor(
        interaction_dir=interaction_dir,
        parse_dir=parse_dir,
        graph_type=graph_type,
        tissue=tissue,
        nodes=nodes,
        genes=target_genes,
    )

    # save indexes before renaming to integers
    rename = {node: idx for idx, node in enumerate(sorted(graph.nodes))}
    with open(
        graph_dir / f"{experiment_name}_{graph_type}_graph_{tissue}_idxs.pkl", "wb"
    ) as output:
        pickle.dump(rename, output)

    # temp save for debugging
    with open(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/slurm_outputs/test.pkl",
        "wb",
    ) as output:
        pickle.dump(graph, output)

    # save idxs and write to tensors
    _nx_to_tensors(
        graph=graph,
        graph_dir=graph_dir,
        graph_type=graph_type,
        prefix=experiment_name,
        rename=rename,
        tissue=tissue,
    )
