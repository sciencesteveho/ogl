#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Metadata handler for graph creation, and inputs metadata into each step to
perform part 1 of the pipeline. Takes config to tell the next 3 steps which
arguments to use."""


import argparse
import contextlib
import pathlib
from typing import Dict, List, Optional, Union

import graph_constructor
from prepare_bedfiles import GenomeDataPreprocessor

from edge_parser import EdgeParser
from linear_context_parser import LinearContextParser
import utils

NODES = [
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
]


def _get_regulatory_element_references(regulatory: str) -> Optional[str]:
    """Returns the filename corresponding to the given regulatory element
    scheme."""
    regulatory_map = {
        "intersect": "intersect_node_attr.bed",
        "union": "union_node_attr.bed",
        "epimap": "epimap_node_attr.bed",
        "encode": "encode_node_attr.bed",
    }
    return regulatory_map.get(regulatory)


def preprocess_bedfiles(
    experiment_params: Dict[str, Union[str, List[str], Dict[str, str]]],
    tissue_params: Dict[str, Union[str, List[str], Dict[str, str]]],
    nodes: List[str],
) -> None:
    """Directory set-up, bedfile symlinking and preprocessing

    Args:
        experiment_params (Dict[str, Union[str, list]]): experiment config
        tissue_params (Dict[str, Union[str, list]]): tissie-specific configs
        nodes (List[str]): nodes to include in graph
    """
    preprocessObject = GenomeDataPreprocessor(
        experiment_name=experiment_params["experiment_name"],
        interaction_types=experiment_params["interaction_types"],
        nodes=nodes,
        regulatory=experiment_params["regulatory"],
        working_directory=experiment_params["working_directory"],
        params=tissue_params,
    )

    preprocessObject.prepare_data_files()
    print("Bedfile preprocessing complete!")


def parse_edges(
    experiment_params: Dict[str, Union[str, List[str], Dict[str, str]]],
    tissue_params: Dict[str, Union[str, List[str], Dict[str, str]]],
) -> None:
    """Parse nodes and edges to create base graph and for local context
    augmentation

    Args:
        experiment_params (Dict[str, Union[str, list]]): experiment config
        tissue_params (Dict[str, Union[str, list]]): tissie-specific configs
    """
    resource_dir = "/ocean/projects/bio210019p/stevesho/resources"
    regulatory_attr = (
        resource_dir
        + "/"
        + _get_regulatory_element_references(experiment_params["regulatory"])
    )

    baseloop_directory = experiment_params["baseloop_directory"]
    baseloops = experiment_params["baseloops"]
    loopfiles = utils._generate_hic_dict(
        resolution=experiment_params["loop_resolution"]
    )
    loopfile = loopfiles[tissue_params["resources"]["tissue"]]

    edgeparserObject = EdgeParser(
        experiment_name=experiment_params["experiment_name"],
        interaction_types=experiment_params["interaction_types"],
        gene_gene=experiment_params["gene_gene"],
        working_directory=experiment_params["working_directory"],
        loop_file=f"{baseloop_directory}/{baseloops}/{loopfile}",
        regulatory=experiment_params["regulatory"],
        regulatory_attr=regulatory_attr,
        params=tissue_params,
    )

    edgeparserObject.parse_edges()
    print("Edges parsed!")


def parse_linear_context(
    experiment_params: Dict[str, Union[str, List[str], Dict[str, str]]],
    tissue_params: Dict[str, Union[str, List[str], Dict[str, str]]],
    nodes: List[str],
) -> None:
    """Add local context edges based on basenode input

    Args:
        experiment_params (Dict[str, Union[str, list]]): experiment config
        tissue_params (Dict[str, Union[str, list]]): tissie-specific configs
        nodes (List[str]): nodes to include in graph
    """
    experiment_name = experiment_params["experiment_name"]
    working_directory = experiment_params["working_directory"]

    remove_nodes = [node for node in utils.POSSIBLE_NODES if node not in nodes]

    local_dir = f"{working_directory}/{experiment_name}/{tissue_params['resources']['tissue']}/local"
    bedfiles = utils._listdir_isfile_wrapper(dir=local_dir)

    adjusted_bedfiles = [
        bed for bed in bedfiles if all(node not in bed for node in remove_nodes)
    ]

    if experiment_params["regulatory"] == "encode":
        with contextlib.suppress(ValueError):
            nodes.remove("dyadic")

    localparseObject = LinearContextParser(
        experiment_name=experiment_name,
        nodes=nodes,
        working_directory=working_directory,
        feat_window=experiment_params["feat_window"],
        bedfiles=adjusted_bedfiles,
        params=tissue_params,
    )

    localparseObject.parse_context_data()
    print("Local context parser complete!")


def create_tissue_graph(
    experiment_params: Dict[str, Union[str, List[str], Dict[str, str]]],
    tissue_params: Dict[str, Union[str, List[str], Dict[str, str]]],
    nodes: List[str],
) -> None:
    """Creates a graph for the individual tissue. Concatting is dealt with after
    this initial pipeline.

    Args:
        experiment_params (Dict[str, Union[str, list]]): _description_
        tissue_params (Dict[str, Union[str, list]]): _description_
        nodes (List[str]): _description_
    """
    nodes = (
        experiment_params["nodes"] + NODES
        if experiment_params["nodes"] is not None
        else NODES
    )

    if experiment_params["regulatory"] == "encode":
        with contextlib.suppress(ValueError):
            nodes.remove("dyadic")

    experiment_name = experiment_params["experiment_name"]
    working_directory = experiment_params["working_directory"]
    tissue = tissue_params["resources"]["tissue"]

    graph_constructor.make_tissue_graph(
        nodes=nodes,
        experiment_name=experiment_name,
        working_directory=working_directory,
        graph_type="full",
        tissue=tissue,
    )
    print(f"Graph for {tissue} created!")


def main() -> None:
    """Pipeline to generate jobs for creating graphs"""

    # Parse arguments for type of graphs to produce
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    parser.add_argument(
        "--tissue_config", type=str, help="Path to .yaml file with filenames"
    )
    args = parser.parse_args()
    experiment_params = utils.parse_yaml(args.experiment_config)
    tissue_params = utils.parse_yaml(args.tissue_config)
    try:
        nodes = experiment_params["nodes"] + NODES
    except TypeError:
        nodes = NODES
    print(f"Starting pipeline for {experiment_params['experiment_name']}!")

    # create working directory for experimnet
    utils.dir_check_make(
        dir=f"{experiment_params['working_directory']}/{experiment_params['experiment_name']}",
    )

    preprocess_bedfiles(
        experiment_params=experiment_params,
        tissue_params=tissue_params,
        nodes=nodes,
    )
    parse_edges(
        experiment_params=experiment_params,
        tissue_params=tissue_params,
    )
    parse_linear_context(
        experiment_params=experiment_params,
        tissue_params=tissue_params,
        nodes=nodes,
    )
    create_tissue_graph(
        experiment_params=experiment_params,
        tissue_params=tissue_params,
        nodes=nodes,
    )


if __name__ == "__main__":
    main()
