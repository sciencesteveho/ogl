#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Metadata handler for graph creation, and inputs metadata into each step to
perform part 1 of the pipeline. Takes config to tell the next 3 steps which
arguments to use."""


import argparse
import contextlib
from typing import Optional

from config_handlers import ExperimentConfig
from config_handlers import TissueConfig
from constants import ATTRIBUTES
import construct_graphs as construct_graphs
from edge_parser import EdgeParser
from linear_context_parser import LinearContextParser
from preprocessor import GenomeDataPreprocessor
from utils import _listdir_isfile_wrapper


def preprocess_bedfiles(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> None:
    """Directory set-up, bedfile symlinking and preprocessing

    Args:
        experiment_config (Dict[str, Union[str, list]]): experiment config
        tissue_config (Dict[str, Union[str, list]]): tissie-specific configs
        nodes (List[str]): nodes to include in graph
    """
    preprocessObject = GenomeDataPreprocessor(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
    )

    preprocessObject.prepare_data_files()
    print("Bedfile preprocessing complete!")


def parse_edges(
    experiment_config: ExperimentConfig, tissue_config: TissueConfig
) -> None:
    """Parse nodes and edges to create base graph and for local context
    augmentation

    Args:
        experiment_config (Dict[str, Union[str, list]]): experiment config
        tissue_config (Dict[str, Union[str, list]]): tissie-specific configs
    """
    edgeparserObject = EdgeParser(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
    )

    edgeparserObject.parse_edges()
    print("Edges parsed!")


def parse_linear_context(
    experiment_config: ExperimentConfig,
    tissue_config: TissueConfig,
) -> None:
    """Add local context edges based on basenode input

    Args:
        experiment_config (Dict[str, Union[str, list]]): experiment config
        tissue_config (Dict[str, Union[str, list]]): tissie-specific configs
        nodes (List[str]): nodes to include in graph
    """
    local_dir = (
        experiment_config.working_directory
        / tissue_config.resources["tissue"]
        / "local"
    )

    keep_files = experiment_config.nodes + ATTRIBUTES

    bedfiles = _listdir_isfile_wrapper(dir=local_dir)
    adjusted_bedfiles = [
        bed for bed in bedfiles if all(feature in bed for feature in keep_files)
    ]

    # instantiate object
    localparseObject = LinearContextParser(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
        bedfiles=adjusted_bedfiles,
    )

    localparseObject.parse_context_data()
    print("Local context parser complete!")


def create_tissue_graph(
    experiment_config: ExperimentConfig,
    tissue_config: TissueConfig,
) -> None:
    """Creates a graph for the individual tissue. Concatting is dealt with after
    this initial pipeline.

    Args:
        experiment_config (Dict[str, Union[str, list]]): _description_
        tissue_config (Dict[str, Union[str, list]]): _description_
        nodes (List[str]): _description_
    """
    tissue = tissue_config.resources["tissue"]

    construct_graphs.make_tissue_graph(
        nodes=experiment_config.nodes,
        experiment_name=experiment_config.experiment_name,
        working_directory=experiment_config.working_directory,
        graph_type="full",
        gencode_ref=experiment_config.gencode_gtf,
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

    # Load in the experiment and tissue config files
    experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
    tissue_config = TissueConfig.from_yaml(args.tissue_config)

    # remove dyadic if encode regulatory schema
    if experiment_config.regulatory_schema == "encode":
        with contextlib.suppress(ValueError):
            experiment_config.nodes.remove("dyadic")

    print(f"Starting pipeline for {experiment_config.experiment_name}!")

    # pipeline!
    # preprocess_bedfiles(
    #     experiment_config=experiment_config,
    #     tissue_config=tissue_config,
    # )
    # parse_edges(
    #     experiment_config=experiment_config,
    #     tissue_config=tissue_config,
    # )
    parse_linear_context(
        experiment_config=experiment_config,
        tissue_config=tissue_config,
    )
    # create_tissue_graph(
    #     experiment_config=experiment_config,
    #     tissue_config=tissue_config,
    # )


if __name__ == "__main__":
    main()
