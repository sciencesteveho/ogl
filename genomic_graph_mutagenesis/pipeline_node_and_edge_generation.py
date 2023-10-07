#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Metadata handler for graph creation, and inputs metadata into each step to
perform part 1 of the pipeline. Takes config to tell the next 3 step which
arguments to use."""

import argparse

from edge_parser import EdgeParser
from local_context_parser import LocalContextParser
from prepare_bedfiles import GenomeDataPreprocessor
from utils import _listdir_isfile_wrapper
from utils import dir_check_make
from utils import LOOPFILES
from utils import parse_yaml

NODES = [
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
]


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
    # args = parser.parse_args(
    #     [
    #         "--experiment_config",
    #         "configs/ablation_experiments/alldata_combinedloops.yaml",
    #         "--tissue_config",
    #         "configs/aorta.yaml",
    #     ]
    # )
    args = parser.parse_args(
        [
            "--experiment_config",
            "configs/ablation_experiments/regulatoryonly_combinedloops.yaml",
            "--tissue_config",
            "configs/aorta.yaml",
        ]
    )
    experiment_params = parse_yaml(args.experiment_config)
    tissue_params = parse_yaml(args.tissue_config)

    # Create working directory for experimnet
    dir_check_make(
        dir=f"{experiment_params['working_directory']}/{experiment_params['experiment_name']}",
    )

    # Prepare bedfiles
    print(f"Starting pipeline for {experiment_params['experiment_name']}!")
    print(f"Bedfile preprocessing for {experiment_params['experiment_name']}!")
    preprocessObject = GenomeDataPreprocessor(
        experiment_name=experiment_params["experiment_name"],
        interaction_types=experiment_params["interaction_types"],
        nodes=experiment_params["nodes"],
        working_directory=experiment_params["working_directory"],
        params=tissue_params,
    )
    preprocessObject.prepare_data_files()
    print("Bedfile preprocessing complete!")

    print(f"Parsing edges for {experiment_params['experiment_name']}!")
    edgeparserObject = EdgeParser(
        experiment_name=experiment_params["experiment_name"],
        interaction_types=experiment_params["interaction_types"],
        nodes=experiment_params["nodes"],
        working_directory=experiment_params["working_directory"],
        loop_file=f"{experiment_params['baseloop_directory']}/{experiment_params['baseloops']}/{LOOPFILES[experiment_params['baseloops']][tissue_params['resources']['tissue']]}",
        params=tissue_params,
    )
    edgeparserObject.parse_edges()
    print("Preprocessing complete!")

    # print("Preprocessing complete!")
    # bedfiles = _listdir_isfile_wrapper(
    #     dir=f"{tissue_params['dirs']['root_dir']}/{tissue_params['resources']['tissue']}/local",
    # )
    # bedfiles = [x for x in bedfiles if "chromatinloops" not in x]
    # localparseObject = LocalContextParser(
    #     bedfiles=bedfiles,
    #     params=tissue_params,
    # )
    # localparseObject.parse_context_data()
    # print("Preprocessing complete!")
    # print(f"Edges and nodes have been generated for {args.experiment_name}!")


if __name__ == "__main__":
    main()
