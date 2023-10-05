#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Metadata handler for graph creation. Takes initial inputs and takes care of part 1 of the process, from preparing the raw data, to parsing interactions, and generating edge lists."""

import argparse
import os
import pickle

from utils import dir_check_make

NODES = [
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
]


def _string_list(arg):
    """Helper function to pass comma separated list of strings from argparse as
    list
    """
    return arg.split(",")


def submit_prepare_bedfiles():
    """Submit jobs to prepare bedfiles for graph creation"""
    with open(job_file) as f:
        f.writelines()


def main() -> None:
    """Pipeline to generate jobs for creating graphs"""
    # Parse arguments for type of graphs to produce
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
    )
    parser.add_argument(
        "--baseloop_directory",
        "-b",
        type=str,
    )
    parser.add_argument(
        "--interaction_types",
        "-i",
        type=_string_list,
        help="Comma separated list of interaction types to include, e.g. 'ppi,mirna'.",
    )
    parser.add_argument(
        "--nodes",
        "-n",
        type=_string_list,
        help="Comma separated list of nodes to include, e.g. 'cpgislands,superenhancers'.",
    )
    parser.add_argument(
        "--tissues",
        "-t",
        type=_string_list,
        help="Comma separated list of tissues to include, e.g. 'aorta,hippocampus'.",
    )
    parser.add_argument(
        "--working_directory",
        "-w",
        type=str,
        help="Directory to create working folder.",
    )
    args = parser.parse_args()

    # Create main directory for experiment
    dir_check_make(f"{args.working_directory}/{args.experiment_name}")

    # Set variables
    interaction_data = args.interaction_types
    nodes = args.nodes + NODES
    tissues = args.tissues


if __name__ == "__main__":
    main()
