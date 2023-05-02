#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import argparse

from .edge_parser import edge_parser
from .graph_constructor import GraphConstructor
from .local_context_parser import LocalContextParser
from .prepare_bedfiles import GenomeDataPreprocessor
from utils import parse_yaml


def place_holder_function():
    """_summary_ of function"""
    pass


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, help="Path to .yaml file with filenames")

    args = parser.parse_args()
    params = parse_yaml(args.config)

    preprocessObj = GenomeDataPreprocessor(params)
    preprocessObj.prepare_bedfiles()

    edge_parserObj = edge_parser(params)
    edge_parserObj.parse_edges()

    local_context_parserObj = LocalContextParser(params)
    local_context_parserObj.parse_local_context()

    graph_constructorObj = GraphConstructor(params)
    graph_constructorObj.construct_graph()


if __name__ == "__main__":
    main()