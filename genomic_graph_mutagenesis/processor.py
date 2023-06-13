#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import argparse
from typing import Dict

from .edge_parser import EdgeParser
from .graph_constructor import GraphConstructor
from .local_context_parser import LocalContextParser
from utils import _listdir_isfile_wrapper, parse_yaml


class GGMProcessor:
    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
    ):
        """_summary_

        Args:
            params (Dict[str, Dict[str, str]]): _description_
        """
        self.bedfiles = _listdir_isfile_wrapper(
            dir=f"{params['dirs']['root_dir']}/{params['resources']['tissue']}/local",
        )
        self.edge_parser = EdgeParser(params=params)
        self.graph_constructor = GraphConstructor(params=params)
        self.local_context_parser = LocalContextParser(
            params=params,
            bedfiles=self.bedfiles
        )
        
    def process(self):
        """_summary_
        """
        self.edge_parser.parse_edges()
        self.local_context_parser.parse_context_data()
        self.graph_constructor.construct_graph()
        

def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, help="Path to .yaml file with filenames")

    args = parser.parse_args()
    params = parse_yaml(args.config)
    
    ogmprocessorObj = GGMProcessor(params=params)
    ogmprocessorObj.process()


if __name__ == "__main__":
    main()