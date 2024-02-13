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
import csv
import pickle
from typing import Any, Dict, Generator, List

import networkx as nx
import numpy as np

import utils


def main() -> None:
    # instantiate objects and process graphs
    max(idxs.values())
    min(idxs.values())

    for idx, tissue in enumerate(params["tissues"]):
        if idx == 0:
            graph = graph_constructor(
                tissue=tissue,
                nodes=nodes,
                root_dir=root_dir,
                graph_type=args.graph_type,
            )

            g_max = max(graph["edge_index"][0] + graph["edge_index"][1])

        else:
            current_graph = graph_constructor(
                tissue=tissue,
                nodes=nodes,
                root_dir=root_dir,
                graph_type=args.graph_type,
            )
            graph = nx.compose(graph, current_graph)

    # save indexes before renaming to integers
    with open(
        f"{graph_dir}/{experiment_name}_{args.graph_type}_graph_idxs.pkl", "wb"
    ) as output:
        pickle.dump(
            {node: idx for idx, node in enumerate(sorted(graph.nodes))},
            output,
        )

    # save idxs and write to tensors
    _nx_to_tensors(
        prefix=experiment_name,
        graph_dir=graph_dir,
        graph=graph,
        graph_type=args.graph_type,
    )


if __name__ == "__main__":
    main()
