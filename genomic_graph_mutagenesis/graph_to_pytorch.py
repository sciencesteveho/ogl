#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 
#

"""Convert graphs from np tensors to pytorch geometric Data objects.

Graphs are padded with zeros to ensure that all graphs have the same number of
numbers, saved as a pytorch geometric Data object, and a mask is applied to only
consider the nodes that pass the TPM filter.

# f"{self.graph_dir}/{self.tissue}_gene_idxs.pkl"
"""

import argparse
import pickle
from typing import Any, Dict, List, Tuple

import torch
from torch_geometric.data import Data

from utils import parse_yaml


def _train_test_split_nodes():
    

def np_to_pytorch_geometric(graph: str) -> None:
    """Convert graph from np tensor to pytorch geometric Data object"""
    # load graph
    with open(graph, 'rb') as f:
        graph = pickle.load(f)
    
    # get data
    x = torch.tensor(graph['x'], dtype=torch.float)
    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float)
    y = torch.tensor(graph['y'], dtype=torch.float)
    
    # create pytorch geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # save pytorch geometric Data object
    with open(graph, 'wb') as f:
        pickle.dump(data, f)


def main() -> None:
    """Pipeline to generate individual graphs"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to .yaml file with filenames"
    )

    args = parser.parse_args()
    params = parse_yaml(args.config)
    
    # instantiate object
    edgeparserObject = EdgeParser(
        params=params,
        )

    # run pipeline!
    edgeparserObject.parse_edges()


if __name__ == '__main__':
    main()

# save as pytorch geometric data object
pyg_graph = from_networkx(graph)
torch.save(pyg_graph, f"{self.graph_dir}/{self.tissue}_full_graph.pt")