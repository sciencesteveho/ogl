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
# size = 7, 64787, 36
"""

import argparse
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from utils import parse_yaml

# Generate a function to create a training mask for the graph based on a nested dictionary with the keys "train", "test", and "validation"
def create_mask(
    index: str,
    ) -> np.ndarray:
    """Create mask for graph"""
    # load graph indexes
    with open(index, 'rb') as f:
        graph_index = pickle.load(f)


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


if __name__ == '__main__':
    main()


graph_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/graphs'
graph = f'{graph_dir}/scaled/all_tissue_base_graph_scaled.pkl'
index = f'{graph_dir}/all_tissue_full_graph_idxs.pkl'

with open(graph, 'rb') as file:
    data = pickle.load(file)

edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
x = torch.tensor(data['node_feat'], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)