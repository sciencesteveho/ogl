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
from torch_geometric.data import Data, InMemoryDataset

from utils import parse_yaml


def _combined_graph_arrays(
    tissue_list: List[str],
    graph_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine graph arrays from multiple tissues"""
    feats = []
    for tissue in tissue_list:
        graph_file = f'{graph_dir}/graph.pkl'
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        feats.append(graph['node_feat'])


class GenomeGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None):
        super(GenomeGraphDataset, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        


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