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

from dataset_split import _chr_split_train_test_val, genes_from_gff

# Generate a function to create a training mask for the graph based on a nested dictionary with the keys "train", "test", and "validation"
def create_mask(
    index: str,
    split: Dict[Dict[str, int]]
    ) -> np.ndarray:
    """Create mask for graph"""
    # load graph indexes
    with open(index, 'rb') as f:
        graph_index = pickle.load(f)

    # create masks
    train_mask = [graph_index[gene] for gene in split['train'] if gene in graph_index.keys()]
    test_mask = [graph_index[gene] for gene in split['test'] if gene in graph_index.keys()]
    val_mask = [graph_index[gene] for gene in split['validation'] if gene in graph_index.keys()]

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
    root_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess"
    gene_gtf = f"{root_dir}/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
    test_chrs = ["chr8", "chr9"]
    val_chrs = ["chr7", "chr13"]

    split = _chr_split_train_test_val(
        genes=genes_from_gff(gene_gtf),
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        tissue_append=True,
    )



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