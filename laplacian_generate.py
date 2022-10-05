#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Code to generate baseline performance to compare against graph dataset.

Baseline is adapted from https://doi.org/10.48550/arXiv.1810.09155, which takes the k-highest eigenvalues from the normalized Laplacian matrix
"""

import argparse
import numpy as np
import pickle

from scipy import sparse
from scipy import linalg as sci_lin
from numpy import linalg as np_lin

from utils import genes_from_gff, filtered_genes, time_decorator


@time_decorator(print_args=True)
def eigenvals_from_laplacian(dir, graph, tissue, k=200):
    '''
    K-smallest eigenvalues 
    Convert dgl to sparse adjacency matrix. Take laplacian of the graph and derive K eigenvalues. If size N is smaller than K, pad the eigenvalues array with zeros until it reaches K.
    '''
    with open(dir + graph, 'rb') as f:
        g = pickle.load(f)
    num_nodes = g['num_nodes']

    # make adjacency matrix
    row, col = g['edge_index']
    adj = sparse.coo_matrix(
        (np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes)
    ).toarray()

    with open(f'/ocean/projects/bio210019p/stevesho/data/preprocess/adj/{gene}_{tissue}_adj', 'rb') as output:
        pickle.dump(adj, output)
    
    # get laplacian
    lap = sparse.csgraph.laplacian(adj, normed=True)

    if num_nodes > k:
        return linalg.eigh(lap)[0:k].real
    else:
        k_eigs = linalg.eigh(lap).real
        return np.pad(k_eigs, [(0, k-num_nodes)])


if __name__ == "__main__":
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--integer', type=int, required=True)
    args = parser.parse_args()

    chunk_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/chunks'
    # output_dir='/ocean/projects/bio210019p/stevesho/data/preprocess/laplacian_baseline'

    with open(f'{chunk_dir}/chunk_{args.integer}.pkl', 'rb') as file:
        chunk = pickle.load(file)

    eig_arrays = {}
    for gene in chunk:
        tissue = gene.split('_')[1]
        graph_dir = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{tissue}/parsing/graphs_scaled/'
        eig_arrays[gene] = eigenvals_from_laplacian(gene)

    with open(f'{chunk_dir}/eigs/chunk_{args.integer}_eigs.pkl', 'wb') as output:
        pickle.dump(eig_arrays, output)