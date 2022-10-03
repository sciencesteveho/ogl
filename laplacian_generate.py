#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Code to generate baseline performance to compare against graph dataset.

Baseline is adapted from https://doi.org/10.48550/arXiv.1810.09155, which takes the k-highest eigenvalues from the normalized Laplacian matrix
"""


import argparse
import numpy as np
import pickle

import scipy.sparse as sp
from scipy.sparse import csgraph
from numpy import linalg as LA

from utils import genes_from_gff, filtered_genes, time_decorator


def _degree_power(A, k):
    """
    Computes \(\D^{k}\) from the given adjacency matrix.

    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: D^k
    """
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.0
    return np.diag(degrees)


def _degrees(A, symmetric):
    """
    Normalizes the given adjacency matrix using the degree matrix as either
    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).

    :param A: rank 2 array
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        normalized_D = _degree_power(A, -0.5)
        output = normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = _degree_power(A, -1.0)
        output = normalized_D.dot(A)
    return output


def _normalize_adj(A, symmetric=False):
    """
    Computes the graph filter described in
    [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).

    :param A: array with rank 2;
    :param symmetric: boolean, whether to normalize the matrix as
    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
    :return: array with rank 2, same as A;
    """
    fltr = A.copy()
    I = np.eye(A.shape[-1], dtype=A.dtype)
    A_tilde = A + I
    fltr = _degrees(A_tilde, symmetric=symmetric)
    return fltr


@time_decorator(print_args=True)
def eigenvals_from_laplacian(graph, k=200):
    '''
    K-smallest eigenvalues 
    Convert dgl to sparse adjacency matrix. Take laplacian of the graph and derive K eigenvalues. If size N is smaller than K, pad the eigenvalues array with zeros until it reaches K.
    '''
    with open(graph, 'rb') as f:
        g = pickle.load(f)
    num_nodes = g['num_nodes']

    # make adjacency matrix
    row, col = g['edge_index']
    adj = sp.coo_matrix(
        (np.ones_like(row), (row, col)), shape=(num_nodes, num_nodes)
    ).toarray()

    adj = _normalize_adj(adj)
    lap = csgraph.laplacian(adj)

    if num_nodes > k:
        return LA.eigvals(lap)[0:k].real
    else:
        k_eigs = LA.eigvals(lap).real
        return np.pad(k_eigs, [(0, k-num_nodes)])


if __name__ == "__main__":
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, help='tissue type')
    args = parser.parse_args()

    root_dir='/ocean/projects/bio210019p/stevesho/data/preprocess'
    output_dir='/ocean/projects/bio210019p/stevesho/data/preprocess/laplacian_baseline'
    directory=f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs_scaled/'

    genes = filtered_genes(f'{root_dir}/{args.tissue}/gene_regions_tpm_filtered.bed')
    eig_arrays = {
        gene: eigenvals_from_laplacian(directory + f'{gene}_{args.tissue}')
        for gene in genes
    }

    with open(f'{output_dir}/{args.tissue}_eigs.pkl', 'wb') as output:
        pickle.dump(eig_arrays, output)