#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Code to generate baseline performance to compare against graph dataset.

Baseline is adapted from https://doi.org/10.48550/arXiv.1810.09155
"""

import os
import dgl
import pickle
import numpy as np
from numpy import linalg as LA
from scipy import sparse

import argparse

from dgl.data.utils import load_graphs

# ### Custom
# from molecules import MoleculeDatasetDGL


###############################################################################
'''

'''
###############################################################################


def eigenvals_from_laplacian(file, k=200):
    '''
    K-smallest eigenvalues 
    Convert dgl to sparse adjacency matrix. Take laplacian of the graph and derive K eigenvalues. If size N is smaller than K, pad the eigenvalues array with zeros until it reaches K.
    '''
    graph_path='/ocean/projects/bio210019p/stevesho/data/graphs'
    savepath='/ocean/projects/bio210019p/stevesho/ogb_lsc/eigs'
    if not os.path.isfile(f'{savepath}/{file}'):
        print(f'starting graph with tensor {file}')
        graph = load_graphs(f'{graph_path}/{file}')[0][0]
        num_nodes = graph.number_of_nodes()
        laplacian = sparse.eye(num_nodes) - dgl.DGLGraph.adjacency_matrix(graph, scipy_fmt="csr").astype(float)
        if num_nodes > k:
            np.savetxt(f'{savepath}/{file}', LA.eigvals(laplacian.toarray())[0:k].real)
        else:
            k_eigs = LA.eigvals(laplacian.toarray())[0:num_nodes-2]
            np.savetxt(f'{savepath}/{file}', np.pad(k_eigs, [(0, k-num_nodes+2)]).real)
        print(f'finished graph with tensor {file}')
    else:
        print(f'graph with tensor {file} already done')


def save_pkl(vals, name):
    '''
    '''
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(vals, f)


if __name__ == "__main__":
    ###
    parser = argparse.ArgumentParser(description='Get chunk number')
    parser.add_argument('--chunk', type=str, default='0',
                        help='chunk index')
    args = parser.parse_args()

    with open('/ocean/projects/bio210019p/stevesho/ogb_lsc/all_files_50_chunk.pkl', 'rb') as f:
        partition = pickle.load(f)

    working = partition[int(args.chunk)]

    for file in working:
        eigenvals_from_laplacian(file)

### allg = os.listdir(graph_path)
### allsplit = np.array_split(allg, 50)
# with open('all_files_50_chunk.pkl', 'wb') as f:
#     pickle.dump(allsplit, f)

### for i in {0..49}; do sbatch lap2.sh $i; done

    # save_pkl(eigsz, 'train_eigs.pkl')

    # eigs_2 = [a.real for b in eigsz for a in b]
    # save_pkl(eigs_2, 'train_eigs_real.pkl')

    # del eigsz
    # del eigs_2

    # targets = [count[1] for count in testset]
    # targets_2 = [tens.numpy()[0] for tens in targets]
    # save_pkl(targets_2, 'train_targets.pkl')

    # del targets
    # del targets_2

    # partition = valset

    # pool=Pool(processes=6)
    # eigsz = pool.map(eigenvals_from_laplacian, valset)
    # pool.close()

    # save_pkl(eigsz, 'val_eigs.pkl')

    # eigs_2 = [a.real for b in eigsz for a in b]
    # save_pkl(eigs_2, 'val_eigs_real.pkl')

    # targets = [count[1] for count in testset]
    # targets_2 = [tens.numpy()[0] for tens in targets]
    # save_pkl(targets_2, 'val_targets.pkl')