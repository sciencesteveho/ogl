#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Code to scale node_feats
"""

import argparse
import joblib
import pickle

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from target_labels_train_split import _chr_split_train_test_val
from utils import genes_from_gff, filtered_genes, time_decorator


if __name__ == "__main__":
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='mammary',
                        help='tissue_type')
    args = parser.parse_args()
    
    root_dir='/ocean/projects/bio210019p/stevesho/data/preprocess'
    graph_dir=f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs'
    scale_dir=f'/ocean/projects/bio210019p/stevesho/data/preprocess/data_scaler'
    genes = filtered_genes(f'{root_dir}/{args.tissue}/gene_regions_tpm_filtered.bed')

    scalers = {i: joblib.load(f'{scale_dir}/feat_{i}_scaler.pt') for i in range(0, 34)}

    for gene in genes:
        with open(f'{graph_dir}/{gene}_{args.tissue}', 'rb') as f:
            g = pickle.load(f)
        node_feat = g['node_feat'].astype(np.float32)
        for i in range(0, 34):
            node_feat[:,i] = scalers[i].transform(node_feat[:,i].reshape(-1,1)).reshape(1, -1)[0]
        g['node_feat'] = node_feat 
        with open(f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs_scaled/{gene}_{args.tissue}', 'wb') as output:
            pickle.dump(g, output)
