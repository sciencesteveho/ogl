#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Fit a scaler for node feats"""

import argparse
import joblib
import os
import pickle

from sklearn.preprocessing import MinMaxScaler

from target_labels_train_split import _chr_split_train_test_val
from utils import genes_from_gff, filtered_genes


TISSUES = [
    'hippocampus',
    'liver',
    'lung',
    'mammary', 
    'pancreas',
    'skeletal_muscle',
    'left_ventricle',
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feat', type=int, required=True)
    args = parser.parse_args()

    genes = genes_from_gff('/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed')
    test_chrs = ['chr8', 'chr9']
    val_chrs = ['chr7', 'chr13']

    # split genes by chr holdouts
    split = _chr_split_train_test_val(
        genes=genes,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )

    train = split['train']
    root_dir='/ocean/projects/bio210019p/stevesho/data/preprocess'
    output_dir='/ocean/projects/bio210019p/stevesho/data/preprocess/data_scaler'

    scaler = MinMaxScaler()
    for tissue in TISSUES:
        directory=f'/ocean/projects/bio210019p/stevesho/data/preprocess/{tissue}/parsing/graphs'
        # genes = filtered_genes(f'{root_dir}/{tissue}/gene_regions_tpm_filtered.bed')
        genes = os.listdir(f'{directory}')
        genes = [gene.split('_')[0] for gene in genes if gene.split('_')[0] in train]
        for gene in genes:
            with open(f'{directory}/{gene}_{tissue}', 'rb') as f:
                g = pickle.load(f)
            node_feat = g['node_feat']
            scaler.partial_fit(node_feat[:,args.feat].reshape(-1, 1))

    ### save
    joblib.dump(scaler, f'{output_dir}/feat_{args.feat}_scaler.pt')

    
if __name__ == '__main__':
    main()