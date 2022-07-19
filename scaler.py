#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Get min/max across features, max num_nodes, and build scaler according to training set"""

import argparse
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Any, Dict, List, Tuple

from utils import genes_from_gff, time_decorator


@time_decorator(print_args=True)
def _tensor_min_max_and_numnodes(filename: str) -> Tuple[pd.DataFrame, int]:
    """Lorem ipsum"""
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    file.close()
    return pd.DataFrame([
        [np.min(arr), np.max(arr)]
        for index, arr in enumerate(np.stack(graph['node_feat'].numpy(), axis=1))],
        columns=['min', 'max']
    ), graph['num_nodes']


@time_decorator(print_args=False)
def _keep_vals(df1, df2, node_1, node_2):
    """_lorem ipsum"""
    df1[min] = np.where(df1[min] > df2[min], df1[min], df2[min])
    df1[max] = np.where(df1[max] < df2[max], df1[max], df2[max])
    if node_2 > node_1:
        node_1 = node_2
    return df1, node_1


def main() -> None:
    """Pipeline to generate individual graphs"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tissue', type=str, required=True)
    parser.add_Argument('-p', '--partition', type=str, required=True)
    args = parser.parse_args()

    if args.partition == 'all':
        gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'
        genes = list(genes_from_gff(gff))
    else:
        gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/graph_partition_test_chr8-chr9_val_chr7-chr13.pkl'
        with open(gff, 'rb') as file:
            partition = pickle.load(file)
        genes = list(partition['train'])

    directory = f'/ocean/projects/bio210019p/stevesho/data/preprocess/{args.tissue}/parsing/graphs'
    shared_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/'

    for idx, gene in enumerate(genes):
        if idx == 0:
            min_max_df, max_nodes = _tensor_min_max_and_numnodes(f'{directory}/{gene}_{args.tissue}')
        else:
            compare_df, compare_nodes = _tensor_min_max_and_numnodes(f'{directory}/{gene}_{args.tissue}')
            min_max_df, max_nodes = _keep_vals(min_max_df, compare_df, max_nodes, compare_nodes)

    with open(f'{shared_dir}/min_max_{partition}.pkl', 'rb') as output:
        pickle.dump((min_max_df, max_nodes), output)

if __name__ == '__main__':
    main()