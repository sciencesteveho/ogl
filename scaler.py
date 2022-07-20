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
    parser.add_argument('-p', '--partition', type=str, required=True)
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
    shared_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data'

    for idx, gene in enumerate(genes):
        if idx == 0:
            min_max_df, max_nodes = _tensor_min_max_and_numnodes(f'{directory}/{gene}_{args.tissue}')
        else:
            compare_df, compare_nodes = _tensor_min_max_and_numnodes(f'{directory}/{gene}_{args.tissue}')
            min_max_df, max_nodes = _keep_vals(min_max_df, compare_df, max_nodes, compare_nodes)

    with open(f'{shared_dir}/min_max_{args.tissue}_{args.partition}.pkl', 'wb') as output:
        pickle.dump((min_max_df, max_nodes), output)

if __name__ == '__main__':
    main()


    
'''
files = ['min_max_hippocampus_all.pkl', 'min_max_left_ventricle_all.pkl', 'min_max_mammary_all.pkl']

files = ['min_max_hippocampus_train.pkl', 'min_max_left_ventricle_train.pkl', 'min_max_mammary_train.pkl']

max_nodes = 0
for file in files:
    with open(file, 'rb') as f:
        tup = pickle.load(f)
    if file == files[0]:
        min_max_df, max_nodes = tup
    else:
        compare_df, compare_nodes = tup
        min_max_df, max_nodes = _keep_vals(min_max_df, compare_df, max_nodes, compare_nodes)


ALL
In [15]: min_max_df
Out[15]:
      min          max
0     0.0  248946000.0
1   647.0  248956416.0
2     1.0   35158064.0
3     0.0    9396924.0
4     0.0     342414.0
5     0.0      90880.0
6     0.0     273898.0
7     0.0      17328.0
8     0.0    1149831.0
9     0.0      93172.0
10    0.0    5055102.0
11    0.0          1.0
12    0.0     166624.0
13    0.0      58156.0
14    0.0     100982.0
15    0.0     103221.0
16    0.0      22345.0
17    0.0      25363.0
18    0.0      58156.0

In [16]: max_nodes
Out[16]: 41200

TRAIN
In [18]: min_max_df
Out[18]:
      min          max
0     0.0  248946000.0
1   647.0  248956416.0
2     1.0   26000000.0
3     0.0    9396924.0
4     0.0     342414.0
5     0.0      90880.0
6     0.0     273898.0
7     0.0      17328.0
8     0.0    1149831.0
9     0.0      93172.0
10    0.0    5055102.0
11    0.0          1.0
12    0.0     166624.0
13    0.0      42376.0
14    0.0     100982.0
15    0.0     103221.0
16    0.0      22345.0
17    0.0      25363.0
18    0.0      42376.0

In [19]: max_nodes
Out[19]: 41200
'''