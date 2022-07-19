#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Get min/max across features, max num_nodes, and build scaler according to training set"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import repeat
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from utils import genes_from_gff, time_decorator


@time_decorator(print_args=True)
def _get_tensor_min_max(filename: str) -> None:
    """Lorem ipsum"""
    with open(filename, 'rb') as file:
        graph = pickle.load(file)
    file.close()
    return pd.DataFrame([
        [np.min(arr), np.max(arr)]
        for index, arr in enumerate(np.stack(graph['node_feat'].numpy(), axis=1))],
        columns=['min', 'max']
    )

def main() -> None:
    """Pipeline to generate individual graphs"""
    dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/mammary/parsing/graphs'
    tissue = 'mammary'
    genes = genes_from_gff('/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed')
    genes = ['ENSG00000102974.14', 'ENSG00000124092.12']

    for gene in genes:
        if gene == genes[0]:
            min_max_df = _get_tensor_min_max(f'{dir}/{gene}_{tissue}')
        else:
            compare_df = _get_tensor_min_max(f'{dir}/{gene}_{tissue}')
            min_max_df[min] = np.where(min_max_df[min] < compare_df[0], min_max_df[min], compare_df[0])
            min_max_df[max] = np.where(min_max_df[max] > compare_df[1], min_max_df[max], compare_df[1])


if __name__ == '__main__':
    main()