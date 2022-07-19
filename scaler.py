#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Get min/max across features, max num_nodes, and build scaler according to training set"""

import argparse
import os
import pickle

import np as np
import pandas as pd
import tensorflow as tf

from itertools import repeat
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from utils import time_decorator


@time_decorator(print_args=True)
def _get_tensor_min_max(
    dir: str,
    tissue: str,
    gene: str, 
    ) -> None:
    """Lorem ipsum"""
    with open(f'{dir}/{gene}_{tissue}', 'rb') as file:
        graph = pickle.load(file)
    file.close()
    return pd.DataFrame([
        [np.min(arr), np.max(arr)]
        for index, arr in enumerate(np.stack(graph['node_feat'].numpy(), axis=1))
    ])

def main() -> None:
    """Pipeline to generate individual graphs"""
    dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/mammary/parsing/graphs'
    tissue = 'mammary'
    genes = ['ENSG00000102974.14', 'ENSG00000124092.12']

    test1 = _get_tensor_min_max(dir, tissue, genes[0])
    test2 = _get_tensor_min_max(dir, tissue, genes[1])


if __name__ == '__main__':
    main()