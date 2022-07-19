#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Look into synder paper for mass spec values as a potential target

"""Get dataset train/val/test split"""


import csv
import pickle
from webbrowser import Chrome

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import repeat
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from utils import time_decorator, TISSUE_TPM_KEYS


@time_decorator(print_args=True)

def chr_split_train_test_val(gff, test_chrs, val_chrs):
    """
    create a list of training, test, and val IDs
    """
    with open(gff, newline = '') as file:
        genes = {line[3]: line[0] for line in csv.reader(file, delimiter='\t')}

    return {
        'train': [gene for gene in genes if genes[gene] not in test_chrs + val_chrs],
        'test': [gene for gene in genes if genes[gene] in test_chrs],
        'validation': [gene for gene in genes if genes[gene] in val_chrs],
    }

def std_dev_and_mean_gtex() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get means and standard deviation for TPM across all GTEx tissues"""
    
def main() -> None:
    """Pipeline to generate dataset split and target values"""

    gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'
    test_chrs=['chr8', 'chr9']
    val_chrs=['chr7', 'chr13']

    test = chr_split_train_test_val(
        gff=gff,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )

    output = open(f'', 'wb')
    try:
        pickle.dump(test, output)
    finally:
        output.close()


if __name__ == '__main__':
    main()