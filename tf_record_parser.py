#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Look into synder paper for mass spec values as a potential target

"""Convert tensors to TFrecords for running on CS-2"""


import csv
import pickle
import pandas as pd

from typing import Tuple

from utils import genes_from_gff, time_decorator, TISSUE_TPM_KEYS


@time_decorator(print_args=True)
def chr_split_train_test_val(genes, test_chrs, val_chrs):
    """
    create a list of training, split, and val IDs
    """
    return {
        'train': [gene for gene in genes if genes[gene] not in test_chrs + val_chrs],
        'split': [gene for gene in genes if  genes[gene] in test_chrs],
        'validation': [gene for gene in genes if genes[gene] in val_chrs],
    }

def std_dev_and_mean_gtex() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get means and standard deviation for TPM across all GTEx tissues"""
    
def main() -> None:
    """Pipeline to generate dataset split and target values"""
    genes = genes_from_gff('/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed')
    test_chrs=['chr8', 'chr9']
    val_chrs=['chr7', 'chr13']

    split = chr_split_train_test_val(
        genes=genes,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )

    directory = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data'
    with open(f"{directory}/graph_partition_test_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}.pkl", 'wb') as output:
        pickle.dump(split, output)


if __name__ == '__main__':
    main()