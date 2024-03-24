#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Scripts to process Hi-C output from HiCDCPlus. We filter the output by FDR,
split it by chromosome, and save it as both a matrix and as contact anchors. The
code uses a few local variables that assume you are working with hg38. If
working with another reference genome, you will need to adjust these variables.

Much of the code is adapted from:
    https://github.com/karbalayghareh/GraphReg/blob/master/utils/hic_to_graph.py"""


import argparse
import contextlib
import csv
import os
import scipy
import pathlib 

from typing import List, Tuple
import csv
from contextlib import ExitStack
import numpy as np
import pandas as pd

# global variables, hg38
CHR_LIST = [f"chr{str(i)}" for i in range(1, 23)]  # autosomes only

CHR_LENS = [
    248950000,
    242185000,
    198290000,
    190205000,
    181530000,
    170800000,
    159340000,
    145130000,
    138385000,
    133790000,
    135080000,
    133270000,
    114355000,
    107035000,
    101985000,
    90330000,
    83250000,
    80365000,
    58610000,
    64435000,
    46700000,
    50810000,
    156035000,
]


QVAL_CUTOFFS = {
    0.1: 1,
    0.01: 01,
    0.001: 001,
}


def _split_hicdcplus_to_chrs(input_dir: pathlib.PosixPath, input: str, qvalue_cutoff: float) -> None:
    """Lorem"""
    with open(input, "r") as input_file, ExitStack() as stack:
        reader = csv.DictReader(input_file, delimiter="\t")
        opened_files = {}
        for row in reader:
            if float(row["qvalue"]) <= qvalue_cutoff:
                chr_file_name = f"input/{row['chrI']}.txt"
                if chr_file_name not in opened_files:
                    file_handle = stack.enter_context(open(chr_file_name, "w"))
                    opened_files[chr_file_name] = csv.writer(file_handle, delimiter="\t")
                    opened_files[chr_file_name].writerow(reader.fieldnames)
                opened_files[chr_file_name].writerow(row.values())

    print(
        "Data processing complete. Files have been written for each chromosome with qvalues at or below the cutoff."
    )



def _create_ref_length_matrix(chr_list: List[str], chr_lens: List[int]) -> None:
    """Sets up arrays for padding and creates a DataFrame for each chromosome.

    Args:
        chr_list (List[str]): A list of chromosome names.
        chr_lens (List[int]): A list of chromosome lengths.

    Returns None. Writes CSV files to disk.
    """
    # set up arrays for 0 padding
    padding_bps = 400
    padding_length = padding_bps + padding_bps // 2
    padding = np.zeros(padding_length, dtype=int)

    # create a dataframe for each chromosome
    for idx, chr in enumerate(chr_list):
        nodes = np.arange(5000, chr_lens[idx] + 5000)
        nodes = np.concatenate((padding, nodes, padding))
        df = pd.DataFrame({"chr": chr, "start": nodes, "end": nodes + 5000})
        df.to_csv(
            f"ref_length_matrix_chr{idx}.csv", index=False, header=False, sep="\t"
        )


def _filtered_hic_to_sparse_matrix(input: str, chr: str) -> None:
    """Lorem"""
    # read in the data
    hic_df = pd.read_csv(input, header=None, delimiter='\t', names=['start_i', 'start_j', 'count'])
    seq_df = pd.read_csv(input_bed, header=None, delimiter='\t', names=['start'])
    node_list_all = seq_df['start'].values
    n_nodes_all = len(node_list_all)
    
    # create the initial matrix
    hic = np.zeros((n_nodes_all, n_nodes_all), dtype=np.float32)
    nodes_all = pd.Series(index=node_list_all, data=np.arange(n_nodes_all))
    
    cut = hic_df[hic_df['start_i'].isin(node_list_all)]
    cut = cut[cut['start_j'].isin(node_list_all)]
    cut['i'] = nodes_all[cut['start_i']].values
    cut['j'] = nodes_all[cut['start_j']].values

    hic[cut['i'], cut['j']] = cut['count'].values

    hic_sym = hic + hic.T
    sparse_matrix = scipy.sparse.csr_matrix(hic_sym)
    scipy.sparse.save_npz('path_to_save', sparse_matrix)
    
    
def process_matrices(input: str, qval: float, chr_list: List[str], chr_lens: List[int]) -> None:
    """Lorem"""
    # get seq files
    _split_hicdcplus_to_chrs(input_dir=input_dir, input=input, qval=qval)
    
    # create ref length matrices
    _create_ref_length_matrix(chr_list=chr_list, chr_lens=chr_lens)
    
    # create sparse matrices
    for chr in chr_list:
        _filtered_hic_to_sparse_matrix(input=input, chr=chr)

    
def main(chromsize_file: str, binsize: int) -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Process Hi-C output from HiCDCPlus.")
    parser.add_argument(
        "--hic_input",
        type=str,
        help="Path to .hic file",
    )
    
    process_matrices(chromsize_file, binsize, CHR_LIST)


if __name__ == "__main__":
    main()
