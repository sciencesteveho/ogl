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
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

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

CHR_LIST = [f"chr{str(i)}" for i in range(1, 23)]  # autosomes only


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


def _split_hicdcplus_to_chrs() -> None:
    """Lorem"""
    pass


def _filter_interactions_by_fdr() -> None:
    """Lorem"""
    pass


def main(chromsize_file: str, binsize: int) -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
