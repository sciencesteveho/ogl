#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Code to download and process GTEx gene expression data for training targets. Note that the protein expression data is available as XLSX files and thus are not included in the script."""

import os
import pathlib
import pickle

from cmapPy.pandasGEXpress.parse_gct import parse  # type: ignore
import pandas as pd

DOWNLOADS_URLS = [
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
]


def _check_and_download(file: str) -> None:
    """Download a given file if the file does not already exist"""
    if not os.path.exists(file):
        print(f"Downloading {file}...")
        os.system(
            f"wget -O {file} https://s3.amazonaws.com/keras-datasets/mnist.pkl.gz"
        )
    else:
        print(f"{file} already exists.")


def _tpm_median_across_all_tissues(
    median_across_all_file: pathlib.PosixPath,
    all_matrix_gct: str,
) -> None:
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT and
    saves it. Because the file is large and requires a lot of memory, we ran
    this separately from the produce_training_targets function and is only run once.

    Args:
        median_across_all_file (str): /path/to/median_across_all_file
        all_matrix_gct (str): /path/to/gtex gct file
    """
    try:
        if not median_across_all_file.exists():
            median_series = pd.Series(
                parse(all_matrix_gct).data_df.median(axis=1), name="all_tissues"
            ).to_frame()
            median_series.to_pickle(median_across_all_file, mode="xb")
        else:
            print("File already exists")
    except FileExistsError:
        print("File already exists!")


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
