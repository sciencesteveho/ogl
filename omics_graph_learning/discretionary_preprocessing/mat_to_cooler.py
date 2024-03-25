#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Scripts to process Hi-C data from Schmitt et al, 2016. We convert N x N
matrices to cooler format at 40kb resolution against hg19."""


import argparse
import os

import cooler  # type: ignore
import numpy as np
import pandas as pd


def _get_bins(chromsize_file: str, binsize: int) -> pd.DataFrame:
    """Get the bins for a given chromosome size file and bin size.

    Args:
        chromsize_file: The path to the chromosome size file.
        binsize: The size of the bins.

    Returns:
        pd.DataFrame: The bins dataframe.
    """
    chromsizes_df = pd.read_csv(chromsize_file, sep="\t", names=["name", "length"])
    chromsizes = chromsizes_df.set_index("name")["length"]
    return cooler.binnify(chromsizes, binsize)


def _load_matrix(matrix_file: str) -> np.ndarray:
    """Load a matrix from a file.

    Args:
        matrix_file: The path to the matrix file.

    Returns:
        np.ndarray: The matrix.
    """
    matrix = np.loadtxt(matrix_file)
    assert matrix.shape[0] == matrix.shape[1]  # check for square matrix
    return matrix


def _chr_matrix_to_cooler(
    matrix_file: str,
    chrom: str,
    outfile: str,
    bins: pd.DataFrame,
) -> None:
    """_summary_

    Args:
        tissue (str): _description_
        chrom (str): _description_
        bins (pd.DataFrame): _description_
    """
    matrix = _load_matrix(matrix_file=matrix_file)
    pixels = cooler.create.ArrayLoader(
        bins[bins["chrom"] == chrom], matrix, chunksize=10000000
    )
    cooler.create.create_cooler(
        cool_uri=f"{outfile}.cool",
        bins=bins,
        pixels=pixels,
        assembly="hg19",
        dtypes={"count": "int"},
    )


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tissue",
        type=str,
    )
    parser.add_argument(
        "-q",
        "--qq",
        help="Check option if matrices are quantile normalized",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--out_prefix",
        type=str,
        help="Prefix to name output files",
    )
    args = parser.parse_args()

    # get bins
    bins = _get_bins(
        chromsize_file="/ocean/projects/bio210019p/stevesho/resources/hg19.chrom.sizes.txt",
        binsize=40000,
    )

    working_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/coolers"
    tmp_dir = f"{working_dir}/tmp"
    extension = "qq.mat" if args.qq else "mat"

    chrs = []
    for chrom in bins["chrom"].unique():
        if chrom not in ["chrX", "chrY"]:
            if not os.path.exists(
                f"{working_dir}/primary_cohort/{args.tissue}.nor.{chrom}.{extension}"
            ):
                matrix_file = f"{working_dir}/primary_cohort/{args.tissue}.nor.{chrom}.{extension}"
                _chr_matrix_to_cooler(
                    matrix_file=matrix_file,
                    chrom=chrom,
                    outfile=f"{tmp_dir}/{args.tissue}_{chrom}",
                    bins=bins,
                )
            chrs.append(chrom)

    cooler.merge_coolers(
        output_uri=f"{tmp_dir}/{args.out_prefix}.cool",
        input_uris=[f"{tmp_dir}/{args.tissue}_{chrom}.cool" for chrom in chrs],
        mergebuf=10000000,
    )


if __name__ == "__main__":
    main()
