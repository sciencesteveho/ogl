#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Scripts to process Hi-C data from Schmitt et al, 2016. We convert N x N
matrices to cooler format at 40kb resolution against hg19."""


import argparse
import contextlib
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
        bins=bins[bins["chrom"] == chrom],
        pixels=pixels,
        ordered=True,
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

    # convert to 40kb cooler for each chromosome
    offset = 0
    all_bins = []
    chrs = []
    for chrom in bins["chrom"].unique():
        if chrom not in ["chrX", "chrY"]:
            matrix_file = (
                f"{working_dir}/primary_cohort/{args.tissue}.nor.{chrom}.{extension}"
            )
            _chr_matrix_to_cooler(
                matrix_file=matrix_file,
                chrom=chrom,
                outfile=f"{tmp_dir}/{args.tissue}_{chrom}",
                bins=bins,
            )
            chrs.append(chrom)

            clr = cooler.Cooler(f"{tmp_dir}/{args.tissue}_{chrom}.cool")
            bins = clr.bins()[:]
            bins["start"] = offset
            offset += len(bins)
            all_bins.append(bins)

    all_bins_concat = pd.concat(all_bins, ignore_index=True)
    all_pixels = []
    for chrom in chrs:
        clr = cooler.Cooler(f"{tmp_dir}/{args.tissue}_{chrom}.cool")
        pixels = clr.pixels()[:]
        bins = clr.bins()[:]

        bin_offset = bins["start_id"].iloc[0]
        pixels["bin1_id"] += bin_offset
        pixels["bin2_id"] += bin_offset

        all_pixels.append(pixels)

    all_pixels_concat = pd.concat(all_pixels, ignore_index=True)
    # merge all coolers from sample
    # cooler.merge_coolers(
    #     output_uri=f"{tmp_dir}/{args.out_prefix}.cool",
    #     input_uris=[f"{tmp_dir}/{args.tissue}_{chrom}.cool" for chrom in chrs],
    #     mergebuf=10000000,
    # )
    clr = cooler.create.create_cooler(
        f"{tmp_dir}/{args.out_prefix}.cool",
        bins=all_bins_concat,
        pixels=all_pixels_concat,
        ordered=True,
    )


if __name__ == "__main__":
    main()
