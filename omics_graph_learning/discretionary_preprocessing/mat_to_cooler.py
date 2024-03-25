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


def process_chromosome_matrix(
    chrom: str, matrix_file: str, bins: pd.DataFrame
) -> pd.DataFrame:
    """
    Process a single chromosome matrix and return contact data.

    Args:
        chrom: Chromosome identifier corresponding to the matrix.
        matrix_file: The path to the matrix file for the chromosome.
        bins: DataFrame containing the bins across all chromosomes.

    Returns:
        DataFrame containing contact data for the specific chromosome.
    """
    # Load matrix
    matrix = _load_matrix(matrix_file)

    # Get bin id offsets for this chromosome
    offset = bins.index[bins["chrom"] == chrom].values[0]

    # Create non-zero contact pairs from matrix
    bin1, bin2 = np.nonzero(matrix)
    counts = matrix[bin1, bin2]
    return pd.DataFrame(
        {"bin1_id": bin1 + offset, "bin2_id": bin2 + offset, "count": counts}
    )


def create_cooler_file(
    chromsize_file: str, matrix_files: dict, binsize: int, output_cool_uri: str
) -> None:
    """
    Create a cooler file given chromosome sizes and individual chromosome matrix files.

    Args:
        chromsize_file: Path to the chromosome size file.
        matrix_files: Dictionary of chromosome to matrix file paths.
        binsize: The size of the bins to be used.
        output_cool_uri: Path where the cooler file will be saved.
    """
    # Get bins
    bins = _get_bins(chromsize_file, binsize)

    # Process each chromosome matrix file
    contacts = pd.DataFrame(columns=["bin1_id", "bin2_id", "count"])
    for chrom, matrix_file in matrix_files.items():
        chrom_contacts = process_chromosome_matrix(chrom, matrix_file, bins)
        contacts = pd.concat([contacts, chrom_contacts], ignore_index=True)

    # Create cooler file
    cooler.create_cooler(cool_uri=output_cool_uri, bins=bins, pixels=contacts)


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

    working_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/coolers"
    tmp_dir = f"{working_dir}/tmp"
    extension = "qq.mat" if args.qq else "mat"
    chromsize_file = (
        "/ocean/projects/bio210019p/stevesho/resources/hg19.chrom.sizes.txt"
    )
    binsize = 40000

    matrix_files = {
        f"chr{chrom}": f"{working_dir}/primary_cohort/{args.tissue}.nor.chr{chrom}.{extension}"
        for chrom in range(1, 23)
    }
    output_cool_uri = f"{tmp_dir}/{args.out_prefix}.cool"
    create_cooler_file(
        chromsize_file=chromsize_file,
        matrix_files=matrix_files,
        binsize=binsize,
        output_cool_uri=output_cool_uri,
    )


if __name__ == "__main__":
    main()
