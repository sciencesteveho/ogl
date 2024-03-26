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
        bins=bins[bins["chrom"] == chrom],
        pixels=pixels,
        assembly="hg19",
        dtypes={"count": "int"},
    )

    # clr = cooler.Cooler(f"{outfile}.cool")

    # # add empty chrs
    # spans = cooler.tools.partition(0, len(clr.pixels()), 10000000)
    # chunk_generator = (clr.pixels()[lo:hi] for lo, hi in spans)
    # cooler.create_cooler("test2.cool", bins, chunk_generator)


def create_pixel_data(df_matrix, start_bin_id):
    # Convert the upper triangle of the matrix to a DataFrame
    triu_idx = np.triu_indices_from(df_matrix)
    data = {
        "bin1_id": triu_idx[0] + start_bin_id,
        "bin2_id": triu_idx[1] + start_bin_id,
        "count": df_matrix[triu_idx],
    }
    return pd.DataFrame(data)


def create_pixel_data_zeros(start_bin_id, end_bin_id):
    # Define the number of bins
    num_bins = end_bin_id - start_bin_id

    return pd.DataFrame(
        {
            "bin1_id": np.arange(start_bin_id, end_bin_id),
            "bin2_id": np.arange(start_bin_id, end_bin_id),
            "count": np.zeros(num_bins),  # or whatever the count column is named
        }
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
    # bins = _get_bins(
    #     chromsize_file="/ocean/projects/bio210019p/stevesho/resources/hg19.chrom.sizes.txt",
    #     binsize=40000,
    # )
    chromsizes = cooler.util.fetch_chromsizes("hg19")
    chromsizes = chromsizes.drop("chrM")
    bins = cooler.binnify(chromsizes, binsize=40000)

    working_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/coolers"
    tmp_dir = f"{working_dir}/tmp"
    extension = "qq.mat" if args.qq else "mat"

    all_pixels = []
    for chrom in bins["chrom"].unique():
        if chrom in ["chrX", "chrY"]:
            chr_bins = bins[bins["chrom"] == chrom]
            start, end = (
                chr_bins.index[0],
                chr_bins.index[-1] + 1,
            )

            # Create a df filled with zeros for pixel data for sex chrs
            pixel_data = create_pixel_data_zeros(start_bin_id=start, end_bin_id=end)

        else:
            matrix_file = (
                f"{working_dir}/primary_cohort/{args.tissue}.nor.{chrom}.{extension}"
            )
            matrix = _load_matrix(matrix_file=matrix_file)
            chr_bins = bins[bins["chrom"] == chrom]
            start, end = (
                chr_bins.index[0],
                chr_bins.index[-1] + 1,
            )

            # Generate a df for pixel data for this chromosome
            # using 'start' and 'end' to shift the bin ids
            pixel_data = create_pixel_data(df_matrix=matrix, start_bin_id=start)
        all_pixels.append(pixel_data)

    # Concat for cooler creation
    all_pixels_df = pd.concat(all_pixels, ignore_index=True)

    cooler.create_cooler(
        cool_uri=f"{tmp_dir}/{args.out_prefix}.cool",
        bins=bins,
        pixels=all_pixels_df,
        assembly="hg19",
        dtypes={"count": np.int64},
    )

    # cooler.merge_coolers(
    #     output_uri=f"{tmp_dir}/{args.out_prefix}.cool",
    #     input_uris=[f"{tmp_dir}/{args.tissue}_{chrom}.cool" for chrom in chrs],
    #     mergebuf=10000000,
    # )


if __name__ == "__main__":
    main()
