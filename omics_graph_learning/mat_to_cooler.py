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

import cooler
import numpy as np
import pandas as pd


def _get_bins(chromsize_file: str, binsize: int) -> pd.DataFrame:
    """_summary_

    Args:
        chromsize_file (str): _description_
        binsize (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    chromsizes = pd.read_csv(
        chromsize_file, sep="\t", names=["name", "length"]
    ).set_index("name")["length"]
    return cooler.binnify(chromsizes, binsize)


def _chr_matrix_to_cooler(
    data_directory: str,
    savedir: str,
    tissue: str,
    chrom: str,
    bins: pd.DataFrame,
    qq: bool,
) -> None:
    """_summary_

    Args:
        tissue (str): _description_
        chrom (str): _description_
        bins (pd.DataFrame): _description_
    """
    if qq:
        matrix = np.loadtxt(f"{data_directory}/{tissue}.nor.{chrom}.qq.mat")
    else:
        matrix = np.loadtxt(f"{data_directory}/{tissue}.nor.{chrom}.mat")
    pixels = cooler.create.ArrayLoader(
        bins[bins["chrom"] == chrom], matrix, chunksize=10000000
    )
    cooler.create_cooler(
        f"{savedir}/chrs/{tissue}_{chrom}.cool",
        bins,
        pixels,
        dtypes={"count": "int"},
        assembly="hg19",
    )


def main(chromsize_file: str, binsize: int) -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tissue",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--data_directory",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
    )
    parser.add_argument(
        "-q",
        "--qq",
        help="Check option if matrices are quantile normalized",
        action="store_true",
    )
    args = parser.parse_args()

    # get bins
    bins = _get_bins(
        chromsize_file="/ocean/projects/bio210019p/stevesho/resources/hg19.chrom.sizes.txt",
        binsize=40000,
    )

    try:
        os.makedirs(f"{args.savedir}/chrs")
    except FileExistsError:
        pass

    # convert to 40kb cooler for each chromosome
    chrs = []
    for chrom in bins["chrom"].unique():
        if chrom == "chrY":
            pass
        else:
            _chr_matrix_to_cooler(
                data_directory=args.data_directory,
                savedir=args.savedir,
                tissue=args.tissue,
                chrom=chrom,
                bins=bins,
                qq=args.qq,
            )
            chrs.append(chrom)

    # merge all coolers from sample
    cooler.merge_coolers(
        output_uri=f"{args.savedir}/{args.tissue}.cool",
        input_uris=[
            f"{args.savedir}/chrs/{args.tissue}_{chrom}.cool" for chrom in chrs
        ],
        mergebuf=20000000,
    )


if __name__ == "__main__":
    main(
        chromsize_file="/ocean/projects/bio210019p/stevesho/resources/hg19.chrom.sizes.txt",
        binsize=40000,
    )
