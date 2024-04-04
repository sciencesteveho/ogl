#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Iterative balancing and adaptive coarse-graining of contact matrices"""

import argparse
import os
from typing import List, Tuple, Union

import cooler  # type: ignore
from cooltools.lib.numutils import adaptive_coarsegrain  # type: ignore
from cooltools.lib.numutils import interp_nan  # type: ignore
import numpy as np
import pandas as pd


def _check_balance(clr: cooler.Cooler) -> bool:
    """Check if the cooler is balanced"""
    return "weight" in clr.bins().columns


def _balance_cooler_chr(
    cool: cooler.Cooler, chromosome: str, start: int, end: int
) -> np.ndarray:
    """Returns the balanced matrix for selected bins within a chromosome"""
    return cool.matrix(balance=True).fetch((chromosome, start, end))


def _smoothed_matrix(
    clr: cooler.Cooler, chromosome: str, start: int, end: int
) -> np.ndarray:
    """Apply balancing and adaptive coarse-graining to a chromosome. Returns a
    smoothed matrix."""
    balanced = _balance_cooler_chr(clr, chromosome, start, end)
    raw = clr.matrix(balance=False).fetch((chromosome, start, end))
    return adaptive_coarsegrain(balanced, raw, cutoff=2, max_levels=8).astype(
        np.float32
    )


def _chunk_bins(bins: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """Chunk the bins into smaller bins"""
    return [bins[i : i + chunk_size] for i in range(0, len(bins), chunk_size)]


def _get_chunk_start_end(bins: np.ndarray) -> Tuple[int, int]:
    """Return the start and end of the chunk"""
    return bins[0][1], bins[-1][2]


def process_chromosome(
    clr: cooler.Cooler,
    chromosome: str,
    tissue: str,
    cutoff: float,
    chunk_size: int = 1000,
) -> None:
    """Process each chromosome and write out results to a BEDPE file if above
    threshold. Writes out results to a BEDPE file."""
    bins = clr.bins().fetch(chromosome).to_numpy()
    chunked_bins = _chunk_bins(
        bins=bins, chunk_size=chunk_size
    )  # get chunks for processing

    # Write out contacts to BEDPE file
    outfile = f"{tissue}/{tissue}_{chromosome}_balanced_corse_grain_{cutoff}.bedpe"

    with open(outfile, "a+") as file:
        for bin_chunk in chunked_bins:
            start, end = _get_chunk_start_end(bin_chunk)
            smoothed_matrix = _smoothed_matrix(clr, chromosome, start, end)

            for i in range(len(bin_chunk)):
                for j in range(i, len(bin_chunk)):
                    count = smoothed_matrix[i][j]
                    if count >= cutoff:
                        # Get the genomic coordinates for bin i and bin j
                        start1, end1 = bin_chunk[i][1], bin_chunk[i][2]
                        start2, end2 = bin_chunk[j][1], bin_chunk[j][2]
                        file.write(
                            f"{chromosome}\t{start1}\t{end1}\t{chromosome}\t{start2}\t{end2}\t{count}\n"
                        )


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Iterative balancing and adaptive coarse-graining of contact matrices"
    )
    parser.add_argument(
        "-c",
        "--cooler",
        type=str,
        required=True,
        help="Path to cooler for processing",
    )
    parser.add_argument(
        "-t",
        "--tissue",
        type=str,
        required=True,
        help="Tissue name for output file",
    )
    # parser.add_argument(
    #     "--min",
    #     type=float,
    #     required=False,
    #     default=0.2,
    #     help="Minimum cutoff value for contact matrix",
    # )
    args = parser.parse_args()
    if not os.path.exists(args.tissue):
        os.makedirs(args.tissue)

    # Load cooler
    clr = cooler.Cooler(args.cooler)

    # Check for balancing weights
    if not _check_balance(clr):
        cooler.balance_cooler(clr, store=True)

    # process each chromosome
    # cutoff = 0.2 if args.min is None else args.min
    # for chromosome in clr.chromnames:
    #     process_chromosome(
    #         clr=clr, chromosome=chromosome, tissue=args.tissue, cutoff=cutoff
    #     )

    for cutoff in [0.1, 0.125, 0.15, 0.175, 0.2]:
        process_chromosome(clr=clr, chromosome="10", tissue=args.tissue, cutoff=cutoff)
        process_chromosome(clr=clr, chromosome="1", tissue=args.tissue, cutoff=cutoff)


if __name__ == "__main__":
    main()
