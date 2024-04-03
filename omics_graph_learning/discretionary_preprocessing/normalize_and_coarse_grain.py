#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Iterative balancing and adaptive coarse-graining of contact matrices"""

import argparse
from typing import List, Union

import cooler  # type: ignore
from cooltools.lib.numutils import adaptive_coarsegrain  # type: ignore
from cooltools.lib.numutils import interp_nan  # type: ignore
import numpy as np
import pandas as pd


def _balance_cooler_chr(cool: cooler.Cooler, chromosome: str) -> np.ndarray:
    """Returns the balanced matrix for a chromosome"""
    return cool.matrix(balance=True).fetch(chromosome)


def _write_bedpe(
    contacts: List[List[Union[str, float, int]]],
    chromosome: str,
    tissue: str,
    cutoff: float,
) -> None:
    """Write matrix to BEDPE format"""
    contacts_df = pd.DataFrame(
        contacts,
        columns=["chrom1", "start1", "end1", "chrom2", "start2", "end2", "count"],
    )
    contacts_df.to_csv(
        f"{tissue}_{chromosome}_balanced_corse_grain_{cutoff}.bedpe",
        sep="\t",
        index=False,
    )


def process_chromosome(
    clr: cooler.Cooler, chromosome: str, tissue: str, cutoff: float
) -> None:
    """Apply matrix balancing and adaptive coarse-graining to a chromosome. Writes out results to a BEDPE file."""
    balanced = _balance_cooler_chr(clr, chromosome)  # balanced matrix
    raw = clr.matrix(balance=False).fetch(chromosome)  # get raw vals
    bins = clr.bins().fetch(chromosome).to_numpy()  # get bins for writing

    smoothed_matrix = adaptive_coarsegrain(
        balanced, raw, cutoff=2, max_levels=8
    ).astype(np.float32)

    # Convert the smoothed matrix to coordinates with counts greater than a cutoff value.
    contacts = []
    for i in range(len(smoothed_matrix)):
        for j in range(i, len(smoothed_matrix)):
            count = smoothed_matrix[i][j]
            if count >= cutoff:
                # Get the genomic coordinates for bin i and bin j
                start1, end1 = bins[i][1], bins[i][2]
                start2, end2 = bins[j][1], bins[j][2]
                contacts.append(
                    [chromosome, start1, end1, chromosome, start2, end2, count]
                )

    _write_bedpe(contacts=contacts, chromosome=chromosome, tissue=tissue, cutoff=cutoff)


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

    # load cooler
    clr = cooler.Cooler(args.cooler)

    # store balancing weights
    cooler.balance_cooler(clr, store=True)

    # process each chromosome
    # cutoff = 0.2 if args.min is None else args.min
    # for chromosome in clr.chromnames:
    #     process_chromosome(
    #         clr=clr, chromosome=chromosome, tissue=args.tissue, cutoff=cutoff
    #     )

    for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5]:
        process_chromosome(
            clr=clr, chromosome="chr10", tissue=args.tissue, cutoff=cutoff
        )


if __name__ == "__main__":
    main()
