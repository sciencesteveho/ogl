#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple utility to map deeploop anchors back to their chromosome positions"""

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr", type=str)
    parser.add_argument("--cell_line", type=str)
    args = parser.parse_args()

    # set up static paths
    map_file_path = Path(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/process_deeploop/scripts/deeploop/DeepLoop_models/ref/hg38_DPNII_anchor_bed"
    )
    working_dir = Path(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic/process_deeploop/scripts"
    )

    map_file = map_file_path / f"{args.chr}.bed"
    anchor_file_path = (
        working_dir
        / args.cell_line
        / "deeploop_output"
        / f"{args.chr}.denoised.anchor.to.anchor"
    )
    output_file_path = (
        working_dir
        / args.cell_line
        / "deeploop_output"
        / f"{args.chr}.denoised.anchor.to.anchor.mapped"
    )

    # create a dictionary from the map file
    anchor_to_location = {}
    with open(map_file, "r") as map_file:
        for line in map_file:
            chrom, start, end, anchor = line.strip().split()
            anchor_to_location[anchor] = (chrom, start, end)

    # process the anchor file and write to output file
    with open(anchor_file_path, "r") as anchor_file, open(
        output_file_path, "w"
    ) as output_file:
        for line in anchor_file:
            anchor1, anchor2, value = line.strip().split()

            chrom_loc1 = anchor_to_location.get(anchor1)
            chrom_loc2 = anchor_to_location.get(anchor2)

            if chrom_loc1 and chrom_loc2:
                output_line = (
                    f"{chrom_loc1[0]}\t{chrom_loc1[1]}\t{chrom_loc1[2]}\t"
                    f"{chrom_loc2[0]}\t{chrom_loc2[1]}\t{chrom_loc2[2]}\t{value}\n"
                )
                output_file.write(output_line)

    print("Remapping complete.")


if __name__ == "__main__":
    main()
