#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import os
import pickle
from typing import Dict, List, Tuple

import pybedtools  # type: ignore


def process_binding_sites(
    postar_path: str, gencode_path: str, output_path="rbp_edges.txt"
) -> None:

    def process_line(line):

        fields = line.rstrip().split("\t")
        return pybedtools.create_interval_from_list([fields[3], fields[8], fields[4]])

    # Read the postar3 binding sites and exclude lines with 'RBP_occupancy' in
    # column 6
    postar_data = pybedtools.BedTool(postar_path).filter(
        lambda x: x.fields[5] != "RBP_occupancy"
    )
    postar_data_cut = postar_data.cut([0, 1, 2, 5, 7])

    gencode_data = pybedtools.BedTool(gencode_path)
    intersection = postar_data_cut.intersect(gencode_data, wa=True, wb=True)
    return [(line[3], line[8], line[4]) for line in intersection]


def process_data(filename: str) -> List[Tuple[str, str, List[str]]]:
    """Process the data from the file and return a list of tuples, where each tuple is RBP"""
    samples = {}

    with open(filename, "r") as file:
        for line in file:
            col1, col2, col3 = line.strip().split("\t")  # assuming tab-separated values
            key = (col1, col2)
            if key not in samples:
                samples[key] = set()
            samples[key].add(col3)

    # Prepare the result list
    result_data = []
    for (col1, col2), col3_values in samples.items():
        if len(col3_values) > 1:
            result_data.append((col1, col2, list(col3_values)))
        else:
            result_data.append((col1, col2, next(iter(col3_values))))

    return result_data


def save_to_file(data: List[Tuple[str, str, List[str]]], output_filename: str) -> None:
    with open(output_filename, "w") as file:
        for col1, col2, _ in data:
            file.write(f"{col1}\t{col2}\t\n")  # Tab-separated output


def main() -> None:
    """Main function"""
    gencode_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
    postar_path = "/ocean/projects/bio210019p/stevesho/data/data_preparse/human.txt"
    process_binding_sites(postar_path, gencode_path)
    data = process_data("rbp_edges.txt")

    # only keep gene -> rbp pairs present in at least 3 separate samples
    triple = [row for row in data if len(row[2]) > 2]
    save_to_file(triple, "rbp_network.txt")


if __name__ == "__main__":
    main()
