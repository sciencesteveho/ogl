#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to check the accuracy of saved node features."""


from pathlib import Path
import pickle
import random
from typing import Any, Dict

import numpy as np
import pybedtools  # type: ignore

from constants import ATTRIBUTES


class NodeFeatureAccuracy:
    """Manually spotcheck the accuracy of derived & aggregated attributes over a
    node.
    """

    def __init__(self, basenodes_file: Path, fasta_file: str) -> None:
        """Instantiate the NodeFeatureAccuracy object."""
        self.basenodes = self.load_basenodes(basenodes_file)
        self.fasta_file = fasta_file

    def aggregate_attribute(self, ref_file, attribute):
        if attribute == "gc":
            return (
                ref_file.nucleotide_content(fi=self.fasta_file)
                .each(
                    lambda feature: [
                        feature[0],
                        feature[1],
                        feature[2],
                        feature[3],
                        feature[4],
                        int(feature[8]) + int(feature[9]),
                    ]
                )
                .groupby(g=[0, 1, 2, 3], c=[4, 5], o=["sum"])
            )
        elif attribute == "recombination":
            return ref_file.intersect(
                pybedtools.BedTool(f"{attribute}.bed"),
                wao=True,
                sorted=True,
            ).groupby(g=[0, 1, 2, 3], c=[4, 8], o=["sum", "mean"])
        else:
            return ref_file.intersect(
                pybedtools.BedTool(f"{attribute}.bed"),
                wao=True,
                sorted=True,
            ).groupby(g=[0, 1, 2, 3], c=[4, 9], o=["sum"])

    def spot_check_feature(self, attribute):
        # randomly select a node
        node = random.choice(list(self.basenodes.keys()))

        # create a BedTool object for this node
        node_data = self.basenodes[node]
        bed_string = f"{node_data['chrom']}\t{node_data['start']}\t{node_data['end']}\t{node}\t{node_data['end'] - node_data['start']}"
        ref_file = pybedtools.BedTool(bed_string, from_string=True)

        # aggregate the attribute
        aggregated = self.aggregate_attribute(ref_file, attribute)

        # extract the aggregated value
        aggregated_value = float(
            aggregated[0][5]
        )  # Assuming the aggregated value is in the 6th field

        # compare with the stored value
        stored_value = node_data["attributes"][attribute]

        return node, aggregated_value, stored_value

    def run_spot_check(self, num_checks: int = 1000):
        results = []
        for _ in range(num_checks):
            attribute = random.choice(self.attributes)
            node, aggregated, stored = self.spot_check_feature(attribute)
            results.append(
                {
                    "node": node,
                    "attribute": attribute,
                    "aggregated": aggregated,
                    "stored": stored,
                    "difference": abs(aggregated - stored),
                    "relative_difference": (
                        abs(aggregated - stored) / max(abs(aggregated), abs(stored))
                        if max(abs(aggregated), abs(stored)) > 0
                        else 0
                    ),
                }
            )

        return results

    @staticmethod
    def load_basenodes(file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load the basenodes dictionary."""
        with open(file_path, "r") as file:
            return pickle.load(file)

    @staticmethod
    def analyze_results(results):
        differences = [r["difference"] for r in results]
        relative_differences = [r["relative_difference"] for r in results]

        print(f"Average absolute difference: {np.mean(differences)}")
        print(f"Median absolute difference: {np.median(differences)}")
        print(f"Average relative difference: {np.mean(relative_differences)}")
        print(f"Median relative difference: {np.median(relative_differences)}")

        large_diff_threshold = np.percentile(differences, 95)
        large_diffs = [r for r in results if r["difference"] > large_diff_threshold]

        print(f"\nTop 5% largest differences:")
        for r in large_diffs[:5]:  # print first 5 for brevity
            print(
                f"Node: {r['node']}, Attribute: {r['attribute']}, Aggregated: {r['aggregated']}, Stored: {r['stored']}, Difference: {r['difference']}"
            )
