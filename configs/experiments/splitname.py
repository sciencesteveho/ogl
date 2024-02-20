#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]

"""Simple script to get split name"""

import argparse
import sys

sys.path.append("../omics_graph_learning")

import utils


def main() -> None:
    """Main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    parser.add_argument("--tpm_filter", type=float, help="TPM filter for genes")
    parser.add_argument(
        "--percent_of_samples_filter",
        type=float,
        help="Percent of samples filter for genes (e.g. 0.20)",
    )
    args = parser.parse_args()
    params = utils.parse_yaml(args.experiment_config)

    return utils._dataset_split_name(
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        tpm_filter=args.tpm_filter,
        percent_of_samples_filter=args.percent_of_samples_filter,
    )


if __name__ == "__main__":
    print(main())
