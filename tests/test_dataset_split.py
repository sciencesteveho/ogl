#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO

import random

import numpy as np

from .test_dataset_split import _genes_train_test_val_split

"""_summary_ of project"""

import os
import pickle


def _genes_train_test_val_split():
    genes = {
        "Gene1": 1,
        "Gene2": 2,
        "Gene3": 3,
        "Gene4": 4,
        "Gene5": 5,
        "Gene6": 6,
    }

    test_chrs = [2]
    val_chrs = [4]
    tissues = ["TissueA", "TissueB"]

    result = _genes_train_test_val_split(genes, tissues, True, test_chrs, val_chrs)

    # Define expected results based on the given test_chrs and val_chrs
    expected_train = [
        "Gene1_TissueA",
        "Gene1_TissueB",
        "Gene3_TissueA",
        "Gene3_TissueB",
        "Gene5_TissueA",
        "Gene5_TissueB",
        "Gene6_TissueA",
        "Gene6_TissueB",
    ]
    expected_test = [
        "Gene2_TissueA",
        "Gene2_TissueB",
    ]
    expected_validation = [
        "Gene4_TissueA",
        "Gene4_TissueB",
    ]

    assert result["train"] == expected_train
    assert result["test"] == expected_test
    assert result["validation"] == expected_validation

    # Test with tissue_append=False
    result = _genes_train_test_val_split(genes, tissues, False, test_chrs, val_chrs)

    # Define expected results based on the given test_chrs and val_chrs
    expected_train = ["Gene1", "Gene3", "Gene5", "Gene6"]
    expected_test = ["Gene2"]
    expected_validation = ["Gene4"]

    assert result["train"] == expected_train
    assert result["test"] == expected_test
    assert result["validation"] == expected_validation


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
