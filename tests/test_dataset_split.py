#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO

import pytest

from .test_dataset_split import _genes_train_test_val_split

"""_summary_ of project"""

import os
import pickle


class TestGenesTrainTestValSplit:
    def test_genes_train_test_val_split(self):
        genes = {
            "gene1": 1,
            "gene2": 2,
            "gene3": 3,
            "gene4": 4,
            "gene5": 5,
            "gene6": 6,
        }

        test_chrs = [2]
        val_chrs = [4]
        tissues = ["TissueA", "TissueB"]

        result = _genes_train_test_val_split(
            genes=genes,
            tissues=tissues,
            test_chrs=[2],
            val_chrs=[4],
            tissue_append=True,
        )

        # Define expected results based on the given test_chrs and val_chrs
        expected_train = [
            "gene1_TissueA",
            "gene1_TissueB",
            "gene3_TissueA",
            "gene3_TissueB",
            "gene5_TissueA",
            "gene5_TissueB",
            "gene6_TissueA",
            "gene6_TissueB",
        ]
        expected_test = [
            "gene2_TissueA",
            "gene2_TissueB",
        ]
        expected_validation = [
            "gene4_TissueA",
            "gene4_TissueB",
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

    def test_no_overlap(self):
        genes = {
            "gene1": 1,
            "gene2": 1,
            "gene3": 1,
            "gene4": 7,
            "gene5": 8,
        }
        tissues = ["tissue1", "tissue2", "tissue3"]
        split = _genes_train_test_val_split(
            genes=genes,
            tissues=tissues,
            test_chrs=["chr1"],
            val_chrs=["chr7"],
            tissue_append=False,
        )

        train_set = set(split["train"])
        test_set = set(split["test"])
        validation_set = set(split["validation"])

        assert not train_set & test_set
        assert not train_set & validation_set
        assert not test_set & validation_set


if __name__ == "__main__":
    pytest.main()
