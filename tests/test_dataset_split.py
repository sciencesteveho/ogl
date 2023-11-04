# Testing for dataset_split.py

"""Tests for dataset_split.py to ensure that training targets are split
correctly (to avoid contamentation) and that the targets are accurate (to ensure
proper training labels)."""

from genomic_graph_mutagenesis.dataset_split import _genes_train_test_val_split
from genomic_graph_mutagenesis.dataset_split import tissue_targets_for_training
import numpy as np
import pandas as pd
import pytest


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


# def test_tissue_targets_for_training():
#     # Define test data and parameters
#     average_activity = pd.DataFrame(
#         {
#             "Gene1": [1.0, 2.0, 3.0],
#             "Gene2": [4.0, 5.0, 6.0],
#         }
#     )

#     expression_median_across_all = "expression_median_across_all.csv"
#     expression_median_matrix = "expression_median_matrix.csv"
#     protein_abundance_matrix = "protein_abundance_matrix.csv"
#     protein_abundance_medians = "protein_abundance_medians.csv"
#     tissue_names = ["TissueA", "TissueB"]
#     tissue_keywords = {
#         "TissueA": ("tpm_keyword_A", "protein_keyword_A"),
#         "TissueB": ("tpm_keyword_B", "protein_keyword_B"),
#     }
#     tpm_dir = "tpm_data_directory"
#     split = {
#         "train": ["Gene1", "Gene2"],
#         "test": ["Gene3"],
#         "validation": ["Gene4"],
#     }

#     # Call the function to get the targets
#     targets = tissue_targets_for_training(
#         average_activity,
#         expression_median_across_all,
#         expression_median_matrix,
#         protein_abundance_matrix,
#         protein_abundance_medians,
#         tissue_names,
#         tissue_keywords,
#         tpm_dir,
#         split,
#     )

#     # Define expected results based on the provided data and parameters
#     expected_train = {
#         "TissueA": np.array([1.0, 2.0, 3.0]),
#         "TissueB": np.array([1.0, 2.0, 3.0]),
#     }
#     expected_test = {
#         "TissueA": np.array([4.0]),
#         "TissueB": np.array([4.0]),
#     }
#     expected_validation = {
#         "TissueA": np.array([5.0]),
#         "TissueB": np.array([5.0]),
#     }

#     # Assert the results for each dataset (train, test, validation)
#     assert np.array_equal(targets["train"]["TissueA"], expected_train["TissueA"])
#     assert np.array_equal(targets["train"]["TissueB"], expected_train["TissueB"])

#     assert np.array_equal(targets["test"]["TissueA"], expected_test["TissueA"])
#     assert np.array_equal(targets["test"]["TissueB"], expected_test["TissueB"])

#     assert np.array_equal(
#         targets["validation"]["TissueA"], expected_validation["TissueA"]
#     )
#     assert np.array_equal(
#         targets["validation"]["TissueB"], expected_validation["TissueB"]
#     )


if __name__ == "__main__":
    pytest.main(["-v", "test_dataset_split.py"])
