#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests that a random subset of targets match the median values in their
respective tissues."""

import pickle
import random

from cmapPy.pandasGEXpress.parse_gct import parse
import numpy as np
import pandas as pd
import pytest

TISSUES = {
    "aorta": "Artery - Aorta",
    "hippocampus": "Brain - Hippocampus",
    "left_ventricle": "Heart - Left Ventricle",
    "liver": "Liver",
    "lung": "Lung",
    # "mammary": "Breast - Mammary Tissue",
    "pancreas": "Pancreas",
    "skeletal_muscle": "Muscle - Skeletal",
    # "skin": "Skin - Not Sun Exposed (Suprapubic)",
    "small_intestine": "Small Intestine - Terminal Ileum",
}


def _tissue_rename(
    tissue: str,
    data_type: str,
) -> str:
    """Rename a tissue string for a given data type (TPM or protein).

    Args:
        tissue (str): The original tissue name to be renamed.
        data_type (str): The type of data (e.g., 'tpm' or 'protein').

    Returns:
        str: The renamed and standardized tissue name.
    """
    if data_type == "tpm":
        regex = (("- ", ""), ("-", ""), ("(", ""), (")", ""), (" ", "_"))
    else:
        regex = ((" ", "_"), ("", ""))

    tissue_rename = tissue.casefold()
    for r in regex:
        tissue_rename = tissue_rename.replace(*r)

    return tissue_rename


@pytest.mark.parametrize("split", ["training_targets.pkl"])
def test_tpm_median_values(split):
    matrix_dir = (
        "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/gtex_matrices"
    )

    # load df with true median values
    true_df = parse(
        f"{matrix_dir}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    ).data_df

    # load all tissue medians
    median_across_all = pd.read_pickle(
        f"{matrix_dir}/gtex_tpm_median_across_all_tissues.pkl"
    )

    # load df containing difference from average activity
    average_activity = pd.read_pickle(
        f"{matrix_dir}/average_differences_all_tissues_log2.pkl"
    )

    targets_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_hic_gte2/graphs/tpm_1_samples_0.2_test_8-9_val_7-13"
    with open(f"{targets_dir}/{split}", "rb") as f:
        # open targets
        targets = pickle.load(f)

    for entry in random.sample(list(targets[split]), 100):
        target, tissue = entry.split("_", 1)

        true_median = true_df.loc[target, TISSUES[tissue]]
        assert np.isclose(np.log2(true_median + 0.25), targets[split][entry][0])


@pytest.mark.parametrize("split", ["training_targets.pkl"])
def test_tpm_foldchange_values(split):
    matrix_dir = (
        "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/gtex_matrices"
    )

    # load df with true median values
    true_df = parse(
        f"{matrix_dir}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    ).data_df

    # load all tissue medians
    median_across_all = pd.read_pickle(
        f"{matrix_dir}/gtex_tpm_median_across_all_tissues.pkl"
    )

    # load df containing difference from average activity
    average_activity = pd.read_pickle(
        f"{matrix_dir}/average_differences_all_tissues_log2.pkl"
    )

    targets_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_hic_gte2/graphs/tpm_1_samples_0.2_test_8-9_val_7-13"
    with open(f"{targets_dir}/{split}", "rb") as f:
        # open targets
        targets = pickle.load(f)

    for entry in random.sample(list(targets[split]), 100):
        target, tissue = entry.split("_", 1)

        true_median = true_df.loc[target, TISSUES[tissue]]
        true_all_median = median_across_all.loc[target, "all_tissues"]
        true_fold = np.log2((true_median + 0.25) / (true_all_median + 0.25))

        assert np.isclose(true_fold, targets[split][entry][1])


@pytest.mark.parametrize("split", ["training_targets.pkl"])
def test_difference_from_average_activity(split):
    matrix_dir = (
        "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/gtex_matrices"
    )

    # load df with true median values
    true_df = parse(
        f"{matrix_dir}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    ).data_df

    # load all tissue medians
    median_across_all = pd.read_pickle(
        f"{matrix_dir}/gtex_tpm_median_across_all_tissues.pkl"
    )

    # load df containing difference from average activity
    average_activity = pd.read_pickle(
        f"{matrix_dir}/average_differences_all_tissues_log2.pkl"
    )

    targets_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_hic_gte2/graphs/tpm_1_samples_0.2_test_8-9_val_7-13"
    with open(f"{targets_dir}/{split}", "rb") as f:
        # open targets
        targets = pickle.load(f)

    for entry in random.sample(list(targets[split]), 100):
        target, tissue = entry.split("_", 1)

        tissue_rename = _tissue_rename(tissue=TISSUES[tissue], data_type="tpm")
        true_average_diff = true_df.loc[
            target, f"{tissue_rename}_difference_from_average"
        ]

        assert np.isclose(true_average_diff, targets[split][entry][2])


if __name__ == "__main__":
    pytest.main()
