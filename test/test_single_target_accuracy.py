#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Tests that a random subset of targets match the median values in their
respective tissues. In contrast to typical unit testing, this test is designed
to run post-hoc against an experimental config to ensure that the derived
training targets are accurate if checked manually. Thus, users must provide a
config for the test to run and should initiate the test with the split_name the
experiment was run with.

This test is specifically designed for experiments that have a single tissue in
the config (i.e. single-tissue models). For models that have multiple samples /
tissues, use test_multi_target_accuracy.py.

Example usage
--------
>>> pytest test_single_target_accuracy.py \
        --config path/to/config.yaml \
        --split_name {split_name}
"""


import argparse
import pickle
import random
import sys
from typing import cast, Dict, List, Tuple, Union

import numpy as np
from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.config_handlers import TissueConfig
from omics_graph_learning.constants import TARGET_FILE
from omics_graph_learning.gene_filter import read_encode_rna_seq_data
import pandas as pd
import pytest


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run single target accuracy test")
    parser.add_argument(
        "--config", required=True, help="Path to the experiment config file"
    )
    parser.add_argument(
        "--split_name",
        required=True,
        help="Name of the split to test (see ogl.py pipeline)",
    )
    return parser.parse_args()


def ensure_single_target(config: ExperimentConfig) -> None:
    """Ensure that the config only has a single tissue."""
    if len(config.tissues) > 1:
        print(
            "This test is designed for single-tissue models."
            "Use test_multi_target_accuracy.py for multi-tissue models."
        )
        sys.exit(1)


def _get_gene_name_without_tissue(gene: str) -> str:
    """Return the gene name without the tissue suffix."""
    return gene.split("_")[0]


def pull_configuration_data(
    config: ExperimentConfig, split_name: str
) -> Tuple[Dict[str, Dict[str, np.ndarray]], pd.DataFrame, str]:
    """Prepare the data and filepaths for testing.

    Returns the loaded target and the true value dataframe, which is the rna-seq
    dataframe filtered for genes with 'ENSG' and 'TPM' columns.
    """
    # load the targets
    graph_dir = config.graph_dir
    target_file = graph_dir / split_name / TARGET_FILE
    with open(target_file, "rb") as file:
        targets = pickle.load(file)

    # load the tissue config and load the tsv
    tissue_config = TissueConfig.from_yaml(
        config.config_dir / f"{config.tissues[0]}.yaml"
    )
    df = read_encode_rna_seq_data(tissue_config.resources["rna"])
    return targets, df.filter(items=["TPM"]), config.log_transform


def point_transform_with_pseudocount(
    point: Union[float, int], transform_type: str = "log2", pseudocount: float = 0.25
) -> float:
    """Transform a single point using the same transformation as the original
    data.
    """
    if transform_type == "log2":
        return np.log2(point + pseudocount)
    elif transform_type == "log1p":
        return np.log1p(point + pseudocount)
    elif transform_type == "log10":
        return np.log10(point + pseudocount)
    else:
        raise ValueError(
            "Invalid log transformation type specified. Choose 'log2', 'log1p', or 'log10'."
        )


def compare_target_to_true_value(
    target: np.ndarray, true_value: float, log_transform_type: str
) -> None:
    """Compare the target value to the manually dervied true value (w/
    transformation)."""
    transformed_value = point_transform_with_pseudocount(
        point=true_value, transform_type=log_transform_type
    )
    assert np.isclose(target, transformed_value)


@pytest.fixture(scope="module")
def test_setup() -> (
    Tuple[ExperimentConfig, Dict[str, Dict[str, np.ndarray]], pd.DataFrame]
):
    """A pytest fixture to prepare configuration and data for tests."""
    args = parse_arguments()
    config = ExperimentConfig.from_yaml(args.config)
    ensure_single_target(config)
    targets, rna_seq_df, log_transform_type = pull_configuration_data(
        config, args.split_name
    )
    return config, targets, rna_seq_df


@pytest.mark.parametrize("log_transform_type", ["log2", "log1p", "log10"])
@pytest.mark.parametrize("split", ["train", "test", "validation"])
def test_targets(
    test_setup: Tuple[ExperimentConfig, Dict[str, Dict[str, np.ndarray]], pd.DataFrame],
    split: str,
):
    """Test that randomly selected targets match manually calculated values."""
    config, targets, rna_seq_df = test_setup

    # sourcery skip: no-loop-in-tests
    for target in random.sample(list(targets[split].keys()), 100):
        gene = target.split("_")[0]  #
        true_value = rna_seq_df.loc[gene, "TPM"]
        float_true_value = cast(float, true_value)
        target_value = targets[split][target][0]
        compare_target_to_true_value(
            target=target_value,
            true_value=float_true_value,
            log_transform_type=config.log_transform,
        )


if __name__ == "__main__":
    pytest_args = [sys.argv[0]]
    pytest_args.extend(
        arg
        for arg in sys.argv
        if not arg.startswith("--config") and not arg.startswith("--split_name")
    )
    pytest.main(pytest_args)
