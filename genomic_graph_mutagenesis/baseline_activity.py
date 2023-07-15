#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""Baseline predictors to compare against the performance of the GNN model.

The first predictor is based on a random assignment of activity, from a
distribution of expression activity across all tissues in GTEx. The second is
based on the average activity of that specific gene across all samples adapted
from Schrieber et al., Genome Biology, 2020."""

import pickle
from typing import List, Tuple

from cmapPy.pandasGEXpress.parse_gct import parse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def _tpm_all_tissue_median(expression_gct: str) -> np.ndarray:
    """Get the average (not median!) expression for each gene in GTEx across all
    samples. Formally, the average activity is the summed expression at each
    gene across all samples, divided by the total number of samples.

    Args:
        expression_gct (str): path to GTEx expression .gct

    Returns:
        np.ndarray: array with average activity for each gene
    """
    df = parse(expression_gct).data_df
    sample_count = df.astype(bool).sum(axis=1)
    summed_activity = pd.Series(df.sum(axis=1), name="all_tissues").to_frame()
    summed_activity.to_pickle("baseline_activitiy_gtex_expression.pkl")
    return summed_activity.div(sample_count, axis=0).fillna(0).values


def _get_targets(
    split: str,
    targets: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        list(targets[split].keys()),
        [x[0] for x in np.array(list(targets[split].values()))],
    )


def _avg_activity_baseline_predictions(labels, s):
    return [s["all_tissues"][label.split("_")[0]] for label in labels]


def _nonzero_tpms(tpms: List(List(float))) -> np.ndarray:
    """_summary_

    Args:
        expression_gct (str): _description_

    Returns:
        np.ndarray: _description_
    """


def main(
    expression_gct: str,
) -> None:
    """Main function"""
    # get predictions as the average tpm for that gene across all tissues
    average_activity = _tpm_all_tissue_median(expression_gct)
    # with open('average_activity_before_transform.pkl', 'wb') as f:
    #     pickle.dump(average_activity, f)

    with open("baseline_activity_gtex_expression.pkl", "rb") as f:
        average_activity = pickle.load(f)

    y_pred = np.log2(average_activity + 0.25)  # add 0.01 to avoid log(0)
    s = y_pred.to_dict()

    with open(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/training_targets.pkl",
        "rb",
    ) as f:
        targets = pickle.load(f)

    test_labels, test_true = _get_targets("test", targets)
    val_labels, val_true = _get_targets("validation", targets)
    train_labels, train_true = _get_targets("train", targets)

    train_preds = _avg_activity_baseline_predictions(train_labels, s)
    test_preds = _avg_activity_baseline_predictions(test_labels, s)
    val_preds = _avg_activity_baseline_predictions(val_labels, s)

    train_error = mean_squared_error(train_true, train_preds, squared=False)
    test_error = mean_squared_error(test_true, test_preds, squared=False)
    val_error = mean_squared_error(val_true, val_preds, squared=False)

    print(f"Train error: {train_error}")
    print(f"Test error: {test_error}")
    print(f"Validation error: {val_error}")


if __name__ == "__main__":
    main(expression_gct="GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct")
