#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Fit a scaler for node feats"""

from contextlib import ExitStack
from pathlib import Path
import pickle
from typing import Dict, List

import joblib  # type: ignore
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from utils import dir_check_make
from utils import ScalerUtils


def load_training_split(partition: Path) -> Dict[str, list]:
    """Load the training split from a file."""
    with open(partition, "rb") as file:
        split = pickle.load(file)
    return split


def fit_scaler_and_save(
    node_features: int,
    skip_idxs: List[int],
    feat: int,
    scaler_dir: str,
) -> None:
    """Fit the scaler and save to file. Removes any genes in the validation or
    test sets before scaling.
    """
    scaler = MinMaxScaler()
    node_feat = np.delete(node_features, skip_idxs, axis=0)
    scaler.fit(node_feat[:, feat].reshape(-1, 1))
    with ExitStack() as stack:
        file = stack.enter_context(open(f"{scaler_dir}/feat_{feat}_scaler.pt", "wb"))
        joblib.dump(scaler, file)


def main() -> None:
    """Main function"""
    # parse args and unpack params
    (
        feat,
        split_path,
        scaler_dir,
        _,
        graphdir_prefix,
        _,
    ) = ScalerUtils._handle_scaler_prep()

    dir_check_make(scaler_dir)

    # set up other vars from functions
    split = load_training_split(partition=split_path / "training_targets_split.pkl")
    exclude = split["validation"] + split["test"]
    idxs, g = ScalerUtils._load_graph_data(graphdir_prefix=f"{graphdir_prefix}")
    skip_idxs = [idxs[gene] for gene in exclude if gene in idxs]
    node_features = g["node_feat"]

    # fit scalers!
    fit_scaler_and_save(
        node_features=node_features,
        skip_idxs=skip_idxs,
        feat=feat,
        scaler_dir=scaler_dir,
    )


if __name__ == "__main__":
    main()
