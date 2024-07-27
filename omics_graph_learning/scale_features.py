#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to scale node features in the graph. We use the simple MinMaxScaler
because other scalers sometimes cause out of bounds errors due to the range that
genomic data can exist across. To avoid leakage, we fit the scalers only on the
node idxs that occur in the training set."""


from multiprocessing import Pool
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

import joblib  # type: ignore
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from utils import dir_check_make
from utils import get_physical_cores
from utils import ScalerUtils

CORES = get_physical_cores()


def scaler_fit_task(scaler_fit_arguments: Tuple[int, np.ndarray, Path]) -> int:
    """Fits scalers according to node idx."""
    feat, node_feat, scaler_dir = scaler_fit_arguments
    scaler = MinMaxScaler()
    scaler.fit(node_feat[:, feat].reshape(-1, 1))
    scaler_path = scaler_dir / f"feat_{feat}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    return feat


def fit_scalers(
    node_features: np.ndarray,
    skip_idxs: List[int],
    scaler_dir: Path,
    feat_range: int,
    n_jobs: int = CORES,
) -> None:
    """Fit the scaler and save to file in parallel. Removes any genes in the
    validation or test sets before scaling.

    Arguments:
        node_features (np.ndarray): The input features to scale.
        skip_idxs (List[int]): Indices to skip (e.g., validation or test set indices).
        scaler_dir (Path): Directory to save the fitted scalers.
        skip_last_n (int): Number of features to skip at the end (default: 2).
        n_jobs (int): Number of parallel jobs to run. If None, uses all available CPU cores.
    """
    node_feat = np.delete(node_features, skip_idxs, axis=0)  # remove validation/test

    with Pool(processes=n_jobs) as pool:
        scaler_fit_arguments = [
            (feat, node_feat, scaler_dir) for feat in range(feat_range)
        ]
        pool.map(scaler_fit_task, scaler_fit_arguments)

    print(
        f"Scalers fitted and saved to {scaler_dir} for {feat_range} features w/ {n_jobs} parallel jobs."
    )


def load_scalers(
    scaler_dir: Path,
    feat_range: int,
) -> Dict[int, MinMaxScaler]:
    """Load pre-fit scalers into a dictionary by feature index"""
    return {
        i: joblib.load(scaler_dir / f"feat_{i}_scaler.joblib")
        for i in range(feat_range)
    }


def scale_feature_task(
    scale_feats_arguments: Tuple[np.ndarray, MinMaxScaler, int]
) -> Tuple[int, np.ndarray]:
    """Scale a single feature using the provided scaler."""
    feature, scaler, index = scale_feats_arguments
    scaled_feature = scaler.transform(feature.reshape(-1, 1)).reshape(1, -1)[0]
    return index, scaled_feature


def scale_node_features(
    node_feat: np.ndarray,
    scalers: Dict[int, MinMaxScaler],
    feat_range: int,
    n_jobs: int = CORES,
) -> np.ndarray:
    """Scale node features using pre-fit scalers in parallel"""
    scaled_node_feat = node_feat.copy()  # copy to avoid modifying original

    with Pool(processes=n_jobs) as pool:
        scale_feats_arguments = [
            (node_feat[:, i], scalers[i], i) for i in range(feat_range)
        ]
        results = pool.map(scale_feature_task, scale_feats_arguments)

    # sort results by feature index and update scaled_node_feat
    for index, scaled_feature in sorted(results):
        scaled_node_feat[:, index] = scaled_feature

    return scaled_node_feat


def verify_feature_order(
    original: np.ndarray, scaled: np.ndarray, feat_range: int
) -> bool:
    """Check that the original and scaled have their node features in the
    same order."""
    for i in range(feat_range):
        original_col = original[:, i]
        scaled_col = scaled[:, i]

        # get the order of the original and scaled columns
        original_order = np.argsort(original_col)
        scaled_order = np.argsort(scaled_col)

        if not np.array_equal(original_order, scaled_order):
            print(f"Order mismatch detected in feature {i}")
            return False
    return True


def main() -> None:
    """Main function to scale graph node features"""
    # parse scaler_fit_arguments and unpack params
    scaler_utility = ScalerUtils()
    dir_check_make(scaler_utility.scaler_dir)

    # load data
    split = scaler_utility.load_split()
    idxs = scaler_utility.load_idxs()
    graph = scaler_utility.load_graph()

    # exclude validation and test genes from fitting scalers
    exclude = split["validation"] + split["test"]
    skip_idxs = [idxs[gene] for gene in exclude if gene in idxs]
    node_features = graph["node_feat"]

    # feats are set up so that the last_n are one-hot encoded, so we only the
    # continuous features
    feat_range = node_features.shape[1] - scaler_utility.onehot_node_feats

    # fit scalers!
    fit_scalers(
        node_features=node_features,
        skip_idxs=skip_idxs,
        scaler_dir=scaler_utility.scaler_dir,
        feat_range=feat_range,
    )

    # load scalers into dict
    scalers = load_scalers(
        scaler_dir=scaler_utility.scaler_dir,
        feat_range=feat_range,
    )

    # scale node features
    original_node_feat: np.ndarray = graph[
        "node_feat"
    ].copy()  # save original shape for test
    scaled_node_feat: np.ndarray = scale_node_features(
        node_feat=graph["node_feat"],
        scalers=scalers,
        feat_range=feat_range,
    )

    # verify feature order
    order_preserved: bool = verify_feature_order(
        original_node_feat, scaled_node_feat, feat_range
    )
    print(f"Feature order preserved: {order_preserved}")

    # additional sanity checks
    assert (
        original_node_feat.shape == scaled_node_feat.shape
    ), "Shapes don't match after scaling!"
    assert np.array_equal(
        original_node_feat[:, feat_range:], scaled_node_feat[:, feat_range:]
    ), "One-hot encoded features changed!"

    # save scaled graph
    graph["node_feat"] = scaled_node_feat  # update
    final_graph = scaler_utility.split_dir / f"{scaler_utility.file_prefix}_scaled.pkl"
    with open(final_graph, "wb") as output:
        pickle.dump(graph, output, protocol=4)


if __name__ == "__main__":
    main()
