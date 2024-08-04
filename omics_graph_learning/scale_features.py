#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to scale node features in the graph. To avoid data leakage, we fit the
scalers only on thenode idxs that occur in the training set."""


from multiprocessing import Pool
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Tuple, Union

import joblib  # type: ignore
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore

from scaled_feature_accuracy import ScaledFeatureAccuracy
from utils import dir_check_make
from utils import get_physical_cores
from utils import ScalerUtils
from utils import setup_logging

logger = setup_logging()
CORES = get_physical_cores()


def inverse_transform_features(
    scaled_features: np.ndarray, scalers: Dict[int, RobustScaler], feat_range: int
) -> np.ndarray:
    """Reverse the scaling of node features"""
    original_features = scaled_features.copy()
    for i in range(feat_range):
        original_features[:, i] = (
            scalers[i]
            .inverse_transform(scaled_features[:, i].reshape(-1, 1))
            .reshape(-1)
        )
    return original_features


def scale_graph(scaler_utility: ScalerUtils) -> Dict:
    """Scale graph node features"""
    dir_check_make(scaler_utility.scaler_dir)

    # load data
    split = scaler_utility.load_split()
    idxs = scaler_utility.load_idxs()
    graph = scaler_utility.load_graph()
    node_features = graph["node_feat"]

    # exclude validation and test genes from fitting scalers
    skip_idxs = test_and_val_genes(split=split, idxs=idxs)

    # feats are set up so that the last_n are one-hot encoded, so we only scale
    # the continous feats
    feat_range = _get_continuous_feat_range(
        node_features=node_features, onehot_feats=scaler_utility.onehot_node_feats
    )

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

    # run accuracy check
    feature_test_suite = ScaledFeatureAccuracy(
        original=original_node_feat,
        scaled=scaled_node_feat,
        split=split,
        idxs=idxs,
        continous_feat_range=feat_range,
        logger=logger,
        scalers=scalers,
    )
    test_result = feature_test_suite.run_all_tests()
    if not test_result:
        logger.error("Node feature scaling failed accuracy tests.")
        sys.exit(1)

    # update graph with scaled features
    graph["node_feat"] = scaled_node_feat
    return graph


def test_and_val_genes(split: Dict[str, List[str]], idxs: Dict[str, int]) -> List[int]:
    """Exclude validation and test genes from fitting scalers"""
    exclude = split["validation"] + split["test"]
    return [idxs[gene] for gene in exclude if gene in idxs]


def _get_continuous_feat_range(node_features: np.ndarray, onehot_feats: int) -> int:
    """Subtract the number of onehot encoded features (at the end of the array)
    from the total number of features"""
    return node_features.shape[1] - onehot_feats


def scaler_fit_task(scaler_fit_arguments: Tuple[int, np.ndarray, Path]) -> int:
    """Fits scalers according to node idx."""
    feat, node_feat, scaler_dir = scaler_fit_arguments
    # scaler = MinMaxScaler()
    scaler = RobustScaler(quantile_range=(5, 95), unit_variance=False)
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
        skip_idxs (List[int]): Indices to skip (e.g., validation or test set
        indices).
        scaler_dir (Path): Directory to save the fitted scalers.
        skip_last_n (int): Number of features to skip at the end (default: 2).
        n_jobs (int): Number of parallel jobs to run. If None, uses all
        available CPU cores.
    """
    node_feat = np.delete(node_features, skip_idxs, axis=0)  # remove validation/test

    with Pool(processes=n_jobs) as pool:
        scaler_fit_arguments = [
            (feat, node_feat, scaler_dir) for feat in range(feat_range)
        ]
        pool.map(scaler_fit_task, scaler_fit_arguments)

    logger.info(
        f"Scalers fit and saved to {scaler_dir} for {feat_range} features w/ {n_jobs} parallel jobs."
    )


def load_scalers(
    scaler_dir: Path,
    feat_range: int,
) -> Dict[int, Union[RobustScaler, MinMaxScaler]]:
    """Load pre-fit scalers into a dictionary by feature index"""
    return {
        i: joblib.load(scaler_dir / f"feat_{i}_scaler.joblib")
        for i in range(feat_range)
    }


def scale_feature_task(
    scale_feats_arguments: Tuple[np.ndarray, Union[RobustScaler, MinMaxScaler], int]
) -> Tuple[int, np.ndarray]:
    """Scale a single feature using the provided scaler."""
    feature, scaler, index = scale_feats_arguments
    scaled_feature = scaler.transform(feature.reshape(-1, 1)).reshape(1, -1)[0]
    return index, scaled_feature


def scale_node_features(
    node_feat: np.ndarray,
    scalers: Dict[int, Union[RobustScaler, MinMaxScaler]],
    feat_range: int,
    n_jobs: int = CORES,
) -> np.ndarray:
    """Scale node features using pre-fit scalers in parallel"""
    scaled_node_feat = node_feat.copy()  # copy to avoid modifying original

    # check the min, max, and mean of the features before scaling
    logger.info(
        f"Before scaling - min: {node_feat[:, :feat_range].min(axis=0)[:5]}, max: {node_feat[:, :feat_range].max(axis=0)[:5]}, mean: {node_feat[:, :feat_range].mean(axis=0)[:5]}"
    )

    with Pool(processes=n_jobs) as pool:
        scale_feats_arguments = [
            (node_feat[:, i], scalers[i], i) for i in range(feat_range)
        ]
        results = pool.map(scale_feature_task, scale_feats_arguments)

    # check the min, max, and mean of the features after scaling
    logger.info(
        f"After scaling - min: {scaled_node_feat[:, :feat_range].min(axis=0)[:5]}, max: {scaled_node_feat[:, :feat_range].max(axis=0)[:5]}, mean: {scaled_node_feat[:, :feat_range].mean(axis=0)[:5]}"
    )

    # sort results by feature index and update scaled_node_feat
    for index, scaled_feature in sorted(results):
        scaled_node_feat[:, index] = scaled_feature

    return scaled_node_feat


def main() -> None:
    """Main function to scale graph node features"""

    # parse scaler_fit_arguments and unpack params
    scaler_utility = ScalerUtils()

    # scale graph
    scaled_graph = scale_graph(scaler_utility)

    # save scaled graph
    final_graph = scaler_utility.split_dir / f"{scaler_utility.file_prefix}_scaled.pkl"
    with open(final_graph, "wb") as output:
        pickle.dump(scaled_graph, output, protocol=4)


if __name__ == "__main__":
    main()
