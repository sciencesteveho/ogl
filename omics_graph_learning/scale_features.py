#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to scale node_feats"""

from pathlib import Path
import pickle
from typing import Any, Dict

import joblib  # type: ignore
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # type: ignore

from utils import ScalerUtils


def load_scalers(
    scaler_dir: Path,
    feat_range: int,
) -> Dict[int, MinMaxScaler]:
    """_summary_

    Args:
        scaler_dir (str): _description_
        feat_range (int): _description_

    Returns:
        Dict[int, MinMaxScaler]: _description_
    """
    return {
        i: joblib.load(scaler_dir / f"feat_{i}_scaler.pt") for i in range(feat_range)
    }


def scale_node_features(
    node_feat: np.ndarray,
    scalers: Dict[int, MinMaxScaler],
    feat_range: int,
) -> np.array:
    """_summary_

    Args:
        node_feat (np.array): _description_
        scalers (Dict[int, MinMaxScaler]): _description_
        feat_range (int): _description_

    Returns:
        np.array: _description_
    """
    if type(node_feat) == list:
        node_feat = np.array(node_feat)
    for i in range(feat_range):
        node_feat[:, i] = (
            scalers[i].transform(node_feat[:, i].reshape(-1, 1)).reshape(1, -1)[0]
        )
    return node_feat


def save_scaled_graph(
    graph: Dict[str, Any],
    split_path: Path,
    prefix: str,
) -> None:
    """Save the scaled graph to a pickle file"""
    with open(split_path / f"{prefix}_scaled.pkl", "wb") as output:
        pickle.dump(graph, output, protocol=4)


def main() -> None:
    """Main function to scale graph node features"""
    # parse args and unpack params
    (
        _,
        split_path,
        scaler_dir,
        _,
        graphdir_prefix,
        filename,
    ) = ScalerUtils._handle_scaler_prep()

    feat_range = 39  # set up which node feats are continuous, and thus should be scaled

    # load scalers into dict
    scalers = load_scalers(
        scaler_dir=scaler_dir,
        feat_range=feat_range,
    )

    # load all tissue graph
    _, g = ScalerUtils._load_graph_data(graphdir_prefix=graphdir_prefix)

    # scale node features
    g["node_feat"] = scale_node_features(
        node_feat=g["node_feat"],
        scalers=scalers,
        feat_range=feat_range,
    )

    # save scaled graph
    save_scaled_graph(graph=g, split_path=split_path, prefix=filename)


if __name__ == "__main__":
    main()
