#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to scale node_feats"""

import argparse
import pickle
from typing import Dict

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import utils


def load_graph(
    graph_dir: str,
    experiment_name: str,
    graph_type: str,
):
    """_summary_

    Args:
        graph_dir (str): _description_
        experiment_name (str): _description_
        graph_type (str): _description_

    Returns:
        _type_: _description_
    """
    graph_path = f"{graph_dir}/{experiment_name}_{graph_type}_graph.pkl"
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def load_scalers(
    scaler_dir: str,
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
        i: joblib.load(f"{scaler_dir}/feat_{i}_scaler.pt") for i in range(feat_range)
    }


def scale_node_features(
    node_feat: np.array,
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
    graph,
    graph_dir,
    experiment_name,
    graph_type,
):
    """_summary_

    Args:
        graph (_type_): _description_
        graph_dir (_type_): _description_
        experiment_name (_type_): _description_
        graph_type (_type_): _description_
    """
    scaled_graph_path = f"{graph_dir}/{experiment_name}_{graph_type}_graph_scaled.pkl"
    with open(scaled_graph_path, "wb") as output:
        pickle.dump(graph, output, protocol=4)


def main() -> None:
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--graph_type",
        type=str,
        default="full",
        help="Graph type to use (full or base)",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    args = parser.parse_args()
    params = utils.parse_yaml(args.experiment_config)

    # set up variables for params to improve readability
    feat_range = 39  # set up which node feats are continuous, and thus should be scaled
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]

    # create directory for experiment specific scalers
    graph_dir = f"{working_directory}/{experiment_name}/graphs"
    scaler_dir = f"{working_directory}/{experiment_name}/data_scaler"

    # load scalers into dict
    scalers = load_scalers(
        scaler_dir=scaler_dir,
        feat_range=feat_range,
    )

    # load all tissue graph
    g = load_graph(graph_dir, experiment_name, args.graph_type)

    # scale node features
    g["node_feat"] = scale_node_features(
        g["node_feat"],
        scalers,
        feat_range=feat_range,
    )

    # save scaled graph
    save_scaled_graph(g, graph_dir, experiment_name, args.graph_type)


if __name__ == "__main__":
    main()
