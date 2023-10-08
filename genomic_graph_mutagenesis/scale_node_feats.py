#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to scale node_feats"""

import argparse
import pickle

import joblib
import numpy as np

from utils import dir_check_make
from utils import parse_yaml


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
    params = parse_yaml(args.experiment_config)

    # set up variables for params to improve readability
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]

    # create directory for experiment specific scalers
    graph_dir = f"{working_directory}/{experiment_name}/graphs"
    scaler_dir = f"{working_directory}/{experiment_name}/data_scaler"

    # load scalers into dict
    scalers = {i: joblib.load(f"{scaler_dir}/feat_{i}_scaler.pt") for i in range(0, 39)}

    # load all tissue graph
    with open(f"{graph_dir}/{experiment_name}_{args.graph_type}_graph.pkl", "rb") as f:
        g = pickle.load(f)

    node_feat = g["node_feat"]
    if type(node_feat) == list:
        node_feat = np.array(node_feat)
    for i in range(0, 39):
        node_feat[:, i] = (
            scalers[i].transform(node_feat[:, i].reshape(-1, 1)).reshape(1, -1)[0]
        )
    g["node_feat"] = node_feat
    with open(
        f"{graph_dir}/{experiment_name}_{args.graph_type}_graph_scaled.pkl", "wb"
    ) as output:
        pickle.dump(g, output, protocol=4)


if __name__ == "__main__":
    main()
