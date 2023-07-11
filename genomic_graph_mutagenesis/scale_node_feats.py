#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to scale node_feats"""

import argparse
import pickle

import joblib
import numpy as np

from utils import dir_check_make


def main(root_dir: str) -> None:
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--graph_type",
        type=str,
        default="full",
        help="Graph type to use (full or base)",
    )
    args = parser.parse_args()

    # set up directories
    graph_dir = f"{root_dir}/graphs"
    out_dir = f"{graph_dir}/scaled"
    scale_dir = f"{root_dir}/data_scaler"
    dir_check_make(out_dir)

    # load scalers into dict
    scalers = {i: joblib.load(f"{scale_dir}/feat_{i}_scaler.pt") for i in range(0, 39)}

    # load all tissue graph
    with open(f"{root_dir}/graphs/all_tissue_{args.graph_type}_graph.pkl", "rb") as f:
        g = pickle.load(f, protocol=4)

    node_feat = g["node_feat"]
    if type(node_feat) == list:
        node_feat = np.array(node_feat)
    for i in range(0, 39):
        node_feat[:, i] = (
            scalers[i].transform(node_feat[:, i].reshape(-1, 1)).reshape(1, -1)[0]
        )
    g["node_feat"] = node_feat
    with open(
        f"{out_dir}/all_tissue_{args.graph_type}_graph_scaled.pkl", "wb"
    ) as output:
        pickle.dump(g, output, protocol=4)


if __name__ == "__main__":
    main(root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess")
