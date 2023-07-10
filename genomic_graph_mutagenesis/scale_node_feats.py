#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to scale node_feats"""

import argparse
import pickle

import joblib
import numpy as np

from dataset_split import TISSUE_KEYS
from utils import dir_check_make

root_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess"
scale_dir = f"{root_dir}/data_scaler"


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
    args = parser.parse_args()

    graph_dir = f"{root_dir}/graphs"
    out_dir = f"{graph_dir}/scaled"
    dir_check_make(out_dir)

    scalers = {i: joblib.load(f"{scale_dir}/feat_{i}_scaler.pt") for i in range(0, 36)}

    with open(f"{root_dir}/graphs/all_tissue_{args.graph_type}_graph.pkl", "rb") as f:
        g = pickle.load(f)
    node_feat = g["node_feat"]
    if type(node_feat) == list:
        node_feat = np.array(node_feat)
    for i in range(0, 36):
        node_feat[:, i] = (
            scalers[i].transform(node_feat[:, i].reshape(-1, 1)).reshape(1, -1)[0]
        )
    g["node_feat"] = node_feat
    with open(
        f"{out_dir}/all_tissue_{args.graph_type}_graph_scaled.pkl", "wb"
    ) as output:
        pickle.dump(g, output)


if __name__ == "__main__":
    main()
