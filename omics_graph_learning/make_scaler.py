#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]

"""Fit a scaler for node feats"""

import argparse
import pickle
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import utils

# from sklearn.preprocessing import StandardScaler


def create_scaler_directory(
    experiment_name: str,
    working_directory: str,
) -> Tuple[str, str]:
    graph_dir = f"{working_directory}/{experiment_name}/graphs"
    scaler_dir = f"{working_directory}/{experiment_name}/data_scaler"
    utils.dir_check_make(scaler_dir)
    return scaler_dir, graph_dir


def load_training_split(graph_dir: str) -> Dict[str, list]:
    """Load the training split from a file."""
    with open(f"{graph_dir}/training_split.pkl", "rb") as file:
        split = pickle.load(file)
    return split


def load_graph_data(
    graph_dir: str,
    experiment_name: str,
    args: Dict[str, Any],
):
    """Load graph data from files."""
    graph_file = f"{graph_dir}/{experiment_name}_{args.graph_type}_graph.pkl"
    idxs_file = f"{graph_dir}/{experiment_name}_{args.graph_type}_graph_idxs.pkl"
    with open(idxs_file, "rb") as file:
        idxs = pickle.load(file)
    with open(graph_file, "rb") as f:
        g = pickle.load(f)
    return idxs, g


def fit_scaler_and_save(
    node_features: int,
    skip_idxs: List[str],
    feat: int,
    scaler_dir: str,
) -> None:
    """Fit the scaler and save to file. Removes any genes in the validation or
    test sets before scaling.
    """
    scaler = MinMaxScaler()
    node_feat = np.delete(node_features, skip_idxs, axis=0)
    scaler.fit(node_feat[:, feat].reshape(-1, 1))
    joblib.dump(scaler, f"{scaler_dir}/feat_{feat}_scaler.pt")


def main(gene_gtf: str, test_chrs: list = None, val_chrs: list = None) -> None:
    """_summary_

    Args:
        gene_gtf (str): /path/to/gene_gtf
        test_chrs (list, optional): Chrs to withhold for the tet set. Defaults to ["chr8", "chr9"].
        val_chrs (list, optional): Chrs to withold for the validation set_. Defaults to ["chr7", "chr13"].
    """
    if test_chrs is None:
        test_chrs = ["chr8", "chr9"]
    if val_chrs is None:
        val_chrs = ["chr7", "chr13"]
    scaler = MinMaxScaler()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--feat",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-g",
        "--graph_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    args = parser.parse_args()
    params = utils.parse_yaml(args.experiment_config)

    # set up variables for params to improve readability
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]

    # set up other vars from functions
    scaler_dir, graph_dir = create_scaler_directory(
        experiment_name=experiment_name, working_directory=working_directory
    )
    split = load_training_split(graph_dir)
    exclude = split["validation"] + split["test"]
    idxs, g = load_graph_data(
        graph_dir=graph_dir,
        experiment_name=experiment_name,
        args=args,
    )
    skip_idxs = [idxs[gene] for gene in exclude if gene in idxs]
    node_features = g["node_feat"]

    # fit scalers!
    fit_scaler_and_save(
        node_features=node_features,
        skip_idxs=skip_idxs,
        feat=args.feat,
        scaler_dir=scaler_dir,
    )


if __name__ == "__main__":
    main(gene_gtf="shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed")
