#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]

"""Fit a scaler for node feats"""

import argparse
import pickle

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dataset_split import _chr_split_train_test_val
from dataset_split import _genes_from_gff
from utils import dir_check_make
from utils import parse_yaml

# from sklearn.preprocessing import StandardScaler


def main(
    gene_gtf: str,
    test_chrs: list = ["chr8", "chr9"],
    val_chrs: list = ["chr7", "chr13"],
) -> None:
    """_summary_

    Args:
        gene_gtf (str): /path/to/gene_gtf
        test_chrs (list, optional): Chrs to withhold for the tet set. Defaults to ["chr8", "chr9"].
        val_chrs (list, optional): Chrs to withold for the validation set_. Defaults to ["chr7", "chr13"].
    """
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
    params = parse_yaml(args.experiment_config)

    # set up variables for params to improve readability
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]

    # create directory for experiment specific scalers
    graph_dir = f"{working_directory}/{experiment_name}/graphs"
    scaler_dir = f"{working_directory}/{experiment_name}/data_scaler"
    dir_check_make(scaler_dir)

    genes = _genes_from_gff(gene_gtf)

    # split genes by chr holdouts
    split = _chr_split_train_test_val(
        genes=genes,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )
    exclude = split["validation"] + split["test"]

    with open(
        f"{graph_dir}/{experiment_name}_{args.graph_type}_graph_idxs.pkl", "rb"
    ) as file:
        idxs = pickle.load(file)
    with open(f"{graph_dir}/{experiment_name}_{args.graph_type}_graph.pkl", "rb") as f:
        g = pickle.load(f)
    skip_idxs = [idxs[gene] for gene in exclude if gene in idxs]
    node_feat = g["node_feat"]
    node_feat = np.delete(
        node_feat, skip_idxs, axis=0
    )  # remove test and val idxs for fitting scaler
    scaler.fit(node_feat[:, args.feat].reshape(-1, 1))

    ### save
    joblib.dump(scaler, f"{scaler_dir}/feat_{args.feat}_scaler.pt")


if __name__ == "__main__":
    main(
        gene_gtf="shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed",
    )
