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
from sklearn.preprocessing import StandardScaler

from dataset_split import _chr_split_train_test_val
from dataset_split import _genes_from_gff


def main(
    gene_gtf: str,
    graph_dir: str,
    output_dir: str,
    test_chrs: list = ["chr8", "chr9"],
    val_chrs: list = ["chr7", "chr13"],
) -> None:
    """_summary_

    Args:
        gene_gtf (str): /path/to/gene_gtf
        graph_dir (str): /path/to/graphs
        output_dir (str): directory to ouput scaler
        test_chrs (list, optional): Chrs to withhold for the tet set. Defaults to ["chr8", "chr9"].
        val_chrs (list, optional): Chrs to withold for the validation set_. Defaults to ["chr7", "chr13"].
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feat", type=int, required=True)
    args = parser.parse_args()

    genes = _genes_from_gff(gene_gtf)

    # split genes by chr holdouts
    split = _chr_split_train_test_val(
        genes=genes,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )
    exclude = split["validation"] + split["test"]

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    with open(f"{graph_dir}/all_tissue_full_graph_idxs.pkl", "rb") as file:
        idxs = pickle.load(file)
    with open(f"{graph_dir}/all_tissue_full_graph.pkl", "rb") as f:
        g = pickle.load(f)
    skip_idxs = [idxs[gene] for gene in exclude if gene in idxs]
    node_feat = g["node_feat"]
    for idx in skip_idxs:
        node_feat = np.delete(
            node_feat, idx, axis=0
        )  # do not use test or val idxs for fitting scaler
    scaler.fit(node_feat[:, args.feat].reshape(-1, 1))

    ### save
    joblib.dump(scaler, f"{output_dir}/feat_{args.feat}_scaler.pt")


if __name__ == "__main__":
    main(
        gene_gtf="shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed",
        graph_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs",
        output_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/data_scaler",
    )
