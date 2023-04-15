#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]

"""Fit a scaler for node feats"""

import argparse
import joblib
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from dataset_split import _chr_split_train_test_val, genes_from_gff

# root_dir='/ocean/projects/bio210019p/stevesho/data/preprocess'
# genes = gene_list_from_graphs(root_dir=root_dir, tissue=args.tissue)

TISSUES = [
    "hippocampus",
    "liver",
    "lung",
    "mammary",
    "pancreas",
    "skeletal_muscle",
    "left_ventricle",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--feat", type=int, required=True)
    args = parser.parse_args()

    test_chrs = ["chr8", "chr9"]
    val_chrs = ["chr7", "chr13"]

    gene_gtf = "shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
    graph_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs"
    output_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/data_scaler"

    genes = genes_from_gff(gene_gtf)

    # split genes by chr holdouts
    split = _chr_split_train_test_val(
        genes=genes,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )

    exclude = split["validation"] + split["test"]

    scaler = MinMaxScaler()
    for tissue in TISSUES:
        with open(f"{graph_dir}/{tissue}/{tissue}_gene_idxs.pkl", "rb") as file:
            idxs = pickle.load(file)
        with open(f"{graph_dir}/{tissue}/{tissue}_full_graph.pkl", "rb") as f:
            g = pickle.load(f)
        skip_idxs = [idxs[gene] for gene in exclude if gene in idxs]
        node_feat = g["node_feat"]
        for idx in skip_idxs:
            node_feat = np.delete(
                node_feat, idx, axis=0
            )  # do not use test or val idxs for fitting scaler
        scaler.partial_fit(node_feat[:, args.feat].reshape(-1, 1))

    ### save
    joblib.dump(scaler, f"{output_dir}/feat_{args.feat}_scaler.pt")


if __name__ == "__main__":
    main()
