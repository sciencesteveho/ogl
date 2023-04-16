#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]
#

"""Get train / test / val splits for nodes in graphs and generate targets for
training the network.
"""

import argparse
import csv
import os
import pickle
from typing import Any, Dict, List, Tuple

from cmapPy.pandasGEXpress.parse_gct import parse
import networkx as nx
import numpy as np
import pandas as pd

from utils import parse_yaml, time_decorator


DATA_SPLITS = ["train", "test", "validation"]

TISSUE_KEYS = {
    "mammary": (
        "breast_mammary_tissue",
        "breast",
        "breast_mammary_tissue",
    ),
    "hippocampus": (
        "brain_hippocampus",
        "brain_cortex",
        "brain_hippocampus",
    ),
    "left_ventricle": (
        "heart_left_ventricle",
        "heart_ventricle",
        "heart_left_ventricle",
    ),
    "liver": (
        "liver",
        "liver",
        "liver",
    ),
    "lung": (
        "lung",
        "lung",
        "lung"
    ),
    "pancreas": (
        "pancreas",
        "pancreas",
        "pancreas"
    ),
    "skeletal_muscle": (
        "muscle_skeletal",
        "muscle_skeletal",
        "muscle_skeletal"
    ),
}  # tissue: (tpm_key, protein_key, filtered_tpm_filename)


def genes_from_gff(gff: str) -> List[str]:
    """Get list of gtex genes from GFF file"""
    with open(gff, newline = '') as file:
        return {
            line[3]: line[0] for line in csv.reader(file, delimiter='\t')
            if line[0] not in ['chrX', 'chrY', 'chrM']
            }


@time_decorator(print_args=False)
def _chr_split_train_test_val(
    genes,
    test_chrs,
    val_chrs,
    tissue_append: bool = False,
):
    """
    Create a list of training, split, and val IDs
    """
    if tissue_append:
        return {
            "train": [
                f"{gene}_{tissue}"
                for gene in genes
                if genes[gene] not in test_chrs + val_chrs
                for tissue in TISSUE_KEYS
            ],
            "test": [
                f"{gene}_{tissue}"
                for gene in genes
                if genes[gene] in test_chrs
                for tissue in TISSUE_KEYS
            ],
            "validation": [
                f"{gene}_{tissue}"
                for gene in genes
                if genes[gene] in val_chrs
                for tissue in TISSUE_KEYS
            ],
        }
    else:
        return {
            "train": [
                gene for gene in genes if genes[gene] not in test_chrs + val_chrs
            ],
            "test": [
                gene for gene in genes if genes[gene] in test_chrs
            ],
            "validation": [
                gene for gene in genes if genes[gene] in val_chrs
            ],
        }


def _tpm_all_tissue_median(gct_file):
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT"""
    df = parse(gct_file).data_df
    median_series = pd.Series(df.median(axis=1), name="all_tissues").to_frame()
    median_series.to_pickle("gtex_tpm_median_across_all_tissues.pkl")


def _protein_abundance_all_tissue_median(protein_file: str):
    """
    For now, "BRAIN-CORTEX" values are being used for hippocampus. We choose cortex due
    to similarites shown between the tissues in GTEx Consortium, Science, 2020.
    Values are log2 so we inverse log them (2^x)
    """
    df = pd.read_csv(protein_file, sep=",", index_col="gene.id.full").drop(
        columns=["gene.id"]
    )

    df = df.apply(np.exp2).fillna(0)  # relative abundances are log2
    return pd.Series(df.median(axis=1), name="all_tissues").to_frame()


def _protein_std_dev_and_mean(protein_median_file: str) -> pd.DataFrame:
    """Get means and standard deviation for protein abundance for specific tissue
    Calulates fold change of median of tissue relative to all_tissue_median
    """
    tissues = [
        "Heart Ventricle",
        "Brain Cortex",
        "Breast",
        "Liver",
        "Lung",
        "Muscle Skeletal",
        "Pancreas",
    ]
    df = pd.read_csv(
        protein_median_file,
        sep=",",
        index_col="gene.id.full",
        usecols=["gene.id.full"] + tissues,
    )

    return df.apply(np.exp2).fillna(0)  # relative abundances are log2


@time_decorator(print_args=False)
def _fold_change_median(tissue_df, all_median_df, type=None) -> pd.DataFrame:
    """_lorem"""
    if type == "tpm":
        regex = (("- ", ""), ("-", ""), ("(", ""), (")", ""), (" ", "_"))
    else:
        regex = ((" ", "_"), ("", ""))  # second added as placeholder so we can use *r

    df = pd.concat([tissue_df, all_median_df], axis=1)
    for tissue in list(df.columns)[:-1]:
        tissue_rename = tissue.casefold()
        for r in regex:
            tissue_rename = tissue_rename.replace(*r)
        df.rename(columns={f"{tissue}": f"{tissue_rename}"}, inplace=True)
        df[f"{tissue_rename}_foldchange"] = (df[f"{tissue_rename}"] + 0.01) / (
            df["all_tissues"] + 0.01
        )  # add .01 to TPM to avoid negative infinity
    return df.apply(lambda x: np.log1p(x))


@time_decorator(print_args=False)
def _get_dict_with_target_array(
    split_dict, tissue, tpmkey, prokey, tpm_targets_df, protein_targets_df
):
    """_lorem"""
    new = {}
    for gene in split_dict:
        new[gene] = np.array(
            [
                tpm_targets_df[tpmkey].loc[gene],  # median tpm in the tissue
                tpm_targets_df[tpmkey + "_foldchange"].loc[gene],  # fold change
                protein_targets_df[prokey].loc[gene]
                if gene in protein_targets_df.index
                else -1,
                protein_targets_df[prokey + "_foldchange"].loc[gene]
                if gene in protein_targets_df.index
                else -1,
            ]
        )
    return new


@time_decorator(print_args=False)
def _tissue_dicts(
    tissue_params: dict,
    split: dict,
    tpm_targets_df: pd.DataFrame,
    protein_targets_df: pd.DataFrame,
):
    """_lorem"""
    all_dict = {}
    for idx, tissue in enumerate(tissue_params):
        all_dict[tissue] = _get_dict_with_target_array(
            split_dict=split,
            tissue=tissue,
            tpmkey=tissue_params[tissue][0],
            prokey=tissue_params[tissue][1],
            tpm_targets_df=tpm_targets_df,
            protein_targets_df=protein_targets_df,
        )
    return all_dict


def tissue_targets(
    split: dict,
    tissue_params: dict,
    tpm_pkl: str,
    tpm_median_file: str,
    protein_file: str,
    protein_median_file: str,
):
    """
    """

    # proteins
    pro_median_df = _protein_std_dev_and_mean(protein_median_file)
    pro_all_median = _protein_abundance_all_tissue_median(protein_file)
    protein_targets_df = _fold_change_median(
        pro_median_df, pro_all_median, type="protein"
    )

    # expression TPMs
    with open(tpm_pkl, "rb") as file:
        tpm_all_median = pickle.load(file)

    tpm_median_df = parse(tpm_median_file).data_df
    tpm_targets_df = _fold_change_median(tpm_median_df, tpm_all_median, type="tpm")

    targets = {
        data_split: _tissue_dicts(
            tissue_params,
            split[data_split],
            tpm_targets_df,
            protein_targets_df,
        )
        for data_split in DATA_SPLITS
    }

    return targets


def filtered_targets(
    tissue_params,
    targets,
):
    """Filters a dict of every possible target (all 56200 genes)
    Takes the pre-tpm filtered list of genes in each directory and concats into
    a list Keeps targets if they exist within this concatenated list.
    """

    def filtered_genes(tpm_filtered_genes: str) -> List[str]:
        with open(tpm_filtered_genes, newline="") as file:
            return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]

    for idx, tissue in enumerate(tissue_params):
        if idx == 0:
            genes = filtered_genes(f"{tissue}/gene_regions_tpm_filtered.bed")
        else:
            update_genes = filtered_genes(f"{tissue}/gene_regions_tpm_filtered.bed")
            genes += update_genes

    for key in targets.keys():
        targets[key] = {
            gene: targets[key][gene] for gene in targets[key].keys() if gene in genes
        }
    return targets


def _get_split_by_node_idxs(graph_dir, tissue, split, targets):
    tissue_specific_targets = {}
    with open(f"{graph_dir}/{tissue})gene_idxs.pkl", "rb") as file:
        node_idxs = pickle.load(file)

    for training_split in split:
        tissue_specific_targets[training_split] = {
            gene: targets[training_split][gene] for gene in node_idxs.keys()
        }

def main() -> None:
    """Pipeline to generate dataset split and target values"""

    gene_gtf = "shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
    test_chrs = ["chr8", "chr9"]
    val_chrs = ["chr7", "chr13"]

    root_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess"
    tpm_pkl = "gtex_tpm_median_across_all_tissues.pkl"
    tpm_median_file = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    protein_file = "protein_relative_abundance_all_gtex.csv"
    protein_median_file = "protein_relative_abundance_median_gtex.csv"

    save_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs"
    chr_split_dictionary = f"{save_dir}/graph_partition_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}.pkl"

    split = _chr_split_train_test_val(
        gene_gtf=genes_from_gff(gene_gtf),
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )

    with open(chr_split_dictionary, "wb") as output:
        pickle.dump(split, output)

    # get targets - 313502 /31283/28231 train/test/validation, total = 373016
    targets = tissue_targets(
        split=split,
        tissue_params=TISSUE_KEYS,
        tpm_pkl=f"{root_dir}/shared_data/{tpm_pkl}",
        tpm_median_file=f"{root_dir}/shared_data/{tpm_median_file}",
        protein_file=f"{root_dir}/shared_data/{protein_file}",
        protein_median_file=f"{root_dir}/shared_data/{protein_median_file}",
    )

    # save targets
    with open("graphs/all_targets_unfiltered.pkl", "wb") as output:
        pickle.dump(targets, output)


if __name__ == "__main__":
    main()