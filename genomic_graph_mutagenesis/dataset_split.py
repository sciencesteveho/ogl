#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]
#

"""Get train / test / val splits for nodes in graphs and generate targets for
training the network.
"""

import csv
import os
import pickle
from typing import Any, Dict, List, Tuple

from cmapPy.pandasGEXpress.parse_gct import parse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import filter_target_split
from utils import parse_yaml
from utils import time_decorator
from utils import TISSUES

DATA_SPLITS = ["train", "test", "validation"]


def _tpm_all_tissue_median(gct_file: str) -> None:
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT and
    saves it. Because the file is large and requires a lot of memory, we ran
    this separately from the main function.

    Args:
        gct_file (str): /path/to/gtex gct file
    """
    df = parse(gct_file).data_df
    median_series = pd.Series(df.median(axis=1), name="all_tissues").to_frame()
    median_series.to_pickle("gtex_tpm_median_across_all_tissues.pkl")


def _genes_from_gff(gff: str) -> List[str]:
    """Get list of gtex genes from GFF file

    Args:
        gff (str): /path/to/genes.gtf

    Returns:
        List[str]: genes
    """
    with open(gff, newline="") as file:
        return {
            line[3]: line[0]
            for line in csv.reader(file, delimiter="\t")
            if line[0] not in ["chrX", "chrY", "chrM"]
        }


@time_decorator(print_args=False)
def _chr_split_train_test_val(
    genes,
    test_chrs,
    val_chrs,
    tissue_append: bool = True,
) -> Dict[str, List[str]]:
    """Creates training, test, and validation splits for genes based on
    chromosome. Adds tissues to each label to differentiate targets between
    graphs

    Args:
        genes (_type_): list of all genes in genome.
        test_chrs (_type_)
        val_chrs (_type_)
        tissue_append (bool, optional): Whether or not to append tissue name to
        ID. Defaults to True.

    Returns:
        Dict[str, List[str]]: dictionary of genes split into train, test, and
        validation.
    """
    if tissue_append:
        return {
            "train": [
                f"{gene}_{tissue}"
                for gene in genes
                if genes[gene] not in test_chrs + val_chrs
                for tissue in TISSUES
            ],
            "test": [
                f"{gene}_{tissue}"
                for gene in genes
                if genes[gene] in test_chrs
                for tissue in TISSUES
            ],
            "validation": [
                f"{gene}_{tissue}"
                for gene in genes
                if genes[gene] in val_chrs
                for tissue in TISSUES
            ],
        }
    else:
        return {
            "train": [
                gene for gene in genes if genes[gene] not in test_chrs + val_chrs
            ],
            "test": [gene for gene in genes if genes[gene] in test_chrs],
            "validation": [gene for gene in genes if genes[gene] in val_chrs],
        }


def _get_protein_adbunance_median_across_all_tissues_samples(
    protein_abundance_matrix: str,
) -> pd.DataFrame:
    """Returns a pandas object with median protein abundance for each gene
    across all samples and tissues. Values are log2, so we inverse log them,
    as we transform them ourselves later.

    Args:
        protein_abundance_matrix (str): /path/to/protein_abundance_matrix.csv

    Returns:
        _type_: pandas series with median protein abundance for each gene across
        all samples and tissues
    """
    df = pd.read_csv(protein_abundance_matrix, sep=",", index_col="gene.id.full").drop(
        columns=["gene.id"]
    )

    df = df.apply(np.exp2).fillna(0)  # relative abundances are log2
    return pd.Series(df.median(axis=1), name="all_tissues").to_frame()


def _get_protein_adundance_tissue_matrix(
    protein_abundance_medians: str,
    graph: str = "tissue",
) -> pd.DataFrame:
    """Gets

    Args:
        protein_abundance_medians (str): _description_
        graph (str, optional): _description_. Defaults to "tissue".

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    if graph == "tissue":
        tissues = [
            "Heart Ventricle",
            "Brain Cortex",
            "Breast",
            "Liver",
            "Lung",
            "Muscle Skeletal",
            "Pancreas",
            "Skin Unexpo",
            "Small Intestine",
        ]
        df = pd.read_csv(
            protein_abundance_medians,
            sep=",",
            index_col="gene.id.full",
            usecols=["gene.id.full"] + tissues,
        )
    elif graph == "universal":
        df = pd.read_csv(
            protein_abundance_medians,
            sep=",",
            index_col="gene.id.full",
        )
    if graph not in ("tissue", "universal"):
        raise ValueError("Graph type must be either 'tissues' or 'universal'")

    return df.apply(np.exp2).fillna(0)  # relative abundances are log2


@time_decorator(print_args=False)
def _calculate_fold_change_from_medians(
    median_matrix,
    median_across_tissues,
    type: str = "tpm",
) -> pd.DataFrame:
    """_lorem"""
    if type == "tpm":
        regex = (("- ", ""), ("-", ""), ("(", ""), (")", ""), (" ", "_"))
    else:
        regex = ((" ", "_"), ("", ""))  # second added as placeholder so we can use *r

    df = pd.concat([median_matrix, median_across_tissues], axis=1)
    df["all_tissues"] = np.log2(df["all_tissues"] + 0.25)
    for tissue in list(df.columns)[:-1]:
        tissue_rename = tissue.casefold()
        for r in regex:
            tissue_rename = tissue_rename.replace(*r)
        df.rename(columns={f"{tissue}": f"{tissue_rename}"}, inplace=True)

        # convert tpms to log2 (+0.25 to avoid negative infinity)
        df[f"{tissue_rename}"] = np.log2(df[f"{tissue_rename}"] + 0.25)
        df[f"{tissue_rename}_foldchange"] = df["all_tissues"] - df[f"{tissue_rename}"]
    return df


@time_decorator(print_args=False)
def _get_dict_with_target_array(
    split_dict,
    tpmkey,
    prokey,
    tpm_median_and_fold_change_df,
    protein_median_and_fold_change_df,
):
    """_lorem"""
    new = {}
    for gene_tis in split_dict:
        gene = gene_tis.split("_")[0]
        new[gene_tis] = np.array(
            [
                tpm_median_and_fold_change_df[tpmkey].loc[
                    gene
                ],  # median tpm in the tissue
                tpm_median_and_fold_change_df[tpmkey + "_foldchange"].loc[
                    gene
                ],  # fold change
                protein_median_and_fold_change_df[prokey].loc[gene]
                if gene in protein_median_and_fold_change_df.index
                else -1,
                protein_median_and_fold_change_df[prokey + "_foldchange"].loc[gene]
                if gene in protein_median_and_fold_change_df.index
                else -1,
            ]
        )
    return new


@time_decorator(print_args=False)
def _get_target_values_for_tissues(
    tissue_params: dict,
    split: dict,
    tpm_median_and_fold_change_df: pd.DataFrame,
    protein_median_and_fold_change_df: pd.DataFrame,
):
    """_lorem"""
    all_dict = {}
    for tissue in tissue_params:
        all_dict[tissue] = _get_dict_with_target_array(
            split_dict=split,
            tpmkey=tissue_params[tissue][0],
            prokey=tissue_params[tissue][1],
            tpm_median_and_fold_change_df=tpm_median_and_fold_change_df,
            protein_median_and_fold_change_df=protein_median_and_fold_change_df,
        )
    return all_dict


def tissue_targets_for_training(
    split: dict,
    tissue_params: dict,
    expression_median_across_all: str,
    expression_median_matrix: str,
    protein_abundance_matrix: str,
    protein_abundance_medians: str,
    tissues: bool = True,
):
    """ """
    # load tpm median across all tissues and samples
    with open(expression_median_across_all, "rb") as file:
        tpm_all_median = pickle.load(file)

    # load tpm median matrix
    tpm_median_df = parse(expression_median_matrix).data_df

    # load protein abundance median matrix
    protein_abundance_tissue_matrix = _get_protein_adundance_tissue_matrix(
        protein_abundance_medians=protein_abundance_medians,
        graph="tissue",
    )

    # load protein abundance median across all tissues and samples
    protein_abundance_median_across_all = (
        _get_protein_adbunance_median_across_all_tissues_samples(
            protein_abundance_matrix=protein_abundance_matrix
        )
    )

    # create dataframes with target medians and fold change(median in tissue /
    # median across all tissues) for TPM
    tpm_median_and_fold_change_df = _calculate_fold_change_from_medians(
        median_matrix=tpm_median_df,
        median_across_tissues=tpm_all_median,
        type="tpm",
    )

    # create dataframes with target medians and fold change(median in tissue /
    # median across all tissues) for protein abundance
    protein_median_and_fold_change_df = _calculate_fold_change_from_medians(
        median_matrix=protein_abundance_tissue_matrix,
        median_across_tissues=protein_abundance_median_across_all,
        type="protein",
    )

    # parse targets into a dictionary for each gene/tissue combination
    targets = {
        data_split: _get_target_values_for_tissues(
            tissue_params=tissue_params,
            split=split[data_split],
            tpm_median_and_fold_change_df=tpm_median_and_fold_change_df,
            protein_median_and_fold_change_df=protein_median_and_fold_change_df,
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


def main(
    config_dir: str,
    matrix_dir: str,
    gencode_gtf: str,
    test_chrs: List[str],
    val_chrs: List[str],
    expression_median_across_all: str,
    expression_median_matrix: str,
    protein_abundance_matrix: str,
    protein_abundance_medians: str,
    save_dir: str,
) -> None:
    """Pipeline to generate dataset split and target values"""
    # prepare keys for extracting info from dataframes
    keys = {}
    for tissue in TISSUES:
        params = parse_yaml(f"{config_dir}/{tissue}.yaml")
        keys[tissue] = (
            params["resources"]["key_tpm"],
            params["resources"]["key_protein_abundance"],
        )

    # split genes based on chromosome
    chr_split_dictionary = f"{save_dir}/graph_partition_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}.pkl"
    split = _chr_split_train_test_val(
        genes=_genes_from_gff(gencode_gtf),
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        tissue_append=True,
    )

    # save if it doesnt exist
    if not os.path.exists(chr_split_dictionary):
        with open(chr_split_dictionary, "wb") as output:
            pickle.dump(split, output)

    # get targets - 313502 /31283/28231 train/test/validation, total = 373016
    targets = tissue_targets_for_training(
        split=split,
        tissue_params=keys,
        expression_median_across_all=f"{matrix_dir}/{expression_median_across_all}",
        expression_median_matrix=f"{matrix_dir}/{expression_median_matrix}",
        protein_abundance_matrix=f"{matrix_dir}/{protein_abundance_matrix}",
        protein_abundance_medians=f"{matrix_dir}/{protein_abundance_medians}",
    )

    # # filter targets
    # filtered_split = dict.fromkeys(DATA_SPLITS)
    # filtered_genes = set(filter_genes(tissue_params=keys))
    # for data_split in ["train", "test", "validation"]:
    #     print(data_split)
    #     print(len(split[data_split]))
    #     filtered_split[data_split] = [
    #         x for x in split[data_split] if x in filtered_genes
    #     ]
    #     print(data_split)
    #     print(len(filtered_split[data_split]))

    # flatten nested dictionary
    # right now the targets are messed up and every tissue has the every key. FIX!
    flattened_targets = {}
    flattened_targets["train"] = targets["train"]["mammary"]
    flattened_targets["test"] = targets["test"]["mammary"]
    flattened_targets["validation"] = targets["validation"]["mammary"]

    def filtered_genes(tpm_filtered_genes: str) -> List[str]:
        with open(tpm_filtered_genes, newline="") as file:
            return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]

    for idx, tissue in enumerate(keys):
        if idx == 0:
            genes = filtered_genes(f"{tissue}/tpm_filtered_genes.bed")
        else:
            update_genes = filtered_genes(f"{tissue}/tpm_filtered_genes.bed")
            genes += update_genes

    genes = set(genes)

    parsed_targets = {}
    for key in flattened_targets.keys():
        parsed_targets[key] = {
            gene: flattened_targets[key][gene]
            for gene in flattened_targets[key].keys()
            if gene in genes
        }
    # save targets
    with open("graphs/training_targets.pkl", "wb") as output:
        pickle.dump(parsed_targets, output)

    # store targets from trainset to make standardscaler
    medians = [parsed_targets["train"][target][0] for target in parsed_targets["train"]]
    change = [parsed_targets["train"][target][1] for target in parsed_targets["train"]]


if __name__ == "__main__":
    main(
        config_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs",
        matrix_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data",
        gencode_gtf="shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed",
        test_chrs=["chr8", "chr9"],
        val_chrs=["chr7", "chr13"],
        expression_median_across_all="gtex_tpm_median_across_all_tissues.pkl",
        expression_median_matrix="GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct",
        protein_abundance_matrix="protein_relative_abundance_all_gtex.csv",
        protein_abundance_medians="protein_relative_abundance_median_gtex.csv",
        save_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs",
    )

config_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs"
matrix_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data"
gencode_gtf = "shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
test_chrs = ["chr8", "chr9"]
val_chrs = ["chr7", "chr13"]
expression_median_across_all = "gtex_tpm_median_across_all_tissues.pkl"
expression_median_matrix = (
    "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
)
protein_abundance_matrix = "protein_relative_abundance_all_gtex.csv"
protein_abundance_medians = "protein_relative_abundance_median_gtex.csv"
save_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs"


"""In [83]: len(parsed_targets['train'])
Out[83]: 130087

In [84]: len(parsed_targets['test'])
Out[84]: 11420

In [85]: len(parsed_targets['validation'])
Out[85]: 10505"""
