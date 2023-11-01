#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Remove the file check in _tpmmedianacrossalltissues to the main function
# - [ ] Target dict is super redundant, clean it up so it doesn't hold so many
#

"""Get train / test / val splits for nodes in graphs and generate targets for
training the network.
"""

import argparse
import csv
import os
import pickle
from random import shuffle
from typing import Dict, List, Tuple

from cmapPy.pandasGEXpress.parse_gct import parse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import genes_from_gff
from utils import parse_yaml
from utils import time_decorator

DATA_SPLITS = ["train", "test", "validation"]

PROTEIN_TISSUE_NAMES = [
    "Artery Aorta",
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


def _tpm_median_across_all_tissues(
    gct_file: str,
    save_path: str,
) -> None:
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT and
    saves it. Because the file is large and requires a lot of memory, we ran
    this separately from the main function and is only run once.

    Args:
        gct_file (str): /path/to/gtex gct file
    """
    savefile = f"{save_path}/gtex_tpm_median_across_all_tissues.pkl"
    if os.path.exists(savefile):
        pass
    else:
        df = parse(gct_file).data_df
        median_series = pd.Series(df.median(axis=1), name="all_tissues").to_frame()
        median_series.to_pickle(savefile)


@time_decorator(print_args=False)
def _genes_train_test_val_split(
    genes: Dict[str, str],
    tissues: List[str] = [],
    tissue_append: bool = True,
    test_chrs: List[int] = [],
    val_chrs: List[int] = [],
) -> Dict[str, List[str]]:
    """Creates training, test, and validation splits for genes based on
    chromosome. Adds tissues to each label to differentiate targets between
    graphs. If values are input for test and val chrs, then the genes are
    assigned based on test or val chrs. Otherwise, the genes are assigned at
    random.

    Args:
        genes (Dict[str, str]): Dictionary of all genes in the genome.
        tissues (List[str], optional): List of tissue names. Defaults to an
        empty list.
        tissue_append (bool, optional): Whether or not to append tissue name to
        ID. Defaults to True.
        test_chrs (List[int], optional): List of chromosome numbers for test
        genes. Defaults to an empty list.
        val_chrs (List[int], optional): List of chromosome numbers for
        validation genes. Defaults to an empty list.

    Returns:
        Dict[str, List[str]]: Dictionary of genes split into train, test, and
        validation.
    """
    all_genes = list(genes.keys())
    num_genes = len(all_genes)

    if test_chrs and val_chrs:
        test_genes = [gene for gene in all_genes if genes[gene] in test_chrs]
        val_genes = [gene for gene in all_genes if genes[gene] in val_chrs]
    elif test_chrs:
        test_genes = [gene for gene in all_genes if genes[gene] in test_chrs]
        val_genes = []
    elif val_chrs:
        val_genes = [gene for gene in all_genes if genes[gene] in val_chrs]
        test_genes = []
    else:
        # If no test or val chromosomes are provided, assign randomly
        shuffle(all_genes)
        test_genes = all_genes[: num_genes // 10]
        val_genes = all_genes[num_genes // 10 : 2 * (num_genes // 10)]

    train_genes = [gene for gene in all_genes if gene not in test_genes + val_genes]

    if tissue_append:
        return {
            "train": [f"{gene}_{tissue}" for gene in train_genes for tissue in tissues],
            "test": [f"{gene}_{tissue}" for gene in test_genes for tissue in tissues],
            "validation": [
                f"{gene}_{tissue}" for gene in val_genes for tissue in tissues
            ],
        }
    else:
        return {
            "train": train_genes,
            "test": test_genes,
            "validation": val_genes,
        }


def _protein_adbunance_median_across_all_tissues(
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

    df = df.apply(np.exp2).fillna(
        0
    )  # relative abundances are log2, so we take the inverse using exponential
    return pd.Series(df.median(axis=1), name="all_tissues").to_frame()


def _get_protein_adundance_tissue_matrix(
    protein_abundance_medians: str,
    tissue_names: List[str],
    graph: str = "tissue",
) -> pd.DataFrame:
    """Returns a dataframe containing the protein abundance median for each gene
    specified in the tissue list

    Args:
        protein_abundance_medians (str): _description_
        graph (str, optional): _description_. Defaults to "tissue".

    Raises:
        ValueError if graph data_type is not specified
    """
    if graph == "tissue":
        df = pd.read_csv(
            protein_abundance_medians,
            sep=",",
            index_col="gene.id.full",
            usecols=["gene.id.full"] + tissue_names,
        )
    elif graph == "universal":
        df = pd.read_csv(
            protein_abundance_medians,
            sep=",",
            index_col="gene.id.full",
        )
    if graph not in ("tissue", "universal"):
        raise ValueError("Graph data_type must be either 'tissues' or 'universal'")

    return df.apply(np.exp2).fillna(
        0
    )  # relative abundances are log2, so we take the inverse using exponential


def _tissue_rename(
    tissue: str,
    data_type: str,
) -> str:
    """
    Args:
        tissue (str)
        data_type (str)

    Returns:
        str: casefolded and renamed tissue
    """
    if data_type == "tpm":
        regex = (("- ", ""), ("-", ""), ("(", ""), (")", ""), (" ", "_"))
    else:
        regex = ((" ", "_"), ("", ""))

    tissue_rename = tissue.casefold()
    for r in regex:
        tissue_rename = tissue_rename.replace(*r)
    return tissue_rename


def _add_tpm_pseudocount_and_log2_transform(
    df: pd.DataFrame,
    pseudocount: float,
) -> pd.DataFrame:
    return np.log2(df + pseudocount)


@time_decorator(print_args=False)
def _calculate_fold_change_from_medians(
    median_matrix: pd.DataFrame,
    median_across_tissues: pd.DataFrame,
    psuedocount: float,
    data_type: str = "tpm",
) -> pd.DataFrame:
    """_summary_

    Args:
        median_matrix (pd.DataFrame): _description_
        median_across_tissues (pd.DataFrame): _description_
        data_type (str, optional): _description_. Defaults to "tpm".

    Returns:
        pd.DataFrame: _description_
    """
    df = pd.concat([median_matrix, median_across_tissues], axis=1)
    df["all_tissues"] = _add_tpm_pseudocount_and_log2_transform(
        df=df["all_tissues"],
        pseudocount=psuedocount,
    )
    for tissue in df.columns[:-1]:
        tissue_rename = _tissue_rename(tissue=tissue, data_type=data_type)
        df.rename(columns={tissue: tissue_rename}, inplace=True)
        df[tissue_rename] = _add_tpm_pseudocount_and_log2_transform(
            df=df[tissue_rename],
            pseudocount=psuedocount,
        )  # convert tpms to log2 (+0.25 to avoid negative infinity)
        df[f"{tissue_rename}_foldchange"] = df["all_tissues"] - df[f"{tissue_rename}"]
    return df


@time_decorator(print_args=False)
def _difference_from_average_activity_per_tissue(
    tpm_dir: str,
    tissues: List[str],
    average_activity: pd.DataFrame,
) -> pd.DataFrame:
    """_summary_

    Args:
        tpm_dir (str): _description_
        tissues (List[str]): _description_
        average_activity (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # get average activity of all other tissues, but not the tissue of interest
    dfs = []

    for file in os.listdir(tpm_dir):
        if file.endswith(".tpm.txt"):
            tissue = file.split(".tpm.txt")[0]

            if any(tissue in keywords for keywords in tissues):
                average_remove_tissue = average_activity.drop(columns=[tissue])
                average_remove_tissue["average"] = average_remove_tissue.mean(axis=1)

                df = pd.read_table(f"{tpm_dir}/{file}", index_col=0, header=[2])
                tissue_average = df.mean(axis=1)
                difference = tissue_average.subtract(
                    average_remove_tissue["average"]
                ).abs()
                difference.name = f'{file.split(".tpm.txt")[0]}_difference_from_average'
                dfs.append(difference)

    return pd.concat(dfs, axis=1)


@time_decorator(print_args=False)
def _get_target_values_for_tissues(
    tissue_params: dict,
    split: dict,
    tpm_median_and_fold_change_df: pd.DataFrame,
    diff_from_average_df: pd.DataFrame,
    protein_median_and_fold_change_df: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """_summary_

    Args:
        tissue_params (dict): _description_
        split (dict): _description_
        tpm_median_and_fold_change_df (pd.DataFrame): _description_
        diff_from_average_df (pd.DataFrame): _description_
        protein_median_and_fold_change_df (pd.DataFrame): _description_
    """

    def _get_dict_with_target_array(
        tpmkey: str,
        prokey: str,
    ) -> Dict[str, np.ndarray]:
        """Helper function to get the sub dictionary"""
        new = {}
        for gene_tis in split:
            gene = gene_tis.split("_")[0]
            new[gene_tis] = np.array(
                [
                    tpm_median_and_fold_change_df[tpmkey].loc[
                        gene
                    ],  # median tpm in the tissue
                    tpm_median_and_fold_change_df[tpmkey + "_foldchange"].loc[
                        gene
                    ],  # fold change
                    diff_from_average_df[tpmkey + "_difference_from_average"].loc[gene],
                    protein_median_and_fold_change_df[prokey].loc[gene]
                    if gene in protein_median_and_fold_change_df.index
                    else -1,
                    protein_median_and_fold_change_df[prokey + "_foldchange"].loc[gene]
                    if gene in protein_median_and_fold_change_df.index
                    else -1,
                ]
            )
        return new

    all_dict = {}
    for tissue in tissue_params:
        all_dict[tissue] = _get_dict_with_target_array(
            tpmkey=tissue_params[tissue][0],
            prokey=tissue_params[tissue][1],
        )
    return all_dict


def tissue_targets_for_training(
    average_activity: pd.DataFrame,
    expression_median_across_all: str,
    expression_median_matrix: str,
    protein_abundance_matrix: str,
    protein_abundance_medians: str,
    tissue_names: List[str],
    tissue_params: Dict[str, Tuple[str, str]],
    tpm_dir: str,
    split: Dict[str, List[str]],
):
    """_summary_

    Args:
        split (Dict[str, List[str]]): _description_
        tissue_params (_type_): _description_
        expression_median_matrix (str): _description_
        protein_abundance_matrix (str): _description_
        protein_abundance_medians (str): _description_
        tissue_names (List[str]): _description_
        tissues (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # load tpm median across all tissues and samples
    with open(expression_median_across_all, "rb") as file:
        tpm_all_median = pickle.load(file)

    # load tpm median matrix
    tpm_median_df = parse(expression_median_matrix).data_df

    # load protein abundance median matrix
    protein_abundance_tissue_matrix = _get_protein_adundance_tissue_matrix(
        protein_abundance_medians=protein_abundance_medians,
        tissue_names=tissue_names,
        graph="tissue",
    )

    # load protein abundance median across all tissues and samples
    protein_abundance_median_across_all = _protein_adbunance_median_across_all_tissues(
        protein_abundance_matrix=protein_abundance_matrix
    )

    # create dataframe with difference from average activity for each tissue
    # formally, this is defined as the average expression in the tissue minus
    # the average expression in all tissues, and is an absolute value
    diff_from_average_df = _difference_from_average_activity_per_tissue(
        tpm_dir=tpm_dir,
        tissues=tissue_params,
        average_activity=average_activity,
    )

    # create dataframes with target medians and fold change(median in tissue /
    # median across all tissues) for TPM
    tpm_median_and_fold_change_df = _calculate_fold_change_from_medians(
        median_matrix=tpm_median_df,
        median_across_tissues=tpm_all_median,
        psuedocount=0.25,
        data_type="tpm",
    )

    # create dataframes with target medians and fold change(median in tissue /
    # median across all tissues) for protein abundance
    protein_median_and_fold_change_df = _calculate_fold_change_from_medians(
        median_matrix=protein_abundance_tissue_matrix,
        median_across_tissues=protein_abundance_median_across_all,
        psuedocount=0.25,
        data_type="protein",
    )

    # parse targets into a dictionary for each gene/tissue combination
    targets = {
        data_split: _get_target_values_for_tissues(
            tissue_params=tissue_params,
            split=split[data_split],
            tpm_median_and_fold_change_df=tpm_median_and_fold_change_df,
            diff_from_average_df=diff_from_average_df,
            protein_median_and_fold_change_df=protein_median_and_fold_change_df,
        )
        for data_split in DATA_SPLITS
    }

    return targets


def main(
    average_activity_df: str,
    config_dir: str,
    matrix_dir: str,
    gencode_gtf: str,
    expression_median_across_all: str,
    expression_median_matrix: str,
    protein_abundance_matrix: str,
    protein_abundance_medians: str,
) -> None:
    """Pipeline to generate dataset split and target values"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        data_type=str,
        help="Path to .yaml file with experimental conditions",
    )
    # args = parser.parse_args()
    # params = parse_yaml(args.experiment_config)
    params = parse_yaml(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs/ablation_experiments/regulatory_only_deeploop_only_random_split_mediantpm.yaml"
    )

    # set up variables for params to improve readability
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]
    test_chrs = params["test_chrs"]
    val_chrs = params["val_chrs"]
    tissues = params["tissues"]

    # create directory for experiment specific scalers
    exp_dir = f"{working_directory}/{experiment_name}"
    graph_dir = f"{exp_dir}/graphs"
    partition_dir = (
        "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing"
    )

    # open average activity dataframe
    with open(average_activity_df, "rb") as file:
        average_activity = pickle.load(file)

    # create median tpm file if needed
    _tpm_median_across_all_tissues(
        gct_file=f"{matrix_dir}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct",
        save_path=matrix_dir,
    )

    # prepare keys for extracting info from dataframes
    keys = {}
    for tissue in tissues:
        params = parse_yaml(f"{config_dir}/{tissue}.yaml")
        keys[tissue] = (
            params["resources"]["key_tpm"],
            params["resources"]["key_protein_abundance"],
        )

    # split genes based on chromosome
    split = _genes_train_test_val_split(
        genes=genes_from_gff(gencode_gtf),
        tissues=tissues,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        tissue_append=True,
    )

    # save partitioning split
    if test_chrs and val_chrs:
        chr_split_dictionary = f"{partition_dir}/graph_partition_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}.pkl"
    elif test_chrs and not val_chrs:
        chr_split_dictionary = (
            f"{partition_dir}/graph_partition_test_{('-').join(test_chrs)}.pkl"
        )
    elif val_chrs and not test_chrs:
        chr_split_dictionary = (
            f"{partition_dir}/graph_partition_val_{('-').join(val_chrs)}.pkl"
        )
    else:
        chr_split_dictionary = f"{partition_dir}/graph_partition_random_assign.pkl"
    if not os.path.exists(chr_split_dictionary):
        with open(chr_split_dictionary, "wb") as output:
            pickle.dump(split, output)

    # get targets!
    targets = tissue_targets_for_training(
        average_activity=average_activity,
        expression_median_across_all=f"{matrix_dir}/{expression_median_across_all}",
        expression_median_matrix=f"{matrix_dir}/{expression_median_matrix}",
        protein_abundance_matrix=f"{matrix_dir}/{protein_abundance_matrix}",
        protein_abundance_medians=f"{matrix_dir}/{protein_abundance_medians}",
        tissue_names=PROTEIN_TISSUE_NAMES,
        tissue_params=keys,
        tpm_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/baseline",
        split=split,
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
        tpm_filtered_file = (
            f"{working_directory}/{experiment_name}/{tissue}/tpm_filtered_genes.bed"
        )
        if idx == 0:
            genes = filtered_genes(tpm_filtered_genes=tpm_filtered_file)
        else:
            update_genes = filtered_genes(tpm_filtered_genes=tpm_filtered_file)
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
    with open(f"{graph_dir}/training_targets.pkl", "wb") as output:
        pickle.dump(parsed_targets, output)

    # # scale targets
    # with open(f"{graph_dir}/training_targets.pkl", "rb") as output:
    #     parsed_targets = pickle.load(output)

    # store targets from trainset to make standardscaler
    medians = [parsed_targets["train"][target][0] for target in parsed_targets["train"]]
    change = [parsed_targets["train"][target][1] for target in parsed_targets["train"]]

    scaler_medians, scaler_change = StandardScaler(), StandardScaler()
    scaler_medians.fit(np.array(medians).reshape(-1, 1))
    scaler_change.fit(np.array(change).reshape(-1, 1))

    # scale targets
    for split in parsed_targets:
        for target in parsed_targets[split]:
            parsed_targets[split][target][0] = scaler_medians.transform(
                np.array(parsed_targets[split][target][0]).reshape(-1, 1)
            )
            parsed_targets[split][target][1] = scaler_change.transform(
                np.array(parsed_targets[split][target][1]).reshape(-1, 1)
            )

    for split in parsed_targets:
        for target in parsed_targets[split]:
            parsed_targets[split][target] = parsed_targets[split][target][0:2]

    # save targets
    with open(f"{graph_dir}/training_targets_scaled.pkl", "wb") as output:
        pickle.dump(parsed_targets, output)

    # # get target subsets
    # with open("training_targets_scaled.pkl", "rb") as f:
    #     targets = pickle.load(f)

    # for key in targets:
    #     for gene in targets[key]:
    #         targets[key][gene] = targets[key][gene][0]

    # with open("training_targets_onlyexp_scaled.pkl", "wb") as f:
    #     pickle.dump(targets, f)


if __name__ == "__main__":
    main(
        average_activity_df="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/average_activity_all_tissues_df.pkl",
        config_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs",
        matrix_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data",
        gencode_gtf="shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed",
        expression_median_across_all="gtex_tpm_median_across_all_tissues.pkl",
        expression_median_matrix="GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct",
        protein_abundance_matrix="protein_relative_abundance_all_gtex.csv",
        protein_abundance_medians="protein_relative_abundance_median_gtex.csv",
    )


"""
average_activity_df="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/average_activity_all_tissues_df.pkl"
config_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs"
matrix_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data"
gencode_gtf="shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
expression_median_across_all="gtex_tpm_median_across_all_tissues.pkl"
expression_median_matrix="GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
protein_abundance_matrix="protein_relative_abundance_all_gtex.csv"
protein_abundance_medians="protein_relative_abundance_median_gtex.csv"


split=split
tissue_params=keys
expression_median_across_all=f"{matrix_dir}/{expression_median_across_all}"
expression_median_matrix=f"{matrix_dir}/{expression_median_matrix}"
protein_abundance_matrix=f"{matrix_dir}/{protein_abundance_matrix}"
protein_abundance_medians=f"{matrix_dir}/{protein_abundance_medians}"
tissue_names=PROTEIN_TISSUE_NAMES
"""
