#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [X] Filter genes BEFORE dividing into train/test/val
# - [ ] Fix names of saved targets
#   - [ ] Make sure to save targets at experiment directory
#   - [ ] Fix all the gross directories, and make them part of params
# - [ ] Remove the file check in _tpmmedianacrossalltissues to the produce_training_targets function
# - [ ] Target dict is super redundant, clean it up so it doesn't hold so many
#

"""Get train / test / val splits for nodes in graphs and generate targets for
training the network."""

import argparse
import csv
import os
import pickle
from typing import Dict, List, Tuple, Union

from cmapPy.pandasGEXpress.parse_gct import parse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import genes_from_gff
from utils import parse_yaml
from utils import time_decorator

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


def _get_tissue_keywords(
    tissues: List[str],
    config_dir: str,
) -> Dict[str, Tuple[str, str]]:
    """Get tissue keywords from configuration files.

    Args:
        tissues (List[str]): A list of tissue names.
        config_dir (str): The directory where tissue configuration files are
        stored.

    Returns:
        Dict[str, Tuple[str, str]]: A dictionary mapping tissue names to their
        keywords.
    """
    tissue_keywords = {}

    for tissue in tissues:
        params = parse_yaml(f"{config_dir}/{tissue}.yaml")

        tpm_keyword = params["resources"]["key_tpm"]
        protein_abundance_keyword = params["resources"]["key_protein_abundance"]

        tissue_keywords[tissue] = (tpm_keyword, protein_abundance_keyword)
    return tissue_keywords


def _filtered_genes(
    tpm_filtered_genes: str,
    tissue: str,
) -> List[str]:
    """Return a list of genes from a filtered gtex bedfile"""
    with open(tpm_filtered_genes, newline="") as file:
        return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]


def _get_tpm_filtered_genes(
    tissue_keywords: Dict[str, Tuple[str, str]],
    working_directory: str,
    experiment_name: str,
) -> List[str]:
    """Process tissue targets by filtering them based on tissue keywords and
    genes.

    Args:
        tissue_keywords (Dict[str, Tuple[str, str]): A dictionary mapping tissue
        names to keywords.
        working_directory (str): The directory where the data is located.
        experiment_name (str): The name of the experiment.
        targets (Dict[str, Dict[str, np.ndarray]]): A dictionary of targets for
        different tissues.

    Returns:
        List[str]: A list of filtered genes.
    """
    unique_genes = set()

    # Gather unique genes from all tissues
    for idx, tissue in enumerate(tissue_keywords):
        tpm_filtered_file = (
            f"{working_directory}/{experiment_name}/{tissue}/tpm_filtered_genes.bed"
        )
        tissue_genes = _filtered_genes(
            tpm_filtered_genes=tpm_filtered_file,
            tissue=tissue,
        )

        if idx == 0:
            unique_genes = set(tissue_genes)
        else:
            unique_genes.update(tissue_genes)

    return unique_genes


def _save_partitioning_split(
    save_dir: str,
    test_chrs: List[int],
    val_chrs: List[int],
    split: Dict[str, List[str]],
) -> None:
    """Save the partitioning split to a file based on provided chromosome
    information.

    Args:
        save_dir (str): The directory where the split file will be saved.
        test_chrs (List[int]): A list of test chromosomes.
        val_chrs (List[int]): A list of validation chromosomes.
        split (Dict[str, List[str]]): A dictionary containing the split data.

    Returns:
        None
    """
    chrs = []

    if test_chrs and val_chrs:
        chrs.append(f"test_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}")
    elif test_chrs:
        chrs.append(f"test_{('-').join(test_chrs)}")
    elif val_chrs:
        chrs.append(f"val_{('-').join(val_chrs)}")
    else:
        chrs.append("random_assign")

    chr_split_dictionary = f"{save_dir}/{('').join(chrs)}_training_split.pkl"

    if not os.path.exists(chr_split_dictionary):
        with open(chr_split_dictionary, "wb") as output:
            pickle.dump(split, output)

    # return f"{('').join(chrs)}"


def _tpm_median_across_all_tissues(
    gct_file: str,
    save_path: str,
) -> None:
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT and
    saves it. Because the file is large and requires a lot of memory, we ran
    this separately from the produce_training_targets function and is only run once.

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
    target_genes: List[str],
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
        target_genes (List[str]): List of genes passing tpm filter. tissues
        (List[str], optional): List of tissue names. Defaults to an empty list.
        tissue_append (bool, optional): Whether or not to append tissue name to
        ID. Defaults to True. test_chrs (List[int], optional): List of
        chromosome numbers for test genes. Defaults to an empty list. val_chrs
        (List[int], optional): List of chromosome numbers for validation genes.
        Defaults to an empty list.

    Returns:
        Dict[str, List[str]]: Dictionary of genes split into train, test, and
        validation.
    """
    # get only genes that are in the filtered genes list
    target_genes_tissue_remove = [x.split("_")[0] for x in target_genes]
    target_gene_dict = {gene: genes[gene] for gene in target_genes_tissue_remove}

    # shuffle
    all_genes = list(target_gene_dict.keys())

    if test_chrs and val_chrs:
        test_genes = [gene for gene in all_genes if genes[gene] in test_chrs]
        val_genes = [gene for gene in all_genes if genes[gene] in val_chrs]
        train_genes = [gene for gene in all_genes if gene not in test_genes + val_genes]
    elif test_chrs:
        test_genes = [gene for gene in all_genes if genes[gene] in test_chrs]
        leftover = [gene for gene in all_genes if genes[gene] not in test_chrs]
        train_genes, val_genes = train_test_split(
            leftover, train_size=0.9, shuffle=True
        )
    elif val_chrs:
        val_genes = [gene for gene in all_genes if genes[gene] in val_chrs]
        leftover = [gene for gene in all_genes if genes[gene] not in val_chrs]
        train_genes, test_genes = train_test_split(
            leftover, train_size=0.9, shuffle=True
        )
    else:
        # If no test or val chromosomes are provided, assign randomly
        train_genes, pre_allocate = train_test_split(
            all_genes, train_size=0.8, shuffle=True
        )
        test_genes, val_genes = train_test_split(
            pre_allocate, train_size=0.5, shuffle=True
        )

    if tissue_append:
        return {
            "train": [
                f"{gene}_{tissue}"
                for gene in train_genes
                for tissue in tissues
                if f"{gene}_{tissue}" in target_genes
            ],
            "test": [
                f"{gene}_{tissue}"
                for gene in test_genes
                for tissue in tissues
                if f"{gene}_{tissue}" in target_genes
            ],
            "validation": [
                f"{gene}_{tissue}"
                for gene in val_genes
                for tissue in tissues
                if f"{gene}_{tissue}" in target_genes
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
    """Rename a tissue string for a given data type (TPM or protein).

    Args:
        tissue (str): The original tissue name to be renamed.
        data_type (str): The type of data (e.g., 'tpm' or 'protein').

    Returns:
        str: The renamed and standardized tissue name.
    """
    if data_type == "tpm":
        regex = (("- ", ""), ("-", ""), ("(", ""), (")", ""), (" ", "_"))
    else:
        regex = ((" ", "_"), ("", ""))

    tissue_rename = tissue.casefold()
    for r in regex:
        tissue_rename = tissue_rename.replace(*r)

    return tissue_rename


@time_decorator(print_args=False)
def _calculate_foldchange_from_medians(
    median_matrix: pd.DataFrame,
    median_across_tissues: pd.DataFrame,
    pseudocount: float,
    data_type: str = "tpm",
) -> pd.DataFrame:
    """Calculate fold change between median values and apply the log2
    transformation. Formally, the fold change is calculated by dividing the
    median value in a tissue by the median value across all tissues.

    Args:
        median_matrix (pd.DataFrame): DataFrame containing median values.
        median_across_tissues (pd.DataFrame): DataFrame containing median values
        across tissues.
        pseudocount (float): Pseudocount value added before log2 transformation.
        data_type (str, optional): Type of data, e.g., "tpm". Defaults to "tpm".

    Returns:
        pd.DataFrame: DataFrame with fold change values.
    """
    df = pd.concat([median_matrix, median_across_tissues], axis=1)
    df["all_tissues"] = df["all_tissues"] + pseudocount

    for tissue in median_matrix.columns:
        tissue_rename = _tissue_rename(tissue=tissue, data_type=data_type)
        df.rename(columns={tissue: tissue_rename}, inplace=True)
        df[f"{tissue_rename}"] = df[f"{tissue_rename}"] + pseudocount
        df[f"{tissue_rename}_foldchange"] = df[f"{tissue_rename}"] / df["all_tissues"]

    return np.log2(df.drop(columns=["all_tissues"]))


@time_decorator(print_args=False)
def _difference_from_average_activity_per_tissue(
    average_activity: pd.DataFrame,
    pseudocount: float,
    tissues: List[str],
    tpm_dir: str,
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
                average_remove_tissue = average_activity.copy()
                average_remove_tissue = average_remove_tissue.drop(columns=[tissue])
                average_remove_tissue["average"] = average_remove_tissue.mean(axis=1)

                df = pd.read_table(f"{tpm_dir}/{file}", index_col=0, header=[2])
                tissue_average = df.mean(axis=1)
                difference = tissue_average.subtract(
                    average_remove_tissue["average"]
                ).abs()
                difference.name = f'{file.split(".tpm.txt")[0]}_difference_from_average'
                dfs.append(difference)

    all_tissues = pd.concat(dfs, axis=1)
    return np.log2(all_tissues + pseudocount)


@time_decorator(print_args=False)
def _get_target_values_for_tissues(
    diff_from_average_df: pd.DataFrame,
    protein_median_and_foldchange_df: pd.DataFrame,
    split: Dict[str, List[str]],
    split_dataset: str,
    tissue_keywords: dict,
    tpm_median_and_foldchange_df: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get target values for each tissue.

    Args:
        diff_from_average_df (pd.DataFrame): DataFrame with difference from
        average data.
        protein_median_and_foldchange_df (pd.DataFrame): DataFrame with protein
        median and foldchange data.
        split (Dict[str, List[str]]): Split data.
        split_dataset (str): Dataset for splitting.
        tissue_keywords (Dict[str, Tuple[str, str]]): Tissue keyword mapping.
        tpm_median_and_foldchange_df (pd.DataFrame): DataFrame with TPM median
        and foldchange data.
    """

    def _get_dict_with_target_array(
        tissue: str,
        tpmkey: str,
        prokey: str,
    ) -> Dict[str, np.ndarray]:
        """Helper function to get the sub dictionary"""
        new = {}
        for target in split[split_dataset]:
            gene, tissue_name = target.split("_", 1)
            if tissue_name == tissue:
                new[target] = np.array(
                    [
                        tpm_median_and_foldchange_df.loc[
                            gene, tpmkey
                        ],  # median tpm in the tissue
                        tpm_median_and_foldchange_df.loc[
                            gene, tpmkey + "_foldchange"
                        ],  # fold change
                        diff_from_average_df.loc[
                            gene, tpmkey + "_difference_from_average"
                        ],
                        protein_median_and_foldchange_df.loc[gene, prokey]
                        if gene in protein_median_and_foldchange_df.index
                        else -1,
                        protein_median_and_foldchange_df.loc[
                            gene, prokey + "_foldchange"
                        ]
                        if gene in protein_median_and_foldchange_df.index
                        else -1,
                    ]
                )
        return new

    all_dict = {}
    for tissue, (tpmkey, prokey) in tissue_keywords.items():
        update_dict = _get_dict_with_target_array(
            tissue=tissue,
            tpmkey=tpmkey,
            prokey=prokey,
        )
        all_dict.update(update_dict)

    return all_dict


def _tissue_targets_for_training(
    average_activity: pd.DataFrame,
    expression_median_across_all: pd.DataFrame,
    expression_median_matrix: str,
    protein_abundance_matrix: str,
    protein_abundance_medians: str,
    pseudocount: float,
    tissue_names: List[str],
    tissue_keywords: Dict[str, Tuple[str, str]],
    tpm_dir: str,
    split: Dict[str, List[str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Prepare target values for training by filling in a dictionary with
    appropriate values from dataframes.

    Args:
        average_activity (pd.DataFrame): DataFrame with average activity data.
        expression_median_across_all (str): Path to the file containing TPM
        median across all tissues.
        expression_median_matrix (str): Path to the file containing TPM median
        matrix.
        protein_abundance_matrix (str): Path to the file containing protein
        abundance matrix.
        protein_abundance_medians (str): Path to the file containing protein
        abundance medians.
        tissue_names (List[str]): List of tissue names.
        tissue_keywords (Dict[str, Tuple[str, str]]): Dictionary mapping tissue
        names to TPM and protein abundance keywords.
        tpm_dir (str): Directory containing TPM data.
        split (Dict[str, List[str]]): Split data.

    Returns:
        A dictionary with the ["train", "test", "validation"] keys and target
        arrays for each gene/tissue combination.
    """

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
    # the average expression in all tissues, and is an absolute value. Uses the
    # first name in the keynames tuple, which are gtex names
    diff_from_average_df = _difference_from_average_activity_per_tissue(
        average_activity=average_activity,
        pseudocount=pseudocount,
        tissues=[tissue_names[0] for tissue_names in tissue_keywords.values()],
        tpm_dir=tpm_dir,
    )

    # create dataframes with target medians and fold change(median in tissue /
    # median across all tissues) for TPM
    tpm_median_and_foldchange_df = _calculate_foldchange_from_medians(
        median_matrix=tpm_median_df,
        median_across_tissues=expression_median_across_all,
        pseudocount=pseudocount,
        data_type="tpm",
    )

    # create dataframes with target medians and fold change(median in tissue /
    # median across all tissues) for protein abundance
    protein_median_and_foldchange_df = _calculate_foldchange_from_medians(
        median_matrix=protein_abundance_tissue_matrix,
        median_across_tissues=protein_abundance_median_across_all,
        pseudocount=pseudocount,
        data_type="protein",
    )

    # parse targets into a dictionary for each gene/tissue combination
    targets = {
        dataset: _get_target_values_for_tissues(
            tissue_keywords=tissue_keywords,
            split=split,
            split_dataset=dataset,
            tpm_median_and_foldchange_df=tpm_median_and_foldchange_df,
            diff_from_average_df=diff_from_average_df,
            protein_median_and_foldchange_df=protein_median_and_foldchange_df,
        )
        for dataset in split
    }

    return targets


@time_decorator(print_args=False)
def _scale_targets(
    targets: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Scale targets using StandardScaler and return the scaled targets.

    Args:
        targets (dict): A dictionary of targets to be scaled.

    Returns:
        dict: A dictionary of scaled targets.
    """
    scaled_targets = targets.copy()
    scaler_dict = {
        0: StandardScaler(),
        1: StandardScaler(),
        2: StandardScaler(),
        3: StandardScaler(),
        4: StandardScaler(),
    }

    # Initialize separate scalers for each type of target data
    for i in range(5):
        data = [
            scaled_targets["train"][target][i] for target in scaled_targets["train"]
        ]
        scaler_dict[i].fit(np.array(data).reshape(-1, 1))

    # Scale the targets in all splits
    for split in scaled_targets:
        for target in scaled_targets[split]:
            for i in range(5):
                scaled_targets[split][target][i] = scaler_dict[i].transform(
                    np.array(scaled_targets[split][target][i]).reshape(-1, 1)
                )

    return scaled_targets


def produce_training_targets(
    experiment_params: Dict[str, Union[str, list]],
) -> None:
    """Pipeline to generate dataset split and target values"""
    parser = argparse.ArgumentParser()
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
    tissues = params["tissues"]

    # set up variables specific for this function
    target_params = params["training_targets"]
    average_activity_df = target_params["average_activity_df"]
    config_dir = target_params["config_dir"]
    expression_median_across_all = target_params["expression_median_across_all"]
    expression_median_matrix = target_params["expression_median_matrix"]
    gencode_gtf = target_params["gencode_gtf"]
    matrix_dir = target_params["matrix_dir"]
    protein_abundance_matrix = target_params["protein_abundance_matrix"]
    protein_abundance_medians = target_params["protein_abundance_medians"]
    test_chrs = target_params["test_chrs"]
    val_chrs = target_params["val_chrs"]

    # create directory for experiment specific scalers
    exp_dir = f"{working_directory}/{experiment_name}"
    graph_dir = f"{exp_dir}/graphs"

    # open average activity dataframe
    with open(average_activity_df, "rb") as file:
        average_activity = pickle.load(file)

    # create median tpm file if needed
    _tpm_median_across_all_tissues(
        gct_file=f"{matrix_dir}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct",
        save_path=matrix_dir,
    )

    # load tpm median across all tissues and samples
    with open(f"{matrix_dir}/{expression_median_across_all}", "rb") as file:
        expression_median_across_all = pickle.load(file)

    # prepare tissue_keywords for extracting info from dataframes
    tissue_keywords = _get_tissue_keywords(tissues=tissues, config_dir=config_dir)

    filtered_genes = _get_tpm_filtered_genes(
        tissue_keywords=tissue_keywords,
        working_directory=working_directory,
        experiment_name=experiment_name,
    )

    # split genes based on chromosome
    split = _genes_train_test_val_split(
        genes=genes_from_gff(gencode_gtf),
        target_genes=filtered_genes,
        tissues=tissues,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        tissue_append=True,
    )

    # save partitioning split
    _save_partitioning_split(
        partition_dir=exp_dir,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        split=split,
    )

    # get targets!
    targets = _tissue_targets_for_training(
        average_activity=average_activity,
        expression_median_across_all=expression_median_across_all,
        expression_median_matrix=f"{matrix_dir}/{expression_median_matrix}",
        protein_abundance_matrix=f"{matrix_dir}/{protein_abundance_matrix}",
        protein_abundance_medians=f"{matrix_dir}/{protein_abundance_medians}",
        pseudocount=0.25,
        tissue_names=PROTEIN_TISSUE_NAMES,
        tissue_keywords=tissue_keywords,
        tpm_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/baseline",
        split=split,
    )

    # save targets
    with open(f"{graph_dir}/targets.pkl", "wb") as output:
        pickle.dump(targets, output)

    # scale targets
    scaled_targets = _scale_targets(targets=targets)

    # save scaled targets
    with open(f"{graph_dir}/targets_scaled.pkl", "wb") as output:
        pickle.dump(scaled_targets, output)
