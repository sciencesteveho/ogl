#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Fix the protein tissue names...

"""Get train / test / val splits for nodes in graphs and generate targets for
training the network."""

import argparse
from copy import deepcopy
import csv
import os
import pathlib
from typing import Any, Dict, List, Tuple, Union

from cmapPy.pandasGEXpress.parse_gct import parse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import utils


def _tpm_filter_genes_and_prepare_keywords(
    config_dir: str,
    gencode_gtf: str,
    split_path: pathlib.PosixPath,
    tissues: List[str],
    tpm_filter: Union[float, int],
    percent_of_samples_filter: float,
) -> None:
    """Filter genes for all tissues based on TPM values, creating the file
    'tpm_filtered_genes.bed' in each tissue directory.

    Args:
        config_dir (str): _description_
        working_directory (str): _description_
        tissues (List[str]): _description_
        tpm_filter (Union[float, int]): _description_
        percent_of_samples_filter (float): _description_

    Returns:
        None
    """
    matrix_keywords = {}
    for tissue in tissues:
        params = utils.parse_yaml(f"{config_dir}/{tissue}.yaml")

        # set up params
        genes = split_path / f"{tissue}_tpm_filtered_genes.bed"
        tpm_file = f"{params['resources']['tpm']}"
        tpm_keyword = params["resources"]["key_tpm"]
        protein_abundance_keyword = params["resources"]["key_protein_abundance"]

        # add to protein tissue keywords
        matrix_keywords[tissue] = (tpm_keyword, protein_abundance_keyword)

        if not genes.exists():
            # filter genes here!
            utils.filter_genes_by_tpm(
                gencode=gencode_gtf,
                tpm_file=tpm_file,
                tpm_filter=tpm_filter,
                percent_of_samples_filter=percent_of_samples_filter,
            ).saveas(genes)

    return matrix_keywords


def _open_filtered_genes_and_append_tissue(
    tpm_filtered_genes: str,
    tissue: str,
) -> List[str]:
    """Return a list of genes from a filtered gtex bedfile"""
    with open(tpm_filtered_genes, newline="") as file:
        return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]


def _get_tpm_filtered_genes(
    tissue_keywords: Dict[str, Tuple[str, str]],
    split_path: pathlib.PosixPath,
) -> List[str]:
    """Process tissue targets by filtering them based on tissue keywords and
    genes.

    Args:
        tissue_keywords (Dict[str, Tuple[str, str]): A dictionary mapping tissue
        names to keywords.
        working_directory (str): The directory where the data is located.
        experiment_name (str): The name of the experiment.

    Returns:
        List[str]: A list of filtered genes.
    """
    unique_genes = set()

    # Gather unique genes from all tissues
    for tissue in tissue_keywords:
        tpm_filtered_file = split_path / f"{tissue}_tpm_filtered_genes.bed"
        tissue_genes = _open_filtered_genes_and_append_tissue(
            tpm_filtered_genes=tpm_filtered_file,
            tissue=tissue,
        )
        unique_genes.update(tissue_genes)

    return unique_genes


def _tpm_median_across_all_tissues(
    median_across_all_file: pathlib.PosixPath,
    all_matrix_gct: str,
) -> None:
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT and
    saves it. Because the file is large and requires a lot of memory, we ran
    this separately from the produce_training_targets function and is only run once.

    Args:
        median_across_all_file (str): /path/to/median_across_all_file
        all_matrix_gct (str): /path/to/gtex gct file
    """
    try:
        if not median_across_all_file.exists():
            median_series = pd.Series(
                parse(all_matrix_gct).data_df.median(axis=1), name="all_tissues"
            ).to_frame()
            median_series.to_pickle(median_across_all_file, mode="xb")
        else:
            print("File already exists")
    except FileExistsError:
        print("File already exists!")


@utils.time_decorator(print_args=False)
def _genes_train_test_val_split(
    genes: Union[Dict[str, str], List[str]],
    target_genes: List[str],
    tissues: List[str] = None,
    tissue_append: bool = True,
    test_chrs: List[int] = None,
    val_chrs: List[int] = None,
    rna: bool = False,
) -> Dict[str, List[str]]:
    """Creates training, test, and validation splits for genes based on
    chromosome. Adds tissues to each label to differentiate targets between
    graphs. If values are input for test and val chrs, then the genes are
    assigned based on test or val chrs. Otherwise, the genes are assigned at
    random.

    Args:
        genes (Dict[str, str]): Dictionary of all genes in the genome.
        target_genes (List[str]): List of genes passing tpm filter.
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
    # get only genes that are in the filtered genes list
    if rna:
        all_genes = genes
    else:
        target_gene_chrs = {
            gene.split("_")[0]: genes[gene.split("_")[0]]
            for gene in target_genes
            if gene.split("_")[0] in genes
        }
        all_genes = list(target_gene_chrs.keys())
        test_genes, val_genes = [], []
        for gene in all_genes:
            if genes[gene] in test_chrs:
                test_genes.append(gene)
            elif genes[gene] in val_chrs:
                val_genes.append(gene)

    # if no test or val chrs, then 80% of genes are for training. The remaining
    # 20% are split into test and validation. If test or val chrs are provided,
    # then remaining genes are split 90% for training and 10% for the missing
    # set.
    if not test_chrs and not val_chrs:
        train_genes, test_genes, val_genes = np.split(
            np.random.permutation(all_genes),
            [int(0.8 * len(all_genes)), int(0.9 * len(all_genes))],
        )
    else:
        train_genes = list(set(all_genes) - set(test_genes) - set(val_genes))
        if not test_chrs or not val_chrs:
            train_genes, split_genes = train_test_split(
                train_genes, train_size=0.9, shuffle=True
            )
        if not test_chrs:
            test_genes = split_genes
        if not val_chrs:
            val_genes = split_genes

    if tissue_append:
        return {
            "train": _append_tissues(train_genes, tissues, target_genes),
            "test": _append_tissues(test_genes, tissues, target_genes),
            "validation": _append_tissues(val_genes, tissues, target_genes),
        }
    else:
        return {
            "train": list(train_genes),
            "test": list(test_genes),
            "validation": list(val_genes),
        }


def _append_tissues(genes_set, tissues, target_genes):
    return [
        f"{gene}_{tissue}"
        for gene in genes_set
        for tissue in tissues
        if f"{gene}_{tissue}" in target_genes
    ]


def _protein_abundance_median_across_all_tissues(
    protein_abundance_matrix: str,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with median protein abundance for each gene
    across all samples and tissues. Values are log2, so we inverse log them,
    as we transform them ourselves later.

    Args:
        protein_abundance_matrix (str): /path/to/protein_abundance_matrix.csv

    Returns:
        pandas DataFrame with median protein abundance for each gene across
        all samples and tissues
    """
    return (
        pd.read_csv(protein_abundance_matrix, sep=",", index_col="gene.id.full")
        .drop(columns=["gene.id"])
        .apply(np.exp2)
        .fillna(0)
        .median(axis=1)
        .rename("all_tissues")
        .to_frame()
    )


def _raw_protein_abundance_matrix(
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
    usecols = ["gene.id.full"] + tissue_names if graph == "tissue" else None
    df = pd.read_csv(
        protein_abundance_medians,
        sep=",",
        index_col="gene.id.full",
        usecols=usecols,
    )
    if graph != "tissue":
        rename_pairs = {x: _tissue_rename(tissue=x) for x in df.columns}
        df.rename(columns=rename_pairs, inplace=True)
    return df.apply(np.exp2).fillna(0)


def _tissue_rename(
    tissue: str,
    data_type: str = "",
) -> str:
    """Rename a tissue string for a given data type (TPM or protein).

    Args:
        tissue (str): The original tissue name to be renamed.
        data_type (str): The type of data (e.g., 'tpm' or 'protein').

    Returns:
        str: The renamed and standardized tissue name.
    """
    replacements = (
        {"- ": "", "-": "", "(": "", ")": "", " ": "_"}
        if data_type == "tpm"
        else {" ": "_"}
    )
    for old, new in replacements.items():
        tissue = tissue.replace(old, new)
    return tissue.casefold()


@utils.time_decorator(print_args=False)
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
    rename_pairs = {
        x: _tissue_rename(tissue=x, data_type=data_type) for x in median_matrix.columns
    }

    df = pd.concat([median_matrix, median_across_tissues], axis=1)
    df.rename(columns=rename_pairs, inplace=True)

    df += pseudocount
    df[[f"{tissue}_foldchange" for tissue in df.columns]] = df[df.columns].div(
        df["all_tissues"], axis=0
    )

    return np.log2(df.drop(columns=["all_tissues"]))


@utils.time_decorator(print_args=False)
def _difference_from_average_activity_per_tissue(
    average_activity: pd.DataFrame,
    pseudocount: float,
    tissues: List[str],
    tpm_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        tpm_dir (str): _description_
        tissues (List[str]): _description_
        average_activity (pd.DataFrame): _description_

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: difference and fold change against the average
    """
    differences, fold_changes = [], []
    tpm_path = pathlib.Path(tpm_dir)

    for file_path in tpm_path.glob("*.tpm.txt"):
        tissue = file_path.stem.split(".")[0]

        if tissue in tissues:
            average_remove_tissue = average_activity.drop(columns=[tissue])
            average_remove_tissue["average"] = average_remove_tissue.mean(axis=1)

            df = pd.read_table(file_path, index_col=0, header=[2])
            tissue_average = df.mean(axis=1)

            difference = abs(tissue_average.subtract(average_remove_tissue["average"]))
            difference.name = f"{tissue}_difference_from_average"
            differences.append(difference)

            fold_change = tissue_average.divide(average_remove_tissue["average"])
            fold_change.name = f"{tissue}_foldchange_from_average"
            fold_changes.append(fold_change)

    return np.log2(pd.concat(differences, axis=1) + pseudocount), np.log2(
        pd.concat(fold_changes, axis=1) + pseudocount
    )


def _get_targets_from_rna_seq(
    expression_quantifications: Dict[str, float],
    split: Dict[str, List[str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get target values for each tissue."""
    return {
        part: {
            gene: np.array([expression_quantifications[gene.split("_")[0]]])
            for gene in split[part]
        }
        for part in split
    }


@utils.time_decorator(print_args=False)
def _get_targets_per_partition(
    diff_from_average_df: pd.DataFrame,
    foldchange_from_average_df: pd.DataFrame,
    protein_median_and_foldchange_df: pd.DataFrame,
    split: Dict[str, List[str]],
    partition: str,
    tissue_keywords: dict,
    tpm_median_and_foldchange_df: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get target values for each tissue."""

    targets = {}
    for tissue in tissue_keywords:
        for target in split[partition]:
            gene, gene_tissue = target.split("_", 1)
            if gene_tissue == tissue:
                tpmkey = tissue_keywords[tissue][0]
                prokey = tissue_keywords[tissue][1]
                targets[target] = np.array(
                    [
                        tpm_median_and_foldchange_df.loc[gene, tpmkey],
                        tpm_median_and_foldchange_df.loc[gene, f"{tpmkey}_foldchange"],
                        diff_from_average_df.loc[
                            gene, f"{tpmkey}_difference_from_average"
                        ],
                        foldchange_from_average_df.loc[
                            gene, f"{tpmkey}_foldchange_from_average"
                        ],
                        (
                            protein_median_and_foldchange_df.loc[gene, prokey]
                            if gene in protein_median_and_foldchange_df.index
                            else -1
                        ),
                        (
                            protein_median_and_foldchange_df.loc[
                                gene, f"{prokey}_foldchange"
                            ]
                            if gene in protein_median_and_foldchange_df.index
                            else -1
                        ),
                    ]
                )
    return targets


def _tissue_targets_for_training(
    average_activity: pd.DataFrame,
    expression_median_across_all: pd.DataFrame,
    expression_median_matrix: str,
    protein_abundance_matrix: str,
    protein_abundance_medians: str,
    pseudocount: float,
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
    tissue_names = [tissue_names[0] for tissue_names in tissue_keywords.values()]

    # load tpm median matrix
    tpm_median_df = parse(expression_median_matrix).data_df

    # load protein abundance median matrix
    protein_abundance_tissue_matrix = _raw_protein_abundance_matrix(
        protein_abundance_medians=protein_abundance_medians,
        tissue_names=tissue_names,
        graph="protein",
    )

    # load protein abundance median across all tissues and samples
    protein_abundance_median_across_all = _protein_abundance_median_across_all_tissues(
        protein_abundance_matrix=protein_abundance_matrix
    )

    # create dataframe with difference from average activity for each tissue
    # formally, this is defined as the average expression in the tissue minus
    # the average expression in all tissues, and is an absolute value. Uses the
    # first name in the keynames tuple, which are gtex names
    diff_from_average_df, foldchange_from_average_df = (
        _difference_from_average_activity_per_tissue(
            average_activity=average_activity,
            pseudocount=pseudocount,
            tissues=tissue_names,
            tpm_dir=tpm_dir,
        )
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
    # median across all tissues) for prn abundance
    protein_median_and_foldchange_df = _calculate_foldchange_from_medians(
        median_matrix=protein_abundance_tissue_matrix,
        median_across_tissues=protein_abundance_median_across_all,
        pseudocount=pseudocount,
        data_type="protein",
    )

    return {
        dataset: _get_targets_per_partition(
            tissue_keywords=tissue_keywords,
            split=split,
            partition=dataset,
            tpm_median_and_foldchange_df=tpm_median_and_foldchange_df,
            diff_from_average_df=diff_from_average_df,
            foldchange_from_average_df=foldchange_from_average_df,
            protein_median_and_foldchange_df=protein_median_and_foldchange_df,
        )
        for dataset in split
    }


@utils.time_decorator(print_args=False)
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
    scaled_targets = deepcopy(targets)
    num_target_types = len(next(iter(targets["train"].values())))
    target_scalers = {i: StandardScaler() for i in range(num_target_types)}

    # Fit scalers for each type of target data
    for i in range(num_target_types):
        data = np.stack(
            [scaled_targets["train"][target][i] for target in scaled_targets["train"]]
        )
        target_scalers[i].fit(data.reshape(-1, 1))

    # Scale the targets in all splits
    for split_targets in scaled_targets.values():
        for _, values in split_targets.items():
            for i, scaler in target_scalers.items():
                values[i] = scaler.transform(values[i].reshape(-1, 1)).flatten()

    return scaled_targets


def _unpack_params(params: Dict[str, Union[str, List[str], Dict[str, str]]]):
    """Unpack params from yaml config"""
    experiment_name = params["experiment_name"]
    working_directory = params["working_directory"]
    tissues = params["tissues"]
    target_params = params["training_targets"]
    average_activity_df = target_params["average_activity_df"]
    config_dir = target_params["config_dir"]
    expression_median_across_all_df = target_params["expression_median_across_all"]
    expression_median_matrix = target_params["expression_median_matrix"]
    all_matrix_gct = target_params["expression_all_matrix"]
    gencode_gtf = target_params["gencode_gtf"]
    matrix_path = target_params["matrix_dir"]
    protein_abundance_matrix = target_params["protein_abundance_matrix"]
    protein_abundance_medians = target_params["protein_abundance_medians"]
    test_chrs = target_params["test_chrs"]
    val_chrs = target_params["val_chrs"]
    tpm_dir = target_params["tpm_dir"]
    return (
        experiment_name,
        working_directory,
        tissues,
        average_activity_df,
        config_dir,
        expression_median_across_all_df,
        expression_median_matrix,
        all_matrix_gct,
        gencode_gtf,
        pathlib.Path(matrix_path),
        protein_abundance_matrix,
        protein_abundance_medians,
        test_chrs,
        val_chrs,
        tpm_dir,
    )


def _prepare_split_directories(
    working_directory: str, experiment_name: str, split_name: str
) -> Tuple[pathlib.PosixPath, pathlib.PosixPath]:
    """Prep splite-specific directories for saving data."""
    working_path = pathlib.Path(working_directory)
    graph_dir = working_path / experiment_name / "graphs"
    split_path = graph_dir / split_name
    utils.dir_check_make(split_path)
    return graph_dir, split_path


def _load_dataframes(
    matrix_path: str, average_activity_df: str, expression_median_across_all_df: str
):
    average_activity = utils._load_pickle(average_activity_df)
    expression_median_across_all = utils._load_pickle(
        matrix_path / expression_median_across_all_df
    )
    return average_activity, expression_median_across_all


def _save_splits(
    split: Dict[str, List[str]],
    # barebones_split: Dict[str, List[str]],
    split_path: str,
) -> None:
    chr_split_dictionary = split_path / "training_targets_split.pkl"
    if not os.path.exists(chr_split_dictionary):
        utils._save_pickle(split, chr_split_dictionary)
    # utils._save_pickle(barebones_split, split_path / "training_split.pkl")


def _save_targets(
    targets: Dict[str, Dict[str, np.ndarray]],
    split_path: str,
    scaled_targets: Dict[str, Dict[str, np.ndarray]] = None,
) -> None:
    utils._save_pickle(targets, split_path / "training_targets.pkl")
    if scaled_targets:
        utils._save_pickle(scaled_targets, split_path / "training_targets_scaled.pkl")


def prepare_gnn_training_split_and_targets(args: Any, params: Dict[str, Any]) -> None:
    """Prepares the GNN training split and targets.

    Args:
        args: The arguments for the function.
        params: The parameters for the function.
    """
    (
        experiment_name,
        working_directory,
        tissues,
        average_activity_df,
        config_dir,
        expression_median_across_all_df,
        expression_median_matrix,
        all_matrix_gct,
        gencode_gtf,
        matrix_path,
        protein_abundance_matrix,
        protein_abundance_medians,
        test_chrs,
        val_chrs,
        tpm_dir,
    ) = _unpack_params(params)

    # check if the matrix with a median across all samples exists
    _tpm_median_across_all_tissues(
        median_across_all_file=matrix_path / expression_median_across_all_df,
        all_matrix_gct=matrix_path / all_matrix_gct,
    )

    _, split_path = _prepare_split_directories(
        working_directory=working_directory,
        experiment_name=experiment_name,
        split_name=args.split_name,
    )

    if args.rna_seq:
        _extracted_from_prepare_gnn_training_split_and_targets_39(
            tissues, config_dir, gencode_gtf, split_path
        )
    else:
        tissue_keywords = _tpm_filter_genes_and_prepare_keywords(
            config_dir=config_dir,
            gencode_gtf=gencode_gtf,
            tissues=tissues,
            tpm_filter=args.tpm_filter,
            split_path=split_path,
            percent_of_samples_filter=args.percent_of_samples_filter,
        )

        average_activity, expression_median_across_all = _load_dataframes(
            matrix_path=matrix_path,
            average_activity_df=average_activity_df,
            expression_median_across_all_df=expression_median_across_all_df,
        )

        filtered_genes = _get_tpm_filtered_genes(
            tissue_keywords=tissue_keywords, split_path=split_path
        )

        split = _genes_train_test_val_split(
            genes=utils.genes_from_gff(gencode_gtf),
            target_genes=filtered_genes,
            tissues=tissues,
            test_chrs=test_chrs,
            val_chrs=val_chrs,
            tissue_append=True,
        )

        _save_splits(split=split, split_path=split_path)

        targets = _tissue_targets_for_training(
            average_activity=average_activity,
            expression_median_across_all=expression_median_across_all,
            expression_median_matrix=matrix_path / expression_median_matrix,
            protein_abundance_matrix=matrix_path / protein_abundance_matrix,
            protein_abundance_medians=matrix_path / protein_abundance_medians,
            pseudocount=0.25,
            tissue_keywords=tissue_keywords,
            tpm_dir=tpm_dir,
            split=split,
        )
        scaled_targets = _scale_targets(targets)
        _save_targets(
            targets=targets, scaled_targets=scaled_targets, split_path=split_path
        )


def _extracted_from_prepare_gnn_training_split_and_targets_39(
    tissues, config_dir, gencode_gtf, split_path
):
    for tissue in tissues:
        params = utils.parse_yaml(f"{config_dir}/{tissue}.yaml")
        rna = params["resources"]["rna"]

    with open(rna, "r") as f:
        rna_quantifications = {
            line[0]: np.log2(float(line[1])) for line in csv.reader(f, delimiter="\t")
        }

    split = _genes_train_test_val_split(
        genes=list(rna_quantifications.keys()),
        target_genes=rna_quantifications.keys(),
        tissues=tissues,
        tissue_append=True,
        rna=True,
    )

    targets = _get_targets_from_rna_seq(
        expression_quantifications=rna_quantifications, split=split
    )

    _save_splits(split=split, split_path=split_path)
    _save_targets(targets=targets, split_path=split_path)


def main() -> None:
    """Main function for dataset_split.py. Parses command line arguments and
    calls training split fxn.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    parser.add_argument("--tpm_filter", type=float, help="TPM filter for genes")
    parser.add_argument(
        "--percent_of_samples_filter",
        type=float,
        help="Percent of samples filter for genes (e.g. 0.20)",
    )
    parser.add_argument("--split_name", type=str, help="Name of the split")
    parser.add_argument(
        "--rna_seq",
        action="store_true",
        help="Whether to use RNA-seq data for targets",
    )
    args = parser.parse_args()
    params = utils.parse_yaml(args.experiment_config)

    prepare_gnn_training_split_and_targets(args=args, params=params)


if __name__ == "__main__":
    main()
