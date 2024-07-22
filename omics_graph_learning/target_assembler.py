#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Class to handle the assembly of target values for the regression task. Values
are taken from a variety of different matrices and are scaled using
StandardScaler. The class can handle both GTEx TPM data (median regression) and
rna-seq data from ENCODE (also TPM, but at the sample level and not median)"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cmapPy.pandasGEXpress.parse_gct import parse  # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # type: ignore

from config_handlers import ExperimentConfig
from config_handlers import TissueConfig
from gene_filter import read_encode_rna_seq_data
from utils import _load_pickle
from utils import time_decorator


class TargetAssembler:
    """Train test splitter for genes.

    Attributes:
        experiment_config (ExperimentConfig): Configuration object for the
        experiment.
        split (Dict[str, List[str]]): Split information for genes.

    Methods
    --------
    assemble_matrix_targets:
        Assembles matrix targets based on provided split information.

    scale_targets:
        Scales targets using StandardScaler and returns the scaled targets.

    Examples:
    --------
    >>> assembler = TargetAssembler(
        experiment_config,
        split
    )

    >>> targets = assembler.assemble_matrix_targets()
    >>> targets = assembler.assemble_rna_targets()

    # To scale targets
    >>> scaled_targets = assembler.scale_targets(targets)
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        split: Dict[str, List[str]],
        pseudocount: float = 0.25,
    ):
        """Instantiate the gene splitter."""
        self.experiment_config = experiment_config
        self.split = split
        self.pseudocount = pseudocount

        self.config_dir = experiment_config.config_dir
        self.expression_median_matrix = experiment_config.expression_median_matrix
        self.log_transform = experiment_config.log_transform
        self.protein_abundance_matrix = experiment_config.protein_abundance_matrix
        self.protein_abundance_medians = experiment_config.protein_abundance_medians
        self.tissues = experiment_config.tissues
        self.average_activity = _load_pickle(experiment_config.average_activity_df)
        self.expression_median_across_all = _load_pickle(
            experiment_config.expression_median_across_all
        )

    def assemble_matrix_targets(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Assemble GTEx matrix targets based on provided split information."""
        tissue_keywords = self._prepare_keywords()
        return self._tissue_targets_for_training(
            average_activity=self.average_activity,
            expression_median_across_all=self.expression_median_across_all,
            expression_median_matrix=self.expression_median_matrix,
            log_transform_type=self.log_transform,
            protein_abundance_matrix=self.protein_abundance_matrix,
            protein_abundance_medians=self.protein_abundance_medians,
            pseudocount=self.pseudocount,
            tissue_keywords=tissue_keywords,
            tpm_dir=self.experiment_config.tpm_dir,
            split=self.split,
        )

    def assemble_rna_targets(
        self, tissue_config: TissueConfig
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Assemble RNA targets based on provided split information."""
        rna_matrix = tissue_config.resources["rna"]
        rna_quantifications = self._get_rna_quantifications(rna_matrix=rna_matrix)
        return {
            partition: {
                f'{gene}_{tissue_config.resources["tissue"]}': np.array(
                    [rna_quantifications[gene.split("_")[0]]]
                )
                for gene in self.split[partition]
            }
            for partition in self.split
        }

    def _get_rna_quantifications(self, rna_matrix: str) -> Dict[str, float]:
        """Returns a dictionary of gene: log transformed + pseudocount TPM
        values for an RNA-seq quantification .tsv from ENCODE"""
        df = read_encode_rna_seq_data(rna_matrix)
        return (
            (df["TPM"] + self.pseudocount)
            .apply(
                lambda x: self._apply_log_transform(
                    pd.DataFrame([x]), transform_type=self.log_transform
                )[0][0]
            )
            .to_dict()
        )

    def scale_targets(
        self, targets: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Call the static method to scale targets using StandardScaler."""
        return self._scale_targets(targets)

    def _prepare_keywords(self) -> Dict[str, Tuple[str, str]]:
        """Prepare keywords for data extraction from dataframes.

        Returns:
            Dict[str, Tuple[str, str]]: Dictionary of tissue keywords.
        """
        matrix_keywords = {}
        for tissue in self.tissues:
            tissue_config = TissueConfig.from_yaml(self.config_dir / f"{tissue}.yaml")
            tpm_keyword = tissue_config.resources["key_tpm"]
            protein_abundance_keyword = tissue_config.resources["key_protein_abundance"]
            matrix_keywords[tissue] = (tpm_keyword, protein_abundance_keyword)
        return matrix_keywords

    def _raw_protein_abundance_matrix(
        self,
        protein_abundance_medians: str,
        tissue_names: List[str],
        graph: str = "tissue",
    ) -> pd.DataFrame:
        """Returns a dataframe containing the protein abundance median for each
        gene specified in the tissue list

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
            rename_pairs = {x: self._tissue_rename(tissue=x) for x in df.columns}
            df.rename(columns=rename_pairs, inplace=True)
        return df.apply(np.exp2).fillna(0)

    @time_decorator(print_args=False)
    def _calculate_foldchange_from_medians(
        self,
        log_transform_type: str,
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
            x: self._tissue_rename(tissue=x, data_type=data_type)
            for x in median_matrix.columns
        }

        df = pd.concat([median_matrix, median_across_tissues], axis=1)
        df.rename(columns=rename_pairs, inplace=True)
        df += pseudocount
        df[[f"{tissue}_foldchange" for tissue in df.columns]] = df[df.columns].div(
            df["all_tissues"], axis=0
        )

        return pd.DataFrame(
            self._apply_log_transform(
                df.drop(columns=["all_tissues"]), transform_type=log_transform_type
            )
        )

    @time_decorator(print_args=False)
    def _difference_from_average_activity_per_tissue(
        self,
        average_activity: pd.DataFrame,
        log_transform_type: str,
        pseudocount: float,
        tissues: List[str],
        config_dir: Path,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Takes the average activity of the genes in one tissue, and calculates
        the difference from the average activity of the genes in all tissues. In
        this case, the all tissue average includes our tissue of interest.

        Args:
            tpm_dir (str): _description_
            tissues (List[str]): _description_
            average_activity (pd.DataFrame): _description_

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: difference and fold change
            against the average
        """

        def calculate_avg_metrics(tissue: str) -> Tuple[pd.Series, pd.Series]:
            """Get fold change and difference from average for a given tissue."""
            tissue_config = TissueConfig.from_yaml(config_dir / f"{tissue}.yaml")
            df = pd.read_table(
                tissue_config.resources["tpm"], index_col=1, header=[2]
            ).drop("id", axis=1)
            tissue_average = df.mean(axis=1)

            difference = abs(tissue_average.subtract(average_activity["average"]))
            difference.name = f"{tissue}_difference_from_average"

            fold_change = tissue_average.divide(average_activity["average"])
            fold_change.name = f"{tissue}_foldchange_from_average"

            return difference, fold_change

        differences, fold_changes = zip(
            *(calculate_avg_metrics(tissue) for tissue in tissues)
        )

        return pd.DataFrame(
            self._apply_log_transform(
                pd.concat(differences, axis=1) + pseudocount,
                transform_type=log_transform_type,
            )
        ), pd.DataFrame(
            self._apply_log_transform(
                pd.concat(fold_changes, axis=1) + pseudocount,
                transform_type=log_transform_type,
            )
        )

    @staticmethod
    def _get_target_values(
        gene: str,
        tpmkey: str,
        prokey: str,
        tpm_median_and_foldchange_df: pd.DataFrame,
        diff_from_average_df: pd.DataFrame,
        foldchange_from_average_df: pd.DataFrame,
        protein_median_and_foldchange_df: pd.DataFrame,
    ) -> np.ndarray:
        """Get target values for a specific gene based on various dataframes.

        Args:
            gene (str): The gene for which target values are calculated.
            tpmkey (str): The key for TPM values in the dataframes.
            prokey (str): The key for protein abundance values in the dataframes.
            tpm_median_and_foldchange_df (pd.DataFrame): DataFrame containing
            TPM median and foldchange values.
            diff_from_average_df (pd.DataFrame): DataFrame containing
            differences from average activity.
            foldchange_from_average_df (pd.DataFrame): DataFrame containing fold
            changes from average activity.
            protein_median_and_foldchange_df (pd.DataFrame): DataFrame
            containing protein abundance median and foldchange values.

        Returns:
            np.ndarray: An array of target values for the gene.
        """
        return np.array(
            [
                tpm_median_and_foldchange_df.loc[gene, tpmkey],
                tpm_median_and_foldchange_df.loc[gene, f"{tpmkey}_foldchange"],
                diff_from_average_df.loc[gene, f"{tpmkey}_difference_from_average"],
                foldchange_from_average_df.loc[
                    gene, f"{tpmkey}_foldchange_from_average"
                ],
                (
                    protein_median_and_foldchange_df.loc[gene, prokey]
                    if gene in protein_median_and_foldchange_df.index
                    else -1
                ),
                (
                    protein_median_and_foldchange_df.loc[gene, f"{prokey}_foldchange"]
                    if gene in protein_median_and_foldchange_df.index
                    else -1
                ),
            ]
        )

    @time_decorator(print_args=False)
    def _get_targets_per_partition(
        self,
        diff_from_average_df: pd.DataFrame,
        foldchange_from_average_df: pd.DataFrame,
        protein_median_and_foldchange_df: pd.DataFrame,
        split: Dict[str, List[str]],
        partition: str,
        tissue_keywords: dict,
        tpm_median_and_foldchange_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Get target values for each tissue based on the provided dataframes and
        split information.

        Args:
            diff_from_average_df (pd.DataFrame): DataFrame containing
            differences from average activity.
            foldchange_from_average_df (pd.DataFrame): DataFrame containing fold
            changes from average activity.
            protein_median_and_foldchange_df (pd.DataFrame): DataFrame
            containing protein abundance median and foldchange values.
            split (Dict[str, List[str]]): Dictionary of split information for
            each tissue.
            partition (str): The partition for which target values are
            calculated.
            tissue_keywords (dict): Dictionary mapping tissue names to keywords.
            tpm_median_and_foldchange_df (pd.DataFrame): DataFrame containing
            TPM median and foldchange values.

        Returns:
            Dict[str, np.ndarray]: A dictionary of target values for each tissue.
        """

        return {
            target: self._get_target_values(
                gene,
                tissue_keywords[tissue][0],
                tissue_keywords[tissue][1],
                tpm_median_and_foldchange_df,
                diff_from_average_df,
                foldchange_from_average_df,
                protein_median_and_foldchange_df,
            )
            for tissue in tissue_keywords
            for target in split[partition]
            if (gene := target.split("_", 1)[0]) and target.split("_", 1)[1] == tissue
        }

    def _tissue_targets_for_training(
        self,
        average_activity: pd.DataFrame,
        expression_median_across_all: pd.DataFrame,
        expression_median_matrix: str,
        log_transform_type: str,
        protein_abundance_matrix: str,
        protein_abundance_medians: str,
        pseudocount: float,
        tissue_keywords: Dict[str, Tuple[str, str]],
        tpm_dir: Path,
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

        # load nececcary data
        tpm_median_df = parse(expression_median_matrix).data_df
        protein_abundance_tissue_matrix = self._raw_protein_abundance_matrix(
            protein_abundance_medians=protein_abundance_medians,
            tissue_names=tissue_names,
            graph="protein",
        )
        protein_abundance_median_across_all = (
            self._protein_abundance_median_across_all_tissues(
                protein_abundance_matrix=protein_abundance_matrix
            )
        )

        # create dataframe with difference from average activity for each tissue
        # formally, this is defined as the average expression in the tissue minus
        # the average expression in all tissues, and is an absolute value. Uses the
        # first name in the keynames tuple, which are gtex names
        diff_from_average_df, foldchange_from_average_df = (
            self._difference_from_average_activity_per_tissue(
                average_activity=average_activity,
                log_transform_type=log_transform_type,
                pseudocount=pseudocount,
                tissues=tissue_names,
                tpm_dir=tpm_dir,
            )
        )

        # create dataframes with target medians and fold change(median in tissue /
        # median across all tissues) for TPM
        tpm_median_and_foldchange_df = self._calculate_foldchange_from_medians(
            log_transform_type=log_transform_type,
            median_matrix=tpm_median_df,
            median_across_tissues=expression_median_across_all,
            pseudocount=pseudocount,
            data_type="tpm",
        )

        # create dataframes with target medians and fold change(median in tissue /
        # median across all tissues) for prn abundance
        protein_median_and_foldchange_df = self._calculate_foldchange_from_medians(
            log_transform_type=log_transform_type,
            median_matrix=protein_abundance_tissue_matrix,
            median_across_tissues=protein_abundance_median_across_all,
            pseudocount=pseudocount,
            data_type="protein",
        )

        return {
            dataset: self._get_targets_per_partition(
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

    @staticmethod
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
        scaled_targets = deepcopy(targets)
        num_target_types = len(next(iter(targets["train"].values())))
        target_scalers = {i: StandardScaler() for i in range(num_target_types)}

        # Fit scalers for each type of target data
        for i in range(num_target_types):
            data = np.stack(
                [
                    scaled_targets["train"][target][i]
                    for target in scaled_targets["train"]
                ]
            )
            target_scalers[i].fit(data.reshape(-1, 1))

        # Scale the targets in all splits
        for split_targets in scaled_targets.values():
            for target, values in split_targets.items():
                scaled_values = []
                for i, scaler in target_scalers.items():
                    scaled_value = scaler.transform(values[i].reshape(-1, 1)).flatten()
                    scaled_values.append(scaled_value)
                split_targets[target] = np.array(scaled_values)

        return scaled_targets

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _apply_log_transform(
        data: pd.DataFrame, transform_type: str = "log2"
    ) -> np.ndarray:
        """Applies the specified log transformation to a DataFrame.

        Args:
            data (pd.DataFrame): The data to apply the transformation to.
            transform_type (str): The type of log transformation: 'log2',
            'log1p', or 'log10'.

        Returns:
            pd.DataFrame: The log-transformed data.
        """
        if transform_type == "log2":
            transformed_data = np.log2(data)
        elif transform_type == "log1p":
            transformed_data = np.log1p(data)
        elif transform_type == "log10":
            transformed_data = np.log10(data)
        else:
            raise ValueError(
                "Invalid log transformation type specified. Choose 'log2', 'log1p', or 'log10'."
            )
        return transformed_data
