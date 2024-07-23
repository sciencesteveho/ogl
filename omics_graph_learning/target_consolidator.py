#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Get train / test / val splits for nodes in graphs and generate targets for
training the network."""


import argparse
from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np

from config_handlers import ExperimentConfig
from config_handlers import TissueConfig
from data_splitter import GeneTrainTestSplitter
from gene_filter import TPMFilter
from target_assembler import TargetAssembler
from utils import _save_pickle
from utils import dir_check_make


class TrainingTargetConsolidator:
    """Object that handles the assembly of training targets for the network per
    sample.

    Attributes:
        experiment_config (ExperimentConfig): Experiment configuration object.
        tissue_config (TissueConfig): Tissue configuration object.
        tpm_filter (float): TPM filter for filtering genes.
        percent_of_samples_filter (float): Percent of samples filter for filtering genes.
        filter_mode (str): Mode for filtering genes.
        split_name (str): Name of the split.
        target (str): Target type for the network.

    Methods
    ----------
    consolidate_training_targets(self) -> List[str]:
        Get training targets for the network.

    Examples:
    ----------
    >>> consolidator = TrainingTargetConsolidator(
            experiment_config=experiment_config,
            tissue_config=tissue_config,
            tpm_filter=tpm_filter,
            percent_of_samples_filter=percent_of_samples_filter,
            filter_mode=filter_mode,
            split_name=split_name,
            target=target,
        )

    >>> targets = consolidator.consolidate_training_targets()
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        tissue_config: TissueConfig,
        tpm_filter: float,
        percent_of_samples_filter: float,
        filter_mode: str,
        split_name: str,
        target: str,
    ):
        """Initialize the assembler."""
        self.experiment_config = experiment_config
        self.tissue_config = tissue_config
        self.tpm_filter = tpm_filter
        self.percent_of_samples_filter = percent_of_samples_filter
        self.filter_mode = filter_mode
        self.split_name = split_name
        self.target = target

        self.tissue = tissue_config.resources["tissue"]
        self.sample_config_dir = experiment_config.sample_config_dir
        self.graph_dir = experiment_config.graph_dir
        self.tissue_dir = self.experiment_config.working_directory / self.tissue
        self.local_dir = self.tissue_dir / "local"
        self.split_path = self._prepare_split_directories()

    def _prepare_split_directories(self) -> Path:
        """Prep split-specific directories for saving data."""
        split_path = self.graph_dir / self.split_name
        dir_check_make(split_path)
        return split_path

    def _save_splits(
        self,
        split: Dict[str, List[str]],
    ) -> None:
        """Simple utility function to pickle splits."""
        chr_split_dictionary = self.split_path / "training_targets_split.pkl"
        _save_pickle(split, chr_split_dictionary)

    def _save_targets(
        self,
        targets: Dict[str, Dict[str, np.ndarray]],
        scaled: bool = False,
    ) -> None:
        """Simple utility function to pickle targets."""
        if scaled:
            _save_pickle(targets, self.split_path / "training_targets_scaled.pkl")
        else:
            _save_pickle(targets, self.split_path / "training_targets.pkl")

    def filter_genes(
        self,
    ) -> List[str]:
        """Filter genes based on TPM and percent of samples, looping through all
        tissues in the experiment."""
        # prepare vars
        unique_genes = set()

        # filter GTEx genes
        if self.target == "rna_seq":
            filtered_genes = TPMFilter.filtered_genes_from_encode_rna_data(
                rna_seq_file=self.tissue_config.resources["rna"],
                tpm_filter=self.tpm_filter,
            )
        elif self.filter_mode == "across":
            tpm_file = self.experiment_config.expression_all_matrix
        else:
            tpm_file = self.tissue_config.resources["tpm"]
            TPMFilterObj = TPMFilter(
                tissue_config=self.tissue_config,
                split_path=self.split_path,
                tpm_filter=self.tpm_filter,
                percent_of_samples_filter=self.percent_of_samples_filter,
                local_dir=self.local_dir,
            )
            filtered_genes = TPMFilterObj.filter_genes(
                tissue=self.tissue, tpm_file=tpm_file
            )

        # return unique genes with tissue appended
        unique_genes.update(self._append_tissue_to_genes(filtered_genes, self.tissue))
        return list(unique_genes)

    def remove_active_rbp_genes(
        self,
        target_genes: List[str],
    ) -> List[str]:
        """Remove active RBP genes from the target genes, if rbp_network is used for
        graph construction."""
        active_rbp_file = (
            self.experiment_config.interaction_dir
            / self.experiment_config.experiment_name
            / self.tissue_config.resources["tissue"]
            / "interaction"
            / "active_rbps.pkl"
        )
        with open(active_rbp_file, "rb") as f:
            active_rbps = pickle.load(f)
        target_genes = [gene for gene in target_genes if gene not in active_rbps]
        return target_genes

    def assemble_targets(
        self,
        assembler: TargetAssembler,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Assemble training targets, according to the type of target for the experiment."""
        if self.target == "rna_seq":
            return assembler.assemble_rna_targets(tissue_config=self.tissue_config)
        return assembler.assemble_matrix_targets()
        # collected_targets: Dict[str, Dict[str, np.ndarray]] = {
        #     "train": {},
        #     "test": {},
        #     "validation": {},
        # }
        # for split_key in ["train", "test", "validation"]:
        #     for gene_id, target_values in tissue_targets[split_key].items():
        #         if gene_id in collected_targets[split_key]:
        #             collected_targets[split_key][gene_id] += target_values
        #         else:
        # #             collected_targets[split_key][gene_id] = target_values
        # return collected_targets

    def consolidate_training_targets(
        self,
    ) -> List[str]:
        """Get training targets for the network."""
        # filter genes based on TPM and percent of samples
        target_genes = self.filter_genes()
        print(f"Number of genes after filtering: {len(target_genes)}")
        print(f"Some genes: {target_genes[:5]}")

        # remove RBP genes, if active and stored
        if "rbp_network" in self.experiment_config.interaction_types:
            target_genes = self.remove_active_rbp_genes(
                target_genes=target_genes,
            )

        # get dataset split
        splitter = GeneTrainTestSplitter(target_genes=target_genes)
        split = splitter.train_test_val_split(experiment_config=self.experiment_config)

        # get targets
        assembler = TargetAssembler(
            experiment_config=self.experiment_config, split=split
        )
        targets = self.assemble_targets(assembler=assembler)

        # scale targets
        scaled_targets = assembler.scale_targets(targets)

        # save splits and targets
        self._save_splits(split=split)
        self._save_targets(targets=targets, scaled=False)
        self._save_targets(targets=scaled_targets, scaled=True)

        return target_genes

    @staticmethod
    def _append_tissue_to_genes(
        filtered_genes: List[str],
        tissue: str,
    ) -> List[str]:
        """Add tissue to gene names."""
        return [f"{gene}_{tissue}" for gene in filtered_genes]
