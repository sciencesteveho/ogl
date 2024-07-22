#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Get train / test / val splits for nodes in graphs and generate targets for
training the network. First, genes are filtered based on a TPM cutoff across a
percentage of the samples (e.g. gene must have at least 0.1 TPM across 20% of
samples)."""


import argparse
from pathlib import Path
import pickle
import sys
from typing import Dict, List

import numpy as np

from config_handlers import ExperimentConfig
from config_handlers import TissueConfig
from data_splitter import GeneTrainTestSplitter
from gene_filter import TPMFilter
from target_assembler import TargetAssembler
from utils import _save_pickle
from utils import dir_check_make


def _prepare_split_directories(graph_dir: Path, split_name: str) -> Path:
    """Prep split-specific directories for saving data."""
    split_path = graph_dir / split_name
    dir_check_make(split_path)
    return split_path


def _save_splits(
    split: Dict[str, List[str]],
    split_path: Path,
) -> None:
    """Simple utility function to pickle splits."""
    chr_split_dictionary = split_path / "training_targets_split.pkl"
    _save_pickle(split, chr_split_dictionary)


def _save_targets(
    targets: Dict[str, Dict[str, np.ndarray]],
    split_path: Path,
    scaled: bool = False,
) -> None:
    """Simple utility function to pickle targets."""
    _save_pickle(targets, split_path / "training_targets.pkl")
    if scaled:
        _save_pickle(targets, split_path / "training_targets_scaled.pkl")


def _append_tissue_to_genes(
    filtered_genes: List[str],
    tissue: str,
) -> List[str]:
    """Add tissue to gene names."""
    return [f"{gene}_{tissue}" for gene in filtered_genes]


def filter_genes(
    experiment_config: ExperimentConfig,
    sample_config_dir: Path,
    tissue: str,
    target_mode: str,
    args: argparse.Namespace,
    split_path: Path,
) -> List[str]:
    """Filter genes based on TPM and percent of samples, looping through all
    tissues in the experiment."""
    unique_genes = set()
    for tissue in experiment_config.tissues:
        tissue_config = TissueConfig.from_yaml(sample_config_dir / f"{tissue}.yaml")
        if target_mode == "gtex":
            if args.filter_mode == "across":
                tpm_file = experiment_config.expression_all_matrix
            else:
                tpm_file = tissue_config.resources["tpm"]

            TPMFilterObj = TPMFilter(
                tissue_config=tissue_config,
                split_path=split_path,
                tpm_filter=args.tpm_filter,
                percent_of_samples_filter=args.percent_of_samples_filter,
            )

            filtered_genes = TPMFilterObj.filter_genes(
                tissue=tissue, tpm_file=tpm_file, filter_mode=args.filter_mode
            )
        else:  # target_mode == "rna"
            filtered_genes = TPMFilter.filtered_genes_from_encode_rna_data(
                rna_seq_file=tissue_config.resources["rna"],
                tpm_filter=args.tpm_filter,
            )
        unique_genes.update(_append_tissue_to_genes(filtered_genes, tissue))
    return list(unique_genes)


def remove_active_rbp_genes(
    experiment_config: ExperimentConfig,
    target_genes: List[str],
) -> List[str]:
    """Remove active RBP genes from the target genes, if rbp_network is used for
    graph construction."""
    for tissue in experiment_config.tissues:
        tissue_config = TissueConfig.from_yaml(
            experiment_config.sample_config_dir / f"{tissue}.yaml"
        )
        active_rbp_file = (
            experiment_config.interaction_dir
            / experiment_config.experiment_name
            / tissue_config.resources["tissue"]
            / "interaction"
            / "active_rbps.pkl"
        )
        with open(active_rbp_file, "rb") as f:
            active_rbps = pickle.load(f)
        target_genes = [gene for gene in target_genes if gene not in active_rbps]
    return target_genes


def get_training_targets(
    args: argparse.Namespace,
    experiment_config: ExperimentConfig,
    target_mode: str,
) -> None:
    """Get training targets for the network.

    Args:
        args: argparse.namespace: The arguments for the function.
        experiment_config: ExperimentConfig: The experiment configuration.
        target_mode: str: The mode for selecting targets ("gtex" or "rna").
    """
    # set up vars
    sample_config_dir = experiment_config.sample_config_dir
    graph_dir = experiment_config.graph_dir
    split_path = _prepare_split_directories(graph_dir, args.split_name)

    # filter genes based on TPM and percent of samples
    target_genes = filter_genes(
        experiment_config=experiment_config,
        sample_config_dir=sample_config_dir,
        tissue="all",
        target_mode=target_mode,
        args=args,
        split_path=split_path,
    )

    print(f"Number of genes after filtering: {len(target_genes)}")
    print(f"Some genes: {target_genes[:5]}")

    # remove RBP genes, if active and stored
    if "rbp_network" in experiment_config.interaction_types:
        target_genes = remove_active_rbp_genes(
            experiment_config=experiment_config,
            target_genes=target_genes,
        )

    # get dataset split
    splitter = GeneTrainTestSplitter(target_genes=target_genes)
    split = splitter.train_test_val_split(experiment_config=experiment_config)

    print(f"Checking split: {split}")
    # get targets
    assembler = TargetAssembler(
        experiment_config=experiment_config,
        split=split,
    )

    if target_mode == "gtex":
        targets = assembler.assemble_matrix_targets()
    else:  # target_mode == "rna"
        collected_targets: Dict[str, Dict[str, np.ndarray]] = {
            "train": {},
            "test": {},
            "validation": {},
        }
        for tissue in experiment_config.tissues:
            tissue_config = TissueConfig.from_yaml(sample_config_dir / f"{tissue}.yaml")
            tissue_targets = assembler.assemble_rna_targets(tissue_config=tissue_config)
            for split_key in ["train", "test", "validation"]:
                for gene_id, target_values in tissue_targets[split_key].items():
                    if gene_id not in collected_targets[split_key]:
                        collected_targets[split_key][gene_id] = target_values
                    else:
                        collected_targets[split_key][gene_id] += target_values
        targets = collected_targets

    print(f"Number of genes in training set: {len(targets['train'])}")
    scaled_targets = assembler.scale_targets(targets)

    # save splits and targets
    _save_splits(split=split, split_path=split_path)
    _save_targets(targets=targets, split_path=split_path)
    _save_targets(targets=scaled_targets, split_path=split_path, scaled=True)


def validate_args(args: argparse.Namespace) -> None:
    """Helper function to validate CLI arguments that have dependencies."""
    if args.target != "rna_seq" and args.filter_mode is None:
        print("Error: if target type is not `rna_seq`, --filter_mode is required")
        sys.exit(1)


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
    parser.add_argument(
        "--filter_mode",
        type=str,
        help="Mode to filter genes, specifying within the target tissue or across all possible gtex tissues (e.g. `within` or `across`). This is required if the target type is not `rna_seq`",
        default="within",
    )
    parser.add_argument("--split_name", type=str, help="Name of the split")
    parser.add_argument(
        "--target",
        type=str,
        help="Type of target to generate",
    )
    args = parser.parse_args()
    validate_args(args)

    # load experiment config
    experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
    if args.target == "rna_seq":
        get_training_targets(
            args=args, experiment_config=experiment_config, target_mode="rna"
        )
    else:
        get_training_targets(
            args=args, experiment_config=experiment_config, target_mode="gtex"
        )


if __name__ == "__main__":
    main()
