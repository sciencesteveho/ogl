#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Get train / test / val splits for nodes in graphs and generate targets for
training the network."""

import argparse
from pathlib import Path
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


def get_training_targets(
    args: argparse.namespace, experiment_config: ExperimentConfig, target_mode: str
) -> None:
    """Get training targets for the network.

    Args:
        args: argparse.namespace: The arguments for the function.
        experiment_config: ExperimentConfig: The experiment configuration.
        target_mode: str: The mode for selecting targets ("gtex" or "rna").

    Returns:
        None
    """
    # set up vars
    unique_genes = set()
    config_dir = experiment_config.config_dir
    graph_dir = experiment_config.graph_dir
    split_path = _prepare_split_directories(graph_dir, args.split_name)

    # filter genes based on TPM and percent of samples
    for tissue in experiment_config.tissues:
        tissue_config = TissueConfig.from_yaml(config_dir / f"{tissue}.yaml")
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
    target_genes = list(unique_genes)

    # get dataset split
    splitter = GeneTrainTestSplitter(target_genes=target_genes)
    split = splitter.train_test_val_split(experiment_config=experiment_config)

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
            tissue_config = TissueConfig.from_yaml(config_dir / f"{tissue}.yaml")
            tissue_targets = assembler.assemble_rna_targets(tissue_config=tissue_config)
            for split_key in ["train", "test", "validation"]:
                for gene_id, target_values in tissue_targets[split_key].items():
                    if gene_id not in collected_targets[split_key]:
                        collected_targets[split_key][gene_id] = target_values
                    else:
                        collected_targets[split_key][gene_id] += target_values
        targets = collected_targets

    scaled_targets = assembler.scale_targets(targets)

    # save splits and targets
    _save_splits(split=split, split_path=split_path)
    _save_targets(targets=targets, split_path=split_path)
    _save_targets(targets=scaled_targets, split_path=split_path, scaled=True)


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
        help="Mode to filter genes (e.g. `within` or `across`)",
    )
    parser.add_argument("--split_name", type=str, help="Name of the split")
    parser.add_argument(
        "--target_type",
        type=str,
        help="Type of target to generate (e.g. `gtex` or `rna`)",
    )
    args = parser.parse_args()

    # load experiment config
    experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
    if args.target_type == "gtex":
        get_training_targets(
            args=args, experiment_config=experiment_config, target_mode="gtex"
        )
    elif args.target_type == "rna":
        get_training_targets(
            args=args, experiment_config=experiment_config, target_mode="rna"
        )


if __name__ == "__main__":
    main()
