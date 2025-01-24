#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run graph perturbation experiments."""


import argparse
import os
from pathlib import Path
import pickle
from typing import List, Tuple

import pandas as pd
from scipy.stats import ttest_ind  # type: ignore
import torch

from omics_graph_learning.graph_to_pytorch import GraphToPytorch
from omics_graph_learning.interpret.connected_component_perturbation import (
    ConnectedComponentPerturbation,
)
from omics_graph_learning.interpret.interpret_utils import get_baseline_predictions
from omics_graph_learning.interpret.interpret_utils import get_best_predictions
from omics_graph_learning.interpret.interpret_utils import load_data_and_model
from omics_graph_learning.interpret.node_feat_perturbation import perturb_node_features
from omics_graph_learning.interpret.perturb_runner import PerturbRunner
from omics_graph_learning.utils.common import _dataset_split_name
from omics_graph_learning.utils.config_handlers import ExperimentConfig
from omics_graph_learning.utils.constants import RANDOM_SEEDS

# from omics_graph_learning.interpret.coessential_pair_perturbation import (
#     load_coessential_pairs,
# )
# from omics_graph_learning.interpret.coessential_pair_perturbation import (
#     paired_gene_perturbation,
# )
# from omics_graph_learning.interpret.essential_gene_perturbation import (
#     essential_gene_perturbation,
# )


def invert_symbol_dict(symbol_to_gencode: dict[str, str]) -> dict[str, str]:
    """Create a dictionary that goes ENSG -> symbol."""
    return {ensg: symbol for symbol, ensg in symbol_to_gencode.items()}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config", type=str, help="Path to experiment config."
    )
    parser.add_argument(
        "--run_number",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Run number to determine seed. Options: 1, 2, 3.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of model directory used for training. This will be where the runs are stored. If not specified, will use the model name from the experiment config. If specified, then overwrite (such as for models with replicate runs)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="Name of trained model checkpoint.",
        default="GAT_best_model.pt",
    )
    # parser.add_argument(
    #     "--experiment",
    #     type=str,
    #     default="all",
    #     help="Which experiment(s) to run: node_features, connected_components, essential, nonessential, coessential, all.",
    # )
    # parser.add_argument(
    #     "--lethal_file",
    #     type=str,
    #     default="/lethal_genes.txt",
    #     help="Path to lethal_genes.txt for essential gene perturbation.",
    # )
    # parser.add_argument(
    #     "--pos_pairs_file",
    #     type=str,
    #     default="/coessential_pos.txt",
    #     help="Path to positive coessential pairs file.",
    # )
    # parser.add_argument(
    #     "--neg_pairs_file",
    #     type=str,
    #     default="/coessential_neg.txt",
    #     help="Path to negative coessential pairs file.",
    # )
    return parser.parse_args()


def _derive_paths(
    experiment_config: ExperimentConfig,
) -> Tuple[str, str, str, str, str]:
    """Derive the following requireds paths and variables from the experiment
    config:
        - root directory
        - split_name
        - idx_file
        - gene_id_lookup
    idx file, graph file, gencode lookup
    """
    if len(experiment_config.tissues) > 1:
        raise ValueError("Only one tissue is supported for perturbation experiments.")

    # get root dir
    root_dir = experiment_config.root_dir

    # get split name
    split_name = _dataset_split_name(
        test_chrs=experiment_config.test_chrs,
        val_chrs=experiment_config.val_chrs,
        tpm_filter=0.5,
        percent_of_samples_filter=0.1,
    )
    split_name += "_rna_seq"

    # get idx file
    experiment_name = experiment_config.experiment_name
    graph_type = experiment_config.graph_type
    experiment_dir = f"{root_dir}/experiments/{experiment_name}/graphs/{split_name}"
    idx_file = f"{experiment_dir}/{experiment_name}_{graph_type}_graph_idxs.pkl"

    # get gencode lookup
    gene_id_lookup = (
        f"{experiment_config.reference_dir}/gencode_to_genesymbol_lookup_table.txt"
    )

    return root_dir, split_name, experiment_name, idx_file, gene_id_lookup


def _create_pyg_data(
    experiment_config: ExperimentConfig,
    outpath: Path,
    split_name: str,
    experiment_name: str,
) -> str:
    """Convert numpy graphs to pyg data object and save if it doesn't already
    exist
    """
    # convert graph_data to pytorch
    # check if file exists
    graph_file = f"{outpath}/{experiment_name}_graph_data.pt"
    if os.path.exists(graph_file):
        return graph_file

    target = "rna_seq"
    positional_encoding = True

    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=split_name,
        regression_target=target,
        positional_encoding=positional_encoding,
    ).make_data_object()
    torch.save(data, graph_file)

    return graph_file


def main() -> None:
    """Run graph perturbation experiments."""
    args = parse_args()

    # set seed and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = RANDOM_SEEDS[args.run_number - 1]
    torch.manual_seed(seed)

    # load experiment config to derive paths
    experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
    root_dir, split_name, experiment_name, idx_file, gene_id_lookup = _derive_paths(
        experiment_config=experiment_config
    )

    # make outpath
    outpath = root_dir / "interpretation" / experiment_name
    os.makedirs(outpath, exist_ok=True)

    # get graph file
    graph_file = _create_pyg_data(
        experiment_config=experiment_config,
        outpath=outpath,
        split_name=split_name,
        experiment_name=experiment_name,
    )

    # get model checkpoint
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = experiment_config.model_name

    model_checkpoint = (
        f"{root_dir}/models/{model_name}"
        f"/run_{args.run_number}/{args.model_checkpoint}"
    )

    # load data
    (
        data,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        symbol_to_gencode,
    ) = load_data_and_model(
        lookup_file=gene_id_lookup,
        graph_file=graph_file,
        idx_file=idx_file,
        model_file=model_checkpoint,
        device=device,
    )

    # invert symbol dict
    gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

    # get baseline predictions
    baseline_df = get_baseline_predictions(data=data, runner=runner)
    baseline_df.to_csv(outpath / "baseline_predictions.csv", index=False)

    # get best predictions from model
    print("Getting best predictions...")
    best_prediction_df = get_best_predictions(
        df=baseline_df,
        gene_indices=gene_indices,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=gencode_to_symbol,
    )
    best_prediction_df.to_csv(outpath / "best_predictions.csv", index=False)

    # run connected component perturbations
    print("Running Connected Component Perturbation...")
    experiment = ConnectedComponentPerturbation(
        data=data,
        device=device,
        runner=runner,
        idxs_inv=idxs_inv,
        mask_attr="all",
    )

    genes_to_analyze = best_prediction_df["node_idx"].tolist()
    component_perturbation_results = experiment.run_perturbations(
        genes_to_analyze=genes_to_analyze,
    )
    with open(outpath / "connected_component_perturbations.pkl", "w") as f:
        pickle.dump(component_perturbation_results, f)

    # run node feature ablation
    print("Running Node Feature Perturbation...")
    feature_fold_changes, feature_top_genes = perturb_node_features(
        data=data,
        runner=runner,
        feature_indices=list(range(5, 42)),
        mask="all",
        device=device,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=symbol_to_gencode,
    )

    with open(outpath / "node_feature_perturbations.pkl", "wb") as f:
        pickle.dump(feature_fold_changes, f)
    with open(outpath / "node_feature_top_genes.pkl", "wb") as f:
        pickle.dump(feature_top_genes, f)

    # print("Running Essential Gene Perturbation...")
    # essential_fold_changes = essential_gene_perturbation(
    #     data=data,
    #     runner=runner,
    #     idxs=idxs,
    #     gencode_to_symbol=gencode_to_symbol,
    #     output_prefix=outpath,
    #     sample=sample,
    #     lethal_file=lethal_file,
    #     mask=mask,
    #     device=device,
    #     essential=True,
    # )

    # print("Running Non-Essential Gene Perturbation...")
    # nonessential_fold_changes = essential_gene_perturbation(
    #     data=data,
    #     runner=runner,
    #     idxs=idxs,
    #     gencode_to_symbol=gencode_to_symbol,
    #     output_prefix=outpath,
    #     sample=sample,
    #     lethal_file=lethal_file,
    #     mask=mask,
    #     device=device,
    #     essential=False,
    # )

    # print("Running Coessential Pair Perturbation...")
    # pos_pairs, neg_pairs = load_coessential_pairs(pos_pairs_file, neg_pairs_file, idxs)
    # coessential_changes, random_changes = paired_gene_perturbation(
    #     data=data,
    #     runner=runner,
    #     pairs=pos_pairs,  # or neg_pairs
    #     num_hops=6,
    #     device=device,
    #     random_comparison=True,
    # )

    # # simple t-test for coessential vs random perturbations
    # t_stat, p_value = ttest_ind(coessential_changes, random_changes, equal_var=False)
    # print("T-test results for positive coessential pairs vs random perturbations:")
    # print(f"T-statistic: {t_stat}, P-value: {p_value}")

    # print("All experiments complete.")


if __name__ == "__main__":
    main()
