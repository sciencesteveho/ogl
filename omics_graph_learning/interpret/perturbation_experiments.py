#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run graph perturbation experiments."""


import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import joblib  # type: ignore
import pandas as pd
from scipy.stats import ttest_ind  # type: ignore
import torch

from omics_graph_learning.interpret.connected_component_perturbation import (
    ConnectedComponentPerturbation,
)
from omics_graph_learning.interpret.interpret_utils import _interpret_setup
from omics_graph_learning.interpret.interpret_utils import (
    get_baseline_predictions_k_hop,
)
from omics_graph_learning.interpret.interpret_utils import get_best_predictions
from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import parse_interpret_args
from omics_graph_learning.interpret.node_feat_perturbation import perturb_node_features
from omics_graph_learning.interpret.selected_component_perturbation import (
    SelectedComponentPerturbation,
)
from omics_graph_learning.utils.config_handlers import ExperimentConfig

# from omics_graph_learning.interpret.coessential_pair_perturbation import (
#     load_coessential_pairs,
# )
# from omics_graph_learning.interpret.coessential_pair_perturbation import (
#     paired_gene_perturbation,
# )
# from omics_graph_learning.interpret.essential_gene_perturbation import (
#     essential_gene_perturbation,
# )


def load_scalers(scaler_dir: str, num_features: int = 37) -> Dict[int, object]:
    """Load each feat_{i}_scaler.joblib into a dictionary:
    {feature_idx -> scaler_pipeline}.
    """
    scalers = {}
    for feat_idx in range(num_features):
        path = os.path.join(scaler_dir, f"feat_{feat_idx}_scaler.joblib")
        if os.path.isfile(path):
            scalers[feat_idx] = joblib.load(path)

    return scalers


def convert_top_perturbations(
    top_perturbations: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    sample: str,
    idxs: Dict[str, int],
) -> List[Tuple[int, int]]:
    """Convert a nested dictionary of top perturbations into a list of
    (gene_node_idx, node_to_perturb_idx) tuples.
    """
    perturbations: List[Tuple[int, int]] = []
    tissue_perturbations: Dict[str, Dict[str, Dict[str, float]]] = top_perturbations[
        sample
    ]

    for perturbation_info in tissue_perturbations.values():
        for gene_node, selected_perturbations in perturbation_info.items():
            for node_to_perturb in selected_perturbations.keys():
                node_to_perturb_sample = f"{node_to_perturb}_{sample}"
                if gene_node in idxs and node_to_perturb_sample in idxs:
                    perturbations.append(
                        (idxs[gene_node], idxs[node_to_perturb_sample])
                    )
    return perturbations


def main() -> None:
    """Run graph perturbation experiments."""
    # parse arguments
    args = parse_interpret_args()
    hops = args.hops

    experiment_config = ExperimentConfig.from_yaml(args.experiment_config)
    splitname = "tpm_0.5_samples_0.1_test_8-9_val_10_rna_seq"
    scaler_path = experiment_config.graph_dir / splitname / "scalers"
    scalers = load_scalers(str(scaler_path))

    # check that scalers are real from idx 5 onwards
    assert all(scalers[i] is not None for i in range(37))

    # load experiment setup
    (
        data,
        device,
        runner,
        node_idx_to_gene_id,
        gene_indices,
        idxs_inv,
        idxs,
        symbol_to_gencode,
        outpath,
    ) = _interpret_setup(args)

    # invert symbol dict
    gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

    # get baseline predictions
    baseline_df = get_baseline_predictions_k_hop(
        data=data,
        runner=runner,
        k=hops,
    )
    baseline_df.to_csv(outpath / f"baseline_predictions_{hops}_hop.csv", index=False)

    # get best predictions from model
    print("Getting best predictions...")
    best_prediction_df = get_best_predictions(
        df=baseline_df,
        gene_indices=gene_indices,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=gencode_to_symbol,
    )
    best_prediction_df.to_csv(outpath / f"best_predictions_{hops}_hop.csv", index=False)

    # experiment 3: run systematic selected component perturbations on the node
    # features for top genes
    print("Running Selected Component Perturbation...")
    experiment = SelectedComponentPerturbation(
        data=data,
        device=device,
        runner=runner,
        idxs_inv=idxs_inv,
        mask_attr="all",
        scalers=scalers,
    )

    # load the top perturbations
    top_pert = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/interpretation/top_perturbations_1000.pkl"
    with open(top_pert, "rb") as f:
        top_perturbations = pickle.load(f)

    perturbations = convert_top_perturbations(
        top_perturbations=top_perturbations,
        sample=args.sample,
        idxs=idxs,
    )

    # run perturbations
    results = experiment.run_perturbations(
        gene_node_pairs=perturbations,
    )

    # save results
    with open(outpath / "selected_component_perturbations_1000.pkl", "wb") as f:
        pickle.dump(results, f)

    # experiment 1: run node feature ablation doubles
    print("Running Node Feature Perturbation...")
    avg_diffs, feature_fold_changes, feature_top_genes = perturb_node_features(
        data=data,
        runner=runner,
        feature_indices=list(range(37)),
        # feature_indices=deletion_pairs,
        device=device,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=symbol_to_gencode,
        scalers=scalers,
    )

    # rename idx keys in avg_diffs to gene_ids
    avg_diffs = {node_idx_to_gene_id[k]: v for k, v in avg_diffs.items()}

    with open(outpath / "node_feature_perturbations_avg_diffs.pkl", "wb") as f:
        pickle.dump(avg_diffs, f)
    with open(outpath / "node_feature_perturbations.pkl", "wb") as f:
        pickle.dump(feature_fold_changes, f)
    with open(outpath / "node_feature_top_genes.pkl", "wb") as f:
        pickle.dump(feature_top_genes, f)

    deletion_pairs = [
        (2, 4),  # ATAC + CpG
        (2, 5),  # ATAC + CTCF
        (2, 6),  # ATAC + DNase
        (2, 7),  # ATAC + H3K27ac
        (2, 8),  # ATAC + H3K27me3
        (2, 9),  # ATAC + H3K36me3
        (2, 10),  # ATAC + H3K4me1
        (2, 11),  # ATAC + H3K4me2
        (2, 12),  # ATAC + H3K4me3
        (2, 13),  # ATAC + H3K79me2
        (2, 14),  # ATAC + H3K9ac
        (2, 15),  # ATAC + H3K9me3
        (2, 19),  # ATAC + microsatellites
        (7, 8),  # H3K27ac + H3K27me3
        (7, 9),  # H3K27ac + H3K36me3
        (7, 10),  # H3K27ac + H3K4me1
        (7, 11),  # H3K27ac + H3K4me2
        (7, 12),  # H3K27ac + H3K4me3
        (7, 13),  # H3K27ac + H3K79me2
        (7, 14),  # H3K27ac + H3K9ac
        (7, 15),  # H3K27ac + H3K9me3
    ]

    # experiment 2: run node feature ablation doubles
    print("Running Node Feature Perturbation...")
    feature_fold_changes, feature_top_genes = perturb_node_features(
        data=data,
        runner=runner,
        # feature_indices=list(range(5, 42)),
        feature_indices=deletion_pairs,
        device=device,
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=symbol_to_gencode,
        scalers=scalers,
    )

    with open(outpath / "node_feature_perturbations_double.pkl", "wb") as f:
        pickle.dump(feature_fold_changes, f)
    with open(outpath / "node_feature_top_genes_double.pkl", "wb") as f:
        pickle.dump(feature_top_genes, f)

    # experiment 2: run systematic connected component perturbations on the
    # k-hop subgraph
    print("Running Connected Component Perturbation...")
    experiment = ConnectedComponentPerturbation(
        data=data,
        device=device,
        runner=runner,
        idxs_inv=idxs_inv,
        mask_attr="all",
    )

    best_prediction_df = pd.read_csv(outpath / f"best_predictions_{hops}_hop.csv")
    genes_to_analyze = best_prediction_df["node_idx"].tolist()
    component_perturbation_results = experiment.run_perturbations(
        genes_to_analyze=genes_to_analyze,
    )
    with open(outpath / f"connected_component_perturbations_{hops}_hop.pkl", "wb") as f:
        pickle.dump(component_perturbation_results, f)


if __name__ == "__main__":
    main()
