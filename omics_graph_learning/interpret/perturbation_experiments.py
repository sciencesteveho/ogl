#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run graph perturbation experiments."""


import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

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

# from omics_graph_learning.interpret.coessential_pair_perturbation import (
#     load_coessential_pairs,
# )
# from omics_graph_learning.interpret.coessential_pair_perturbation import (
#     paired_gene_perturbation,
# )
# from omics_graph_learning.interpret.essential_gene_perturbation import (
#     essential_gene_perturbation,
# )


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
    # baseline_df.to_csv(outpath / "baseline_predictions.csv", index=False)
    # baseline_df.to_csv(outpath / f"baseline_predictions_{hops}_hop.csv", index=False)

    # get best predictions from model
    # print("Getting best predictions...")
    # best_prediction_df = get_best_predictions(
    #     df=baseline_df,
    #     gene_indices=gene_indices,
    #     node_idx_to_gene_id=node_idx_to_gene_id,
    #     gencode_to_symbol=gencode_to_symbol,
    # )
    # best_prediction_df.to_csv(outpath / f"best_predictions_{hops}_hop.csv", index=False)

    # experiment 1: run node feature ablation
    # print("Running Node Feature Perturbation...")
    # feature_fold_changes, feature_top_genes = perturb_node_features(
    #     data=data,
    #     runner=runner,
    #     feature_indices=list(range(5, 42)),
    #     device=device,
    #     node_idx_to_gene_id=node_idx_to_gene_id,
    #     gencode_to_symbol=symbol_to_gencode,
    # )

    # with open(outpath / "node_feature_perturbations.pkl", "wb") as f:
    #     pickle.dump(feature_fold_changes, f)
    # with open(outpath / "node_feature_top_genes.pkl", "wb") as f:
    #     pickle.dump(feature_top_genes, f)

    # experiment 2: run systematic connected component perturbations on the
    # k-hop subgraph
    # print("Running Connected Component Perturbation...")
    # experiment = ConnectedComponentPerturbation(
    #     data=data,
    #     device=device,
    #     runner=runner,
    #     idxs_inv=idxs_inv,
    #     mask_attr="all",
    # )

    # best_prediction_df = pd.read_csv(outpath / f"best_predictions_{hops}_hop.csv")
    # genes_to_analyze = best_prediction_df["node_idx"].tolist()
    # component_perturbation_results = experiment.run_perturbations(
    #     genes_to_analyze=genes_to_analyze,
    # )
    # with open(outpath / f"connected_component_perturbations_{hops}_hop.pkl", "wb") as f:
    #     pickle.dump(component_perturbation_results, f)

    # experiment 3: run systematic selected component perturbations on the node
    # features for top genes
    print("Running Selected Component Perturbation...")
    experiment = SelectedComponentPerturbation(
        data=data,
        device=device,
        runner=runner,
        idxs_inv=idxs_inv,
        mask_attr="all",
    )

    # load the top perturbations
    top_pert = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/interpretation/top_perturbations.pkl"
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
    with open(outpath / "selected_component_perturbations.pkl", "wb") as f:
        pickle.dump(results, f)

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
