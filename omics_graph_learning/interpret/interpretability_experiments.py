#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run interpretability experiments on a trained model.
1 - Gradient-based saliency maps
2 - Attention weights, if applicable (only if model is GATv2)
3 - PGExplainer subgraph explanations for top nodes
"""


import torch
from torch import nn  # type: ignore
from torch_geometric.data import Data  # type: ignore

from omics_graph_learning.interpret.attention_weights import get_attention_weights
from omics_graph_learning.interpret.explainer import build_explainer
from omics_graph_learning.interpret.explainer import train_explainer
from omics_graph_learning.interpret.interpret_utils import _interpret_setup
from omics_graph_learning.interpret.interpret_utils import combine_masks
from omics_graph_learning.interpret.interpret_utils import get_baseline_predictions
from omics_graph_learning.interpret.interpret_utils import get_best_predictions
from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import parse_interpret_args
from omics_graph_learning.interpret.node_feat_perturbation import perturb_node_features
from omics_graph_learning.interpret.perturb_runner import PerturbRunner
from omics_graph_learning.interpret.saliency import compute_gradient_saliency
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.constants import RANDOM_SEEDS


def main() -> None:
    """Run experiments!"""
    # parse arguments
    args = parse_interpret_args()

    # load experiment setup
    (
        data,
        device,
        runner,
        _,
        _,
        _,
        _,
        _,
        outpath,
    ) = _interpret_setup(args)

    # get all_mask for all nodes
    data = combine_masks(data)

    # saliency maps
    # we are only interested in computing saliency maps for the genes
    saliency_map = compute_gradient_saliency(
        model=runner.model,
        data=data,
        device=device,
        mask=data.all_mask_loss,
    )
    torch.save(saliency_map, outpath / "saliency_map.pt")

    # attention weights for genes
    attention_weights = get_attention_weights(
        model=runner.model,
        data=data,
        mask=data.all_mask_loss,
    )
    torch.save(attention_weights, outpath / "attention_weights_genes.pt")

    # attention weights for all nodes
    attention_weights_all = get_attention_weights(
        model=runner.model,
        data=data,
        mask=data.all_mask,
    )
    torch.save(attention_weights_all, outpath / "attention_weights_all_nodes.pt")

    # explainer
    pg_explainer = build_explainer(model=runner.model, epochs=30, lr=0.003)
    explanations = train_explainer(pg_explainer, data)
    torch.save(explanations, outpath / "pgexplainer_explanations.pt")


if __name__ == "__main__":
    main()
