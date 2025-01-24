#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Run interpretability experiments on a trained model.
1 - Gradient-based saliency maps
2 - Attention weights, if applicable (only if model is GATv2)
3 - PGExplainer subgraph explanations for top nodes
"""


import torch

from omics_graph_learning.interpret.attention_weights import get_attention_weights
from omics_graph_learning.interpret.explainer import build_explainer
from omics_graph_learning.interpret.explainer import train_explainer
from omics_graph_learning.interpret.interpret_utils import _interpret_setup
from omics_graph_learning.interpret.interpret_utils import combine_masks
from omics_graph_learning.interpret.interpret_utils import parse_interpret_args
from omics_graph_learning.interpret.saliency import compute_gradient_saliency


def scale_saliency(saliency_map: torch.Tensor) -> torch.Tensor:
    """Scale saliency map to [0, 1]."""
    saliency_min, _ = saliency_map.min(dim=0, keepdim=True)  # min feature
    saliency_max, _ = saliency_map.max(dim=0, keepdim=True)  # max feature
    return (saliency_map - saliency_min) / (saliency_max - saliency_min + 1e-9)


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
    gene_indices = data.all_mask_loss.nonzero(as_tuple=False).squeeze()

    saliency_map = compute_gradient_saliency(
        model=runner.model,
        data=data,
        device=device,
        gene_indices=gene_indices,
    )

    # scale saliency map
    scaled_saliency = scale_saliency(saliency_map)
    torch.save(scaled_saliency, outpath / "scaled_saliency_map.pt")

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
