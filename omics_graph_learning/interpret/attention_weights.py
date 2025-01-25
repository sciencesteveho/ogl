# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Module to extract attention weights from a trained GATv2 models. The models
are trained with (return_attention_weights=False), so we:
    1) Set each GATv2Conv's return_attention_weights=True.
    2) Register a forward hook that captures the alpha values.
    3) Perform a forward pass with.
    4) Return a {layer_name: alpha_tensor} dictionary."""


from typing import Dict, List, Optional, Tuple

import torch  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.nn import GATv2Conv  # type: ignore


def register_gat_hooks(model: torch.nn.Module) -> None:
    """Registers a forward hook on each GATv2Conv to capture attention weights
    from the forward pass. The captured tensor is stored in
    `module.cached_alpha`.
    """

    def gat_attention_hook(
        module: GATv2Conv,
        input: Tuple[torch.Tensor, ...],
        output: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """If `return_attention_weights=True`, the forward pass of GATv2Conv
        returns a tuple: (out, (edge_index, alpha)).
        """
        if isinstance(output, tuple) and len(output) == 2:
            _, alpha_info = output  # alpha_info = (edge_index, alpha)
            if len(alpha_info) == 2:
                _, alpha = alpha_info
                module.cached_alpha = alpha.detach().cpu()
            else:
                module.cached_alpha = None
        else:
            module.cached_alpha = None

    # attach hook to each GATv2Conv
    for _, submodule in model.named_modules():
        if isinstance(submodule, GATv2Conv):
            submodule.register_forward_hook(gat_attention_hook)


def get_attention_weights(
    model: torch.nn.Module,
    data: Data,
    mask: torch.Tensor,
    batch_size: int = 128,
) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Extract GATv2 attention weights in batches for a large graph using
    NeighborLoader.

    Returns:
        A dictionary mapping where each element of list_of_batches is
        (global_edges, alpha) for that layer:
            - global_edges is a torch.LongTensor of shape [2, E_sub], giving the
              GLOBAL node indices for the edges in this batch.
            - alpha is the attention tensor of shape [E_sub, num_heads] (or
              similar).
    """
    alphas: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    # return attention weights in forward pass
    for _, submodule in model.named_modules():
        if isinstance(submodule, GATv2Conv):
            submodule.return_attention_weights = True

    # register hooks to capture attention weights
    register_gat_hooks(model)

    # load data in batches
    loader = NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=batch_size,
        input_nodes=mask,
        shuffle=False,
    )

    model.eval()
    with torch.no_grad():
        for batch_data in loader:

            # forward pass
            sub_mask = mask[batch_data.n_id]
            _ = model(batch_data.x, batch_data.edge_index, sub_mask)

            # retrieve the alpha values stored by the hook for each conv
            for layer_name, submodule in model.named_modules():
                if isinstance(submodule, GATv2Conv):
                    alpha_local = getattr(submodule, "cached_alpha", None)
                    if alpha_local is not None:
                        # map local node indices -> global node indices:
                        global_edges = batch_data.n_id[batch_data.edge_index]

                        # add to alpha_dict
                        if layer_name not in alphas:
                            alphas[layer_name] = []
                        alphas[layer_name].append(
                            (global_edges.cpu(), alpha_local.cpu())
                        )

    return alphas
