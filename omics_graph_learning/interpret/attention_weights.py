# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Module to extract attention weights from a trained GATv2 models using a dummy
model with alpha return.
"""


from typing import Dict, List, Optional, Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.nn import GATv2Conv  # type: ignore
from tqdm import tqdm  # type: ignore


class GATv2ConvWithAlpha(nn.Module):
    """A wrapper around the standard GATv2Conv that always returns (out, alpha)."""

    def __init__(self, original_gat: GATv2Conv) -> None:
        """Initialize the wrapper with the original GATv2Conv."""
        super().__init__()
        self.gat = original_gat

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass to return (out, alpha)."""
        out, (edge_iindex, alpha) = self.gat(
            x,
            edge_index,
            return_attention_weights=True,
        )
        return out, alpha


class GATv2AnalysisModel(nn.Module):
    """A dummy model that mimics the original GATv2 model, but with alpha
    extraction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_layers: int = 2,
    ) -> None:
        """Initialize the analysis model with the same layer shapes."""
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                conv = GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                )
            else:
                conv = GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                )
            self.layers.append(GATv2ConvWithAlpha(conv))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run forward pass and return activations and alpha for each layer."""
        all_alphas = []
        for layer in self.layers:
            out, alpha = layer(x, edge_index)
            x = torch.relu(out)  # dummy activation
            all_alphas.append(alpha)
        return x, all_alphas


def load_weights_from_original(
    original_model: nn.Module,
    analysis_model: GATv2AnalysisModel,
) -> None:
    """Copy weights from the original GATv2 model to the analysis model."""
    analysis_layers = analysis_model.layers
    gat_layers: List[Tuple[str, GATv2Conv]] = []
    gat_layers.extend(
        (name, submodule)
        for name, submodule in original_model.named_modules()
        if isinstance(submodule, GATv2Conv)
    )
    gat_layers.sort(key=lambda x: x[0])  # sort by name e.g. convs.0, convs.1

    if len(gat_layers) != len(analysis_layers):
        print("Warning: Mismatch in number of GATv2Conv layers.")
        print(
            f"Original model has {len(gat_layers)} GAT layers; analysis has {len(analysis_layers)}."
        )

    # copy each submodule's weights
    for i, ((original_name, orig_conv), analysis_layer) in enumerate(
        zip(gat_layers, analysis_layers)
    ):
        analysis_layer.gat.load_state_dict(orig_conv.state_dict(), strict=True)
        print(
            f"Copied weights from original layer '{original_name}' -> analysis_layer[{i}]"
        )


def extract_attention(
    analysis_model: GATv2AnalysisModel,
    data: Data,
    mask: Optional[torch.Tensor],
    batch_size: int = 1024,
    num_neighbors: int = 15,
) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Run the analysis model in mini-batches to capture alpha."""
    from torch_geometric.loader import NeighborLoader

    alphas: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {
        i: [] for i in range(analysis_model.num_layers)
    }

    # convert boolean mask -> node indices
    input_nodes = mask.nonzero(as_tuple=True)[0] if mask is not None else None
    loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors] * 2,
        batch_size=batch_size,
        input_nodes=input_nodes,
        shuffle=False,
    )

    total_batches = len(loader)
    analysis_model.eval()
    with torch.no_grad():
        for batch_data in tqdm(loader, total=total_batches):
            # forward pass
            _, all_alphas = analysis_model(batch_data.x, batch_data.edge_index)
            for layer_i, alpha in enumerate(all_alphas):
                # map local->global for edges
                global_edges = batch_data.n_id[batch_data.edge_index.cpu()]
                alphas[layer_i].append((global_edges.cpu(), alpha.cpu()))

    return alphas


def get_attention_weights(
    original_model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    heads: int,
    num_layers: int,
    batch_size: int = 1024,
    num_neighbors: int = 15,
) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Get the attention weights from a trained GATv2 model.
    1 - build dummy
    2 - copy weights
    3 - batch infer alpha
    4 - return {layer_idx: list_of_batches}
    """
    analysis_model = GATv2AnalysisModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        num_layers=num_layers,
    ).to(data.x.device)

    # copy weights from original model
    load_weights_from_original(original_model, analysis_model)

    return extract_attention(
        analysis_model,
        data,
        mask,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
    )
