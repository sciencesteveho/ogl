import torch
from torch import nn
from torch_geometric.data import Data

from graphormer.functional import batched_shortest_path_distance
from graphormer.layers import CentralityEncoding
from graphormer.layers import GraphormerEncoderLayer
from graphormer.layers import SpatialEncoding


class Graphormer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        input_node_dim: int,
        node_dim: int,
        input_edge_dim: int,
        edge_dim: int,
        n_heads: int,
        max_in_degree: int,
        max_out_degree: int,
        max_path_distance: int,
    ):
        """
        Graphormer model for node regression.

        ...
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        ...
        # No need to define 'self.output_dim' as only scalar output is needed.

        ...
        # Initialization of layers remains the same.
        ...

        # Update the final linear layer to output a scalar instead of a vector.
        self.node_out_lin = nn.Linear(self.node_dim, 1)  # Output a single scalar.

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass for node regression."""
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if isinstance(data, Data):
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        # Apply the final linear transformation.
        x = self.node_out_lin(x)

        # Since the task is regression, we typically want to return raw scores or values (not passed through a non-linearity).
        return (
            x.squeeze()
        )  # Optionally, squeeze the last dimension to get a 1D tensor instead of a 2D one.
