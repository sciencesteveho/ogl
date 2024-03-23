#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Regression task layers
        self.regression_head = torch.nn.Linear(hidden_dim, 1)

        # Auxiliary edge prediction task layers
        self.edge_prediction_head = torch.nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, data, edge_index_aux):
        x, edge_index = data.x, data.edge_index

        # Generate node embeddings with the primary graph
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # Node embeddings are used for both regression and auxiliary tasks

        # Regression output (e.g., for graph-level prediction, we pool node features)
        pooled = global_mean_pool(
            x, data.batch
        )  # Assuming we have batch-wise data aggregation
        reg_output = self.regression_head(pooled)

        # Auxiliary edge prediction output
        # Here, we consider pairs of nodes as potential edges in the secondary graph
        edge_pred_output = self.edge_prediction_head(
            x[edge_index_aux[0]], x[edge_index_aux[1]]
        )

        return reg_output, edge_pred_output


# Mock Data
num_nodes = 10
num_node_features = 16
num_edges = 30
hidden_dim = 32
# Primary graph data
data = Data(
    x=torch.randn(num_nodes, num_node_features),
    edge_index=torch.randint(0, num_nodes, (2, num_edges)),
    y=torch.randn(1, 1),  # Graph level regression target
    batch=torch.zeros(
        num_nodes, dtype=torch.long
    ),  # Example for single graph; in practice, you might have a batch of graphs
)

# Secondary graph's potential edges, for which we want the model to predict existence
# Here, we simply use all possible edges, but in practice you would use the specific edges of the secondary graph
# These are pairs of node indices for which we want to predict the presence of an edge.
secondary_edges_idx = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()

# Define the model and move to the correct device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(num_node_features, hidden_dim).to(device)
data = data.to(device)
secondary_edges_idx = secondary_edges_idx.to(device)

# Forward pass (example usage)
reg_output, edge_pred_output = model(data, secondary_edges_idx)

# Assume we have the true edges of the secondary graph in a binary form (1 if edge exists, 0 otherwise)
# For the example, we randomly generate them
true_edge_labels = (
    torch.randint(0, 2, (secondary_edges_idx.size(1),)).to(device).float().view(-1, 1)
)

# Calculate losses
regression_loss = F.mse_loss(reg_output, data.y)
edge_prediction_loss = F.binary_cross_entropy_with_logits(
    edge_pred_output, true_edge_labels
)

# Combine the losses
combined_loss = regression_loss + edge_prediction_loss

# Imagine that we are in a training loop, the following steps would occur:
# optimizer.zero_grad() # Clears existing gradients
# combined_loss.backward() # Backpropagation
# optimizer.step() # Update model parameters
