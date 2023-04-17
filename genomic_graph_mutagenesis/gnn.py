#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]
#

"""Code to train GNNs on the graph data!"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from graph_to_pytorch import graph_to_pytorch


# Define/Instantiate GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_size, embedding_size, out_size):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_size, embedding_size)  # GCNConv, SAGEConv
        self.conv1.aggr = "max"
        self.conv2 = SAGEConv(embedding_size, out_size)
        self.conv2.aggr = "max"
        self.conv3 = SAGEConv(embedding_size, out_size)
        self.conv3.aggr = "max"
        self.conv4 = SAGEConv(embedding_size, out_size)
        self.conv4.aggr = "max"
        self.conv5 = SAGEConv(embedding_size, out_size)
        self.conv5.aggr = "max"
        self.conv6 = SAGEConv(embedding_size, out_size)
        self.conv6.aggr = "max"

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv6(x, edge_index)
        return x


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.mse_loss(out[data.train_mask].squeeze(), data.y[data.train_mask].squeeze())

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()

    out = model(data.x, data.edge_index)

    train_acc = F.mse_loss(
        out[data.train_mask].squeeze(), data.y[data.train_mask].squeeze()
    )
    val_acc = F.mse_loss(
        out[data.val_mask].squeeze(), data.y[data.val_mask].squeeze()
    )
    test_acc = F.mse_loss(
        out[data.test_mask].squeeze(), data.y[data.test_mask].squeeze()
    )

    return train_acc, val_acc, test_acc


def main() -> None:
    """_summary_"""
    ### Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed to use (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset storage.",
        default="/ocean/projects/bio210019p/stevesho/data/preprocess",
    )
    parser.add_argument(
        "--device",
        type=int, 
        default=0,
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="full",
    )
    args = parser.parse_args()

    data = graph_to_pytorch(
        root_dir=args.root,
        graph_type=args.graph_type,
    )

    ### check for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    model = GNN(in_size=data.x.shape[1], embedding_size=500, out_size=1).to_device()
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.MSELoss()

    epochs = 100
    for epoch in range(0, epochs):
        loss = train(model, optimizer, data)
        train_acc, val_acc, test_acc = test(model, data)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss}, Train: {train_acc:.4f}, Validation: {val_acc:.4f}, Test: {test_acc:.4f}"
            )


if __name__ == "__main__":
    main()
