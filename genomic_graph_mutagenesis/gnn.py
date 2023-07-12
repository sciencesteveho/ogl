#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] fix layer sizes
# - [ ] add option for different architectures
# - [ ] add hyperparamters as parser args that output to a log
# - [ ] implement tensorboard

"""Code to train GNNs on the graph data!"""

import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer
from torch_geometric.explain import GNNExplainer
from torch_geometric.loader import NeighborSampler
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch


# Define/Instantiate GNN model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_size, embedding_size, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_size, embedding_size)  # GCNConv, SAGEConv
        self.conv1.aggr = "sum"
        self.conv2 = SAGEConv(embedding_size, embedding_size)
        self.conv2.aggr = "mean"
        self.conv3 = SAGEConv(embedding_size, embedding_size)
        self.conv3.aggr = "max"
        self.conv4 = SAGEConv(embedding_size, embedding_size)
        self.conv4.aggr = "sum"
        self.conv5 = SAGEConv(embedding_size, embedding_size)
        self.conv5.aggr = "mean"
        self.conv6 = SAGEConv(embedding_size, embedding_size)
        self.conv6.aggr = "max"
        self.conv7 = SAGEConv(embedding_size, embedding_size)
        self.conv7.aggr = "sum"
        self.conv8 = SAGEConv(embedding_size, embedding_size)
        self.conv8.aggr = "mean"
        self.conv9 = SAGEConv(embedding_size, embedding_size)
        self.conv9.aggr = "max"
        self.conv10 = SAGEConv(embedding_size, embedding_size)
        self.conv10.aggr = "sum"
        self.conv11 = SAGEConv(embedding_size, embedding_size)
        self.conv11.aggr = "max"
        self.con12 = SAGEConv(embedding_size, embedding_size)
        self.con12.aggr = "mean"
        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x = self.conv7(x, edge_index)
        x = F.relu(x)
        x = self.conv8(x, edge_index)
        x = F.relu(x)
        x = self.conv9(x, edge_index)
        x = F.relu(x)
        x = self.conv10(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv11(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.con12(x, edge_index)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


def train(model, device, optimizer, train_loader, epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:04d}")

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(
            out[data.train_mask].squeeze(), data.y[data.train_mask].squeeze()
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(model, device, test_loader, epoch):
    model.eval()

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

    train_acc = F.mse_loss(
        out[data.train_mask].squeeze(), data.y[data.train_mask].squeeze()
    )
    val_acc = F.mse_loss(out[data.val_mask].squeeze(), data.y[data.val_mask].squeeze())
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
        "--graph_type",
        type=str,
        default="full",
    )
    args = parser.parse_args()

    data = graph_to_pytorch(
        root_dir=args.root,
        graph_type=args.graph_type,
    )

    train_loader = RandomNodeLoader(
        data,
        num_parts=50,
        shuffle=True,
        num_workers=5,
    )

    test_loader = RandomNodeLoader(
        data,
        num_parts=50,
        num_workers=5,
    )

    ### check for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    model = GraphSAGE(in_size=data.x.shape[1], embedding_size=600, out_channels=4).to(
        device
    )
    # model = GCN(in_size=data.x.shape[1], embedding_size=512, out_channels=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # explainer = Explainer(
    #     model=model,
    #     alorigthm=GNNExplainer(epochs=200),
    #     explanation_type="model",
    #     node_mask_type="attributes",
    #     edge_mask_type="object",
    #     model_config=dict(
    #         mode="regression", task_level="node", return_type="log_probs"
    #     ),
    # )

    epochs = 200
    for epoch in range(0, epochs):
        loss = train(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
        )
        train_acc, val_acc, test_acc = test(
            model=model,
            device=device,
            test_loader=test_loader,
            epoch=epoch,
        )
        print(
            f"Epoch: {epoch:03d}, Loss: {loss}, Train: {train_acc:.4f}, Validation: {val_acc:.4f}, Test: {test_acc:.4f}"
        )
        # print(f"Epoch: {epoch:03d}, Loss: {loss}")

    torch.save(model, f"GraphSage_layers_{loss}.pt")


if __name__ == "__main__":
    main()


# class GCN(torch.nn.Module):
#     def __init__(self, in_size, embedding_size, out_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(in_size, embedding_size)  # GCNConv, SAGEConv
#         self.conv1.aggr = "sum"
#         self.conv2 = GCNConv(embedding_size, embedding_size)
#         self.conv2.aggr = "mean"
#         self.conv3 = GCNConv(embedding_size, embedding_size)
#         self.conv3.aggr = "max"
#         self.conv4 = GCNConv(embedding_size, embedding_size)
#         self.conv4.aggr = "sum"
#         self.conv5 = GCNConv(embedding_size, embedding_size)
#         self.conv5.aggr = "mean"
#         self.conv6 = GCNConv(embedding_size, embedding_size)
#         self.conv6.aggr = "max"
#         self.conv7 = GCNConv(embedding_size, embedding_size)
#         self.conv7.aggr = "sum"
#         self.conv8 = GCNConv(embedding_size, embedding_size)
#         self.conv8.aggr = "mean"
#         self.conv9 = GCNConv(embedding_size, embedding_size)
#         self.conv9.aggr = "max"
#         self.conv10 = GCNConv(embedding_size, embedding_size)
#         self.conv10.aggr = "mean"
#         self.lin1 = nn.Linear(embedding_size, 512)
#         self.lin2 = nn.Linear(512, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index)
#         x = F.relu(x)
#         x = self.conv4(x, edge_index)
#         x = F.relu(x)
#         x = self.conv5(x, edge_index)
#         x = F.relu(x)
#         x = self.conv6(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv7(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv8(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv9(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.1, training=self.training)
#         x = self.conv10(x, edge_index)
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = self.lin2(x)
#         return x
