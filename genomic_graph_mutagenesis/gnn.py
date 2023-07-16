#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] implement tensorboard
# - [ ] add dropout as arg

"""Code to train GNNs on the graph data!"""

import argparse
from datetime import datetime
import logging
import math
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.explain import Explainer
from torch_geometric.explain import GNNExplainer
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv

# from torchmetrics.regression import SpearmanCorrCoef
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch
from utils import dir_check_make


# Define/Instantiate GNN model
class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
        num_layers,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_size, embedding_size, aggr="sum"))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(embedding_size, embedding_size, aggr="sum"))
            self.batch_norms.append(BatchNorm(embedding_size))

        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)
        # self.lin2 = nn.Linear(embedding_size, embedding_size)
        # self.lin3 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = self.lin3(x)
        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
        num_layers,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_size, embedding_size))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(embedding_size, embedding_size))
            self.batch_norms.append(BatchNorm(embedding_size))

        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


class GATv2(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
        num_layers,
        heads,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GATv2Conv(in_size, embedding_size, heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(heads * embedding_size, embedding_size, heads))
            self.batch_norms.append(BatchNorm(heads * embedding_size))

        self.lin1 = nn.Linear(heads * embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return x


### baseline MLP
class MLP(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
    ):
        super().__init__()

        self.lin1 = nn.Linear(in_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.lin3 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
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

        # calculate loss
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()
    return total_loss / total_examples


@torch.no_grad()
def test(model, device, data_loader, epoch, mask):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse = []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        print(out)
        print(data.y)

        # calculate loss
        if mask == "val":
            idx_mask = data.val_mask
        if mask == "test":
            idx_mask = data.test_mask
        mse.append(F.mse_loss(out[idx_mask], data.y[idx_mask]).cpu())
        loss = torch.stack(mse)

        pbar.update(1)

    pbar.close()
    return math.sqrt(float(loss.mean()))


@torch.no_grad()
def test_with_idxs(model, device, data_loader, epoch, mask):
    # spearman = SpearmanCorrCoef(num_outputs=2)
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse = []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # calculate loss
        if mask == "val":
            idx_mask = data.val_mask
        if mask == "test":
            idx_mask = data.test_mask
        mse.append(F.mse_loss(out[idx_mask], data.y[idx_mask]).cpu())
        # outs.extend(out[idx_mask])
        # labels.extend(data.y[idx_mask])
        loss = torch.stack(mse)

        pbar.update(1)

    pbar.close()
    # print(spearman(torch.stack(outs), torch.stack(labels)))
    return math.sqrt(float(loss.mean()))


def main() -> None:
    """_summary_"""
    # Parse training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default="2",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default="600",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed to use (default: 42)",
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="neighbor",
        help="'neighbor' or 'random' node loader (default: 'random')",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--idx",
        type=bool,
        default=False,
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

    # make directories and set up training logs

    if args.idx:
        savestr = f"{args.model}_{args.layers}_{args.dimensions}_{args.lr}_batch{args.batch}_{args.loader}_{args.graph_type}_targetnoscale_idx"
    else:
        savestr = f"{args.model}_{args.layers}_{args.dimensions}_{args.lr}_batch{args.batch}_{args.loader}_{args.graph_type}"
    logging.basicConfig(
        filename=f"{args.root}/models/logs/{savestr}.log",
        level=logging.DEBUG,
    )
    dir_check_make("models/logs")
    dir_check_make(f"models/{savestr}")

    # check for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    # prepare data
    data = graph_to_pytorch(
        root_dir=args.root,
        graph_type=args.graph_type,
        only_expression_no_fold=True,
    )

    # data loaders
    if args.loader == "random":
        train_loader = RandomNodeLoader(
            data,
            num_parts=250,
            shuffle=True,
            num_workers=5,
        )
        test_loader = RandomNodeLoader(
            data,
            num_parts=250,
            num_workers=5,
        )

    if args.loader == "neighbor":
        train_loader = NeighborLoader(
            data,
            num_neighbors=[20, 15, 10],
            batch_size=args.batch,
            shuffle=True,
        )
        test_loader = NeighborLoader(
            data,
            num_neighbors=[20, 15, 10],
            batch_size=args.batch,
        )
        if args.idx:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=args.batch,
                input_nodes=data.train_mask,
                shuffle=True,
            )
            test_loader = NeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=args.batch,
                input_nodes=data.test_mask,
            )
            val_loader = NeighborLoader(
                data,
                num_neighbors=[5, 5, 5, 5, 5, 3],
                batch_size=args.batch,
                input_nodes=data.val_mask,
            )

    # CHOOSE YOUR WEAPON
    if args.model == "GraphSAGE":
        model = GraphSAGE(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
            num_layers=args.layers,
        ).to(device)
    if args.model == "GCN":
        model = GCN(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
            num_layers=args.layers,
        ).to(device)
    if args.model == "GAT":
        model = GATv2(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
            num_layers=args.layers,
            heads=2,
        ).to(device)
    if args.model == "MLP":
        model = MLP(
            in_size=data.x.shape[1],
            embedding_size=args.dimensions,
            out_channels=1,
        ).to(device)

    # set gradient descent optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = 100
    best_validation = stop_counter = 0
    for epoch in range(0, epochs + 1):
        loss = train(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
        )
        print(f"Epoch: {epoch:03d}, Train: {loss}")
        logging.info(f"Epoch: {epoch:03d}, Train: {loss}")

        if args.idx:
            val_acc = test_with_idxs(
                model=model,
                device=device,
                data_loader=val_loader,
                epoch=epoch,
                mask="val",
            )

            test_acc = test_with_idxs(
                model=model,
                device=device,
                data_loader=test_loader,
                epoch=epoch,
                mask="test",
            )
        else:
            val_acc = test(
                model=model,
                device=device,
                data_loader=test_loader,
                epoch=epoch,
                mask="val",
            )

            test_acc = test(
                model=model,
                device=device,
                data_loader=test_loader,
                epoch=epoch,
                mask="test",
            )

        if epoch == 0:
            best_validation = val_acc
        else:
            if val_acc < best_validation:
                stop_counter = 0
                best_validation = val_acc
                torch.save(
                    model.state_dict(),
                    f"models/{savestr}/{savestr}_early_epoch_{epoch}_mse_{best_validation}.pt",
                )
            if best_validation < val_acc:
                stop_counter += 1

        print(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}")
        logging.info(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}")

        print(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        logging.info(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        if stop_counter == 10:
            print("***********Early stopping!")
            break

    torch.save(
        model.state_dict(),
        f"models/{savestr}/{savestr}_mse_{best_validation}.pt",
    )

    # # GNN Explainer!
    # if args.model != "MLP":
    #     with open(
    #         "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/explainer_node_ids.pkl",
    #         "rb",
    #     ) as file:
    #         ids = pickle.load(file)
    #     explain_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/explainer"
    #     explainer = Explainer(
    #         model=model,
    #         algorithm=GNNExplainer(epochs=200),
    #         explanation_type="model",
    #         node_mask_type="attributes",
    #         edge_mask_type="object",
    #         model_config=dict(mode="regression", task_level="node", return_type="raw"),
    #     )

    #     data = data.to(device)
    #     for index in random.sample(ids, 5):
    #         explanation = explainer(data.x, data.edge_index, index=index)

    #         print(f"Generated explanations in {explanation.available_explanations}")

    #         path = f"{explain_path}/feature_importance_{savestr}_{best_validation}.png"
    #         explanation.visualize_feature_importance(path, top_k=10)
    #         print(f"Feature importance plot has been saved to '{path}'")

    #         path = f"{explain_path}/subgraph_{savestr}_{best_validation}.pdf"
    #         explanation.visualize_graph(path)
    #         print(f"Subgraph visualization plot has been saved to '{path}'")


if __name__ == "__main__":
    main()
