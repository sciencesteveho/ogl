#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Add number of linear layers as a config param
# - [ ] Check number of optimal lin layers from gc-merge

"""GNN model architectures!"""

from typing import Any, Dict, Optional

import torch
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GPSConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.attention import PerformerAttention
import torch_geometric.transforms as T


# Define/Instantiate GNN model
class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
        num_layers,
        lin_layers=3,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(SAGEConv(in_size, embedding_size, aggr="sum"))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(embedding_size, embedding_size, aggr="sum"))
            self.batch_norms.append(BatchNorm(embedding_size))
            
        self.linears = self.create_linear_layers(embedding_size, out_channels, lin_layers)
        
    def create_linear_layers(
        self,
        in_size,
        out_size,
        num_layers,
    ):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_size, in_size))
        layers.append(nn.Linear(in_size, out_size))
        return nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        for i, linear in enumerate(self.linears):
            if i == len(self.linears) - 1:
                x = linear(
                    F.dropout(
                        x,
                        p=self.dropout_rate,
                        training=self.training
                    )
                )
            else:
                x = F.relu(
                    linear(
                        F.dropout(
                            x,
                            p=0.2,
                            training=self.training
                        )
                    )
                )
                
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
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GCNConv(in_size, embedding_size))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(embedding_size, embedding_size))
            self.batch_norms.append(BatchNorm(embedding_size))

        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.lin3 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
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
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GATv2Conv(in_size, embedding_size, heads))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(heads * embedding_size, embedding_size, heads))
            # self.batch_norms.append(BatchNorm(heads * embedding_size))
            self.batch_norms.append(GraphNorm(heads * embedding_size))

        self.lin1 = nn.Linear(heads * embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GPSTransformer(torch.nn.Module):
    def __init__(
        self,
        in_size,
        embedding_size,
        walk_length: int,
        channels: int,
        pe_dim: int,
        num_layers: int,
    ):
        super().__init__()

        self.node_emb = nn.Linear(in_size, embedding_size - pe_dim)
        self.pe_lin = nn.Linear(walk_length, pe_dim)
        self.pe_norm = nn.BatchNorm1d(walk_length)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            gcnconv = GCNConv(embedding_size, embedding_size)
            conv = GPSConv(channels, gcnconv, heads=4, attn_kwargs={"dropout": 0.5})
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

        self.redraw_projection = RedrawProjection(self.convs, None)

    def forward(self, x, pe, edge_index, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)

        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module
                for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


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