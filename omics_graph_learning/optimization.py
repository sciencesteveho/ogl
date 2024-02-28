#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# from torch_geometric.explain import Explainer
# from torch_geometric.explain import GNNExplainer

"""Code to train GNNs on the graph data!"""

import argparse
import logging
import math
import pathlib
from typing import Any, Dict, Iterator, List, Optional

import optuna
from optuna.trial import TrialState
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch
from models import DeeperGCN
from models import GATv2
from models import GCN
from models import GraphSAGE
from models import MLP
from models import PNA
from models import UniMPTransformer
import utils


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Objective function for the Optuna optimization."""


def main() -> None:
    """Main function to train GNN on graph data!"""
    # Parse training settings


if __name__ == "__main__":
    main()
