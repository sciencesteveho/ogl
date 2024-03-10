# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# from torch_geometric.explain import Explainer
# from torch_geometric.explain import GNNExplainer

"""Code to train GNNs on the graph data!"""

import argparse
import gc
import math
from typing import Any, Dict, Iterator, List, Optional

import optuna
from optuna.trial import TrialState
import plotly
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric
from torch_geometric.data import DataListLoader
from torch_geometric.data import DataLoader
from tqdm import tqdm

from gnn import create_model
from gnn import get_loader
from graph_to_pytorch import graph_to_pytorch

EPOCHS = 50
RANDOM_SEED = 42
ROOT_DIR = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_k562_fdr001_intersect_25kb/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_NAME = "tpm_1.0_samples_0.2_test_8-9_val_7-13_rna_seq"
TARGET = "rna_seq"


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LRScheduler:
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    model: torch.nn.Module,
    device: torch.cuda.device,
    optimizer,
    train_loader: torch_geometric.data.DataLoader,
    epoch: int,
    subset_batches: int = None,
):
    """Train GNN model on graph data"""
    model.train()
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:04d}")

    total_loss = total_examples = 0
    for batch_idx, data in enumerate(train_loader):
        # Break out of the loop if subset_batches is set and reached.
        if subset_batches and batch_idx >= subset_batches:
            break

        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index)

        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()
    return total_loss / total_examples


@torch.no_grad()
def test(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    epoch: int,
    mask: torch.Tensor,
    subset_batches: int = None,
):
    """Test GNN model on test set"""
    model.eval()
    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse = []
    for batch_idx, data in enumerate(data_loader):
        # Break out of the loop if subset_batches is set and reached.
        if subset_batches and batch_idx >= subset_batches:
            break

        data = data.to(device)
        out = model(data.x, data.edge_index)

        if mask == "val":
            idx_mask = data.val_mask
        elif mask == "test":
            idx_mask = data.test_mask
        mse.append(F.mse_loss(out[idx_mask], data.y[idx_mask]).cpu())
        loss = torch.stack(mse)
        pbar.update(1)

    pbar.close()
    return math.sqrt(float(loss.mean()))


def objective(trial: optuna.Trial) -> torch.Tensor:
    """Objective function, optimized by Optuna

    Hyperparameters are based off a mix of ancedotal results and from Tönshoff
    et al. (2023) - "Where Did the Gap Go? Reassessing the Long-Range Graph
    Benchmark"

    Hyperparameters to be optimized:
        gnn base model, number of gnn layers, number of linear layers,
        activation function, hidden dimensions, batch size, learning rate,
        optimizer type, dropout rate, whether to use residual skip connections,
        and number of attention heads if the model uses them
    """
    # define range of values for hyperparameter testing
    # model = trial.suggest_categorical(
    #     "model", ["GCN", "GAT", "GraphSAGE", "PNA", "DeeperGCN"]
    # )
    model = trial.suggest_categorical("model", ["GCN", "GAT", "GraphSAGE", "PNA"])
    gnn_layers = trial.suggest_int(
        name="gnn_layers",
        low=2,
        high=6,
        step=1,
    )
    linear_layers = trial.suggest_int(
        name="linear_layers",
        low=1,
        high=3,
        step=1,
    )
    activation = trial.suggest_categorical("activation", ["relu", "leakyrelu", "gelu"])
    learning_rate = trial.suggest_float(
        name="learning_rate", low=1e-5, high=1e-2, log=True
    )
    optimizer_type = trial.suggest_categorical("optimizer_type", ["Adam", "AdamW"])
    residual = trial.suggest_categorical("residual", [True, False])
    dropout = trial.suggest_float(name="dropout", low=0.0, high=0.5, step=0.1)
    heads = trial.suggest_int(name="heads", low=1, high=4, step=1)
    dimensions = trial.suggest_int("dimensions", low=32, high=1024, step=32)
    batch_size = trial.suggest_int("batch_size", low=16, high=512, step=16)
    # loader = trial.suggest_categorical("loader", ["neighbor", "data"])

    # get dataloaders
    # def _load_data(batch_size, loader, neighbors):
    def _load_data(batch_size):
        data = graph_to_pytorch(
            experiment_name="regulatory_only_k562_fdr001_intersect_25kb",
            graph_type="full",
            root_dir=ROOT_DIR,
            split_name=SPLIT_NAME,
            regression_target=TARGET,
        )

        # temporary - to check number of edges for randomization tests
        print(f"Number of edges: {data.num_edges}")

        # set up data loaders
        train_loader = get_loader(
            data=data,
            mask="train_mask",
            batch_size=batch_size,
            shuffle=True,
            layers=gnn_layers,
        )
        test_loader = get_loader(
            data=data, mask="test_mask", batch_size=batch_size, layers=gnn_layers
        )
        val_loader = get_loader(
            data=data, mask="val_mask", batch_size=batch_size, layers=gnn_layers
        )
        return train_loader, test_loader, val_loader
        # if loader == "data":
        #     train_dataset = Subset(data, data.train_mask)
        #     test_dataset = Subset(data, data.test_mask)
        #     val_dataset = Subset(data, data.val_mask)
        #     train_loader = DataLoader(
        #         train_dataset, batch_size=batch_size, shuffle=True
        #     )
        #     test_loader = DataLoader(test_dataset, batch_size=batch_size)
        #     val_loader = DataLoader(val_dataset, batch_size=batch_size)
        #     return train_loader, test_loader, val_loader

    train_loader, _, val_loader = _load_data(
        batch_size=batch_size,
    )

    # define model and get optimizer
    model = create_model(
        model=model,
        in_size=41,
        embedding_size=dimensions,
        out_channels=1,
        gnn_layers=gnn_layers,
        linear_layers=linear_layers,
        activation=activation,
        residual=residual,
        dropout_rate=dropout,
        heads=heads if model in ("GAT", "UniMPTransformer") else None,
        train_dataset=train_loader if model == "PNA" else None,
    )
    model = model.to(DEVICE)

    # set optimizer
    def _set_optimizer(optimizer_type, learning_rate, model_params):
        """Choose optimizer"""
        # set gradient descent optimizer
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                model_params,
                lr=learning_rate,
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
            )
        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                model_params,
                lr=learning_rate,
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=5,
                num_training_steps=EPOCHS,
            )
        return optimizer, scheduler

    optimizer, scheduler = _set_optimizer(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        model_params=model.parameters(),
    )

    for epoch in range(EPOCHS):
        # train
        _ = train(
            model=model,
            device=DEVICE,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
            # subset_batches=750,
        )
        # validation
        mse = test(
            model=model,
            device=DEVICE,
            data_loader=val_loader,
            epoch=epoch,
            mask="val",
            # subset_batches=225,
        )
        scheduler.step(mse)

        # For pruning (stops trial early if not promising)
        trial.report(mse, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # clear memory to avoid OOM errors
    # del model
    # torch.cuda.empty_cache()
    # gc.collect()

    return mse


def main() -> None:
    """Main function to optimize hyperparameters w/ optuna!"""
    plot_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/optuna"
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")

    # Save results to csv file
    df = study.trials_dataframe().drop(
        ["datetime_start", "datetime_complete", "duration"], axis=1
    )  # Exclude columns
    df = df.loc[df["state"] == "COMPLETE"]  # Keep only results that did not prune
    df = df.drop("state", axis=1)  # Exclude state column
    df = df.sort_values("value")  # Sort based on accuracy
    df.to_csv(f"{plot_dir}/optuna_results.csv", index=False)  # Save to csv file

    # Display results in a dataframe
    print(f"\nOverall Results (ordered by accuracy):\n {df}")

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(
        study, target=None
    )

    # Display the most important hyperparameters
    print("\nMost important hyperparameters:")
    for key, value in most_important_parameters.items():
        print("  {}:{}{:.2f}%".format(key, (15 - len(key)) * " ", value * 100))

    # Plot and save importances to file
    optuna.visualization.plot_optimization_history(study).write_image(
        f"{plot_dir}/history.png"
    )
    optuna.visualization.plot_param_importances(study=study).write_image(
        f"{plot_dir}/importances.png"
    )
    optuna.visualization.plot_slice(study=study).write_image(f"{plot_dir}/slice.png")


if __name__ == "__main__":
    main()
