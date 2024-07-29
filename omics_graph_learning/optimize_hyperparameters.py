# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to handle automated hyperparameter tuning via Optuna's Hyperband
pruner."""


import argparse
import gc
import math
from typing import Any, Dict, Iterator, List, Optional

import optuna
from optuna.trial import TrialState
import plotly  # type: ignore
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric  # type: ignore
from torch_geometric.data import DataListLoader  # type: ignore
from torch_geometric.data import DataLoader
from tqdm import tqdm  # type: ignore

from config_handlers import ExperimentConfig
from gnn import create_model
from gnn import prep_loader
from graph_to_pytorch import graph_to_pytorch
from ogl import parse_pipeline_arguments

EPOCHS = 200
RANDOM_SEED = 42


def train(
    model: torch.nn.Module,
    device: torch.cuda.device,
    optimizer: Optimizer,
    train_loader: torch_geometric.data.DataLoader,
    epoch: int,
    subset_batches: int = None,
):
    """Train GNN model on graph data including an option to subset batches if
    necessary."""
    model.train()
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:04d}")

    total_loss = total_examples = 0
    for batch_idx, data in enumerate(train_loader):
        # break out of the loop if subset_batches is set and reached.
        if subset_batches and batch_idx >= subset_batches:
            break

        optimizer.zero_grad()
        data = data.to(device)

        if model.task_specific_mlp:
            out = model(data.x, data.edge_index, data.train_mask_loss)
        else:
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


def objective(
    trial: optuna.Trial, experiment_config: ExperimentConfig, args: argparse.Namespace
) -> torch.Tensor:
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
    # model = trial.suggest_categorical("model", ["GAT", "GraphSAGE", "PNA"])
    gnn_layers = trial.suggest_int(
        name="gnn_layers",
        low=2,
        high=8,
        step=1,
    )
    linear_layers = trial.suggest_int(
        name="linear_layers",
        low=1,
        high=4,
        step=1,
    )
    activation = trial.suggest_categorical("activation", ["relu", "leakyrelu", "gelu"])
    learning_rate = trial.suggest_float(
        name="learning_rate", low=1e-6, high=1e-2, log=True
    )
    optimizer_type = trial.suggest_categorical("optimizer_type", ["Adam", "AdamW"])
    residual = trial.suggest_categorical("residual", [True, False])
    dropout = trial.suggest_float(name="dropout", low=0.0, high=0.5, step=0.1)
    heads = trial.suggest_int(name="heads", low=1, high=4, step=1)
    dimensions = trial.suggest_int("dimensions", low=32, high=1024, step=32)
    batch_size = trial.suggest_int("batch_size", low=16, high=512, step=16)
    avg_connectivity = trial.suggest_categorical("avg_connectivity", [True, False])
    # loader = trial.suggest_categorical("loader", ["neighbor", "data"])

    # get dataloaders
    # def _load_data(batch_size, loader, neighbors):
    def _load_data(batch_size):
        data = graph_to_pytorch(
            experiment_config=experiment_config,
            graph_type=args.graph_type,
            split_name=args.split_name,
            regression_target=args.target,
        )

        # temporary - to check number of edges for randomization tests
        print(f"Number of edges: {data.num_edges}")

        # set up data loaders
        train_loader = prep_loader(
            data=data,
            mask="train_mask",
            batch_size=batch_size,
            shuffle=True,
            layers=gnn_layers,
            avg_connectivity=avg_connectivity,
        )
        test_loader = prep_loader(
            data=data, mask="test_mask", batch_size=batch_size, layers=gnn_layers
        )
        val_loader = prep_loader(
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
        model="GAT",
        in_size=41,
        embedding_size=dimensions,
        out_channels=1,
        gnn_layers=gnn_layers,
        linear_layers=linear_layers,
        activation=activation,
        residual=residual,
        dropout_rate=dropout,
        # heads=heads if model in ("GAT", "UniMPTransformer") else None,
        heads=heads,
        train_dataset=None,
        # train_dataset=train_loader if model == "PNA" else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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
    args = parse_pipeline_arguments()
    prefix = f"{args.model}_"
    plot_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/optuna"

    # Create a study object with Hyperband Pruner
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3, max_resource=EPOCHS, reduction_factor=10
        ),
    )
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
    df.to_csv(
        f"{plot_dir}/optuna_results_{prefix}.csv", index=False
    )  # Save to csv file

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
        f"{plot_dir}/history_{prefix}.png"
    )
    optuna.visualization.plot_param_importances(study=study).write_image(
        f"{plot_dir}/importances_{prefix}.png"
    )
    optuna.visualization.plot_slice(study=study).write_image(
        f"{plot_dir}/slice_{prefix}.png"
    )


if __name__ == "__main__":
    main()
