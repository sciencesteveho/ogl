#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# from torch_geometric.explain import Explainer
# from torch_geometric.explain import GNNExplainer

"""Code to train GNNs on the graph data!"""

import argparse
import gc
from typing import Any, Dict, Iterator, List, Optional

import optuna
from optuna.trial import TrialState
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gnn import _create_model
from gnn import _set_optimizer
from gnn import get_cosine_schedule_with_warmup
from gnn import test
from gnn import train
from graph_to_pytorch import graph_to_pytorch
import utils


def _get_dataloaders():
    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type=args.graph_type,
        root_dir=root_dir,
        split_name=args.split_name,
        regression_target=args.target,
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
    )


def objective(trial: optuna.Trial, args: argparse.Namespace) -> torch.Tensor:
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
    model = trial.suggest_categorical(
        "model", ["GCN", "GATv2", "GraphSAGE", "PNA", "DeeperGCN", "UniMPTransformer"]
    )
    gnn_layers = trial.suggest_int("gnn_layers", 2, 3, 6, 8, 10)
    linear_layers = trial.suggest_int("linear_layers", 1, 2, 3)
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
    dimensions = trial.suggest_int("dimensions", 64, 128, 256, 512)
    batch_size = trial.suggest_int("batch_size", 64, 128, 256, 512)
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-3, 5e-4, 1e-4, 1e-5, log=True
    )
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    dropout = trial.suggest_float("dropout", 0, 0.1, 0.2)
    residual = trial.suggest_categorical("residual", [True, False])
    heads = trial.suggest_int("heads", 1, 2, 3)

    # get dataloader

    # define model and get optimizer
    model = _create_model(
        model=model,
        in_size=1,
        embedding_size=dimensions,
        out_channels=1,
        gnn_layers=gnn_layers,
        linear_layers=linear_layers,
        activation=activation,
        residual=residual,
        dropout_rate=dropout,
        heads=heads if model in ("GATv2", "UniMPTransformer") else None,
        train_dataset=train_dataset if model == "PNA" else None,
    )

    # set optimizer
    optimizer = _set_optimizer(optimizer, learning_rate)

    for epoch in range(args.epochs):
        train(model, optimizer)  # Train the model
        mse = test(model)  # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(mse, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # clear memory to avoid OOM errors
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return mse


def main() -> None:
    """Main function to optimize hyperparameters w/ optuna!"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 10
    main_train_examples = 500 * batch_size
    random_seed = 1

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save results to csv file
    df = study.trials_dataframe().drop(
        ["datetime_start", "datetime_complete", "duration"], axis=1
    )  # Exclude columns
    df = df.loc[df["state"] == "COMPLETE"]  # Keep only results that did not prune
    df = df.drop("state", axis=1)  # Exclude state column
    df = df.sort_values("value")  # Sort based on accuracy
    df.to_csv("optuna_results.csv", index=False)  # Save to csv file

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


if __name__ == "__main__":
    main()
