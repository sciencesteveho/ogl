# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to handle automated hyperparameter tuning via Optuna. We use the
resource conscious Hyperband pruner."""


import argparse
from typing import Any, Dict

import optuna
from optuna.trial import TrialState
import plotly  # type: ignore
import torch

from config_handlers import ExperimentConfig
from gnn import _set_optimizer
from gnn import _set_scheduler
from gnn import build_gnn_architecture
from gnn import calculate_training_steps
from gnn import prep_loader
from gnn import setup_device
from gnn import test
from gnn import train
from graph_to_pytorch import GraphToPytorch
from ogl import parse_arguments
from utils import dir_check_make

# constant helpers
EPOCHS = 200
MIN_RESOURCE = 3
REDUCTION_FACTOR = 3
N_TRIALS = 200


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for the Optuna trial.

    Hyperparameters are based off a mix of ancedotal results and from Tönshoff
    et al. (2023) - "Where Did the Gap Go? Reassessing the Long-Range Graph
    Benchmark."
    """
    model = trial.suggest_categorical(
        "model", ["GCN", "GraphSAGE", "PNA", "GAT", "UniMPTransformer", "DeeperGCN"]
    )

    params = {
        "model": model,
        "activation": trial.suggest_categorical(
            "activation", ["relu", "leakyrelu", "gelu"]
        ),
        "embedding_size": trial.suggest_int(
            "embedding_size", low=32, high=1024, step=32
        ),
        "linear_layers": trial.suggest_int("linear_layers", low=1, high=4, step=1),
        "dropout_rate": trial.suggest_float(
            "dropout_rate", low=0.0, high=0.5, step=0.1
        ),
        "skip_connection": trial.suggest_categorical(
            "skip_connection", ["shared_source", "distinct_source", None]
        ),
        "task_specific_mlp": trial.suggest_categorical(
            "task_specific_mlp", [True, False]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", low=1e-6, high=1e-2, log=True
        ),
        "optimizer_type": trial.suggest_categorical(
            "optimizer_type", ["Adam", "AdamW"]
        ),
        "scheduler_type": trial.suggest_categorical(
            "scheduler_type", ["plateau", "linear_warmup", "cosine"]
        ),
        "batch_size": trial.suggest_int("batch_size", low=16, high=512, step=16),
        "avg_connectivity": trial.suggest_categorical(
            "avg_connectivity", [True, False]
        ),
    }

    # set heads for attention-based models
    if model in ["GAT", "UniMPTransformer"]:
        params["heads"] = trial.suggest_int("heads", low=1, high=4, step=1)

    # set convolutional layers
    if model == "DeeperGCN":
        params["gnn_layers"] = trial.suggest_int("gnn_layers", low=6, high=32, step=2)
    else:
        params["gnn_layers"] = trial.suggest_int("gnn_layers", low=1, high=8, step=1)

    # set positional encodings
    params["positional_encoding"] = trial.suggest_categorical(
        "positional_encoding", [True, False]
    )

    return params


def objective(
    trial: optuna.Trial, experiment_config: ExperimentConfig, args: argparse.Namespace
) -> torch.Tensor:
    """Objective function to be optimized by Optuna."""
    # set gpu
    device = setup_device(args)

    # get trial hyperparameters
    params = suggest_hyperparameters(trial)

    # Use the suggested parameters
    model = params["model"]
    heads = params.get("heads")  # Only exists for GAT and UniMPTransformer
    activation = params["activation"]
    embedding_size = params["embedding_size"]
    gnn_layers = params["gnn_layers"]
    linear_layers = params["linear_layers"]
    dropout_rate = params["dropout_rate"]
    skip_connection = params["skip_connection"]
    task_specific_mlp = params["task_specific_mlp"]
    learning_rate = params["learning_rate"]
    optimizer_type = params["optimizer_type"]
    scheduler_type = params["scheduler_type"]
    batch_size = params["batch_size"]
    avg_connectivity = params["avg_connectivity"]
    positional_encoding = params["positional_encoding"]

    # load graph data
    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=args.split_name,
        regression_target=args.target,
        positional_encoding=positional_encoding,
    ).make_data_object()

    # set up train, test, and validation loaders
    train_loader = prep_loader(
        data=data,
        mask="train_mask",
        batch_size=batch_size,
        shuffle=True,
        layers=gnn_layers,
        avg_connectivity=avg_connectivity,
    )

    val_loader = prep_loader(
        data=data,
        mask="val_mask",
        batch_size=batch_size,
        layers=gnn_layers,
        avg_connectivity=avg_connectivity,
    )

    # define model and get optimizer
    model = build_gnn_architecture(
        model=model,
        activation=activation,
        in_size=data.x.shape[1],
        embedding_size=embedding_size,
        out_channels=1,
        gnn_layers=gnn_layers,
        shared_mlp_layers=linear_layers,
        heads=heads,
        dropout_rate=dropout_rate,
        skip_connection=skip_connection,
        task_specific_mlp=task_specific_mlp,
        train_dataset=train_loader.dataset,
    )
    model = model.to(device)

    # set up optimizer
    total_steps, warmup_steps = calculate_training_steps(
        train_loader=train_loader,
        batch_size=batch_size,
        epochs=100,
    )

    optimizer = _set_optimizer(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        model_parameters=model.parameters(),
    )
    scheduler = _set_scheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        training_steps=total_steps,
    )

    for epoch in range(EPOCHS + 1):
        # train
        _ = train(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
            # subset_batches=750,
        )

        # validation
        mse = test(
            model=model,
            device=device,
            data_loader=val_loader,
            epoch=epoch,
            mask="val",
            # subset_batches=225,
        )
        scheduler.step(mse)

        # report for pruning
        trial.report(mse, epoch)

        # handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return mse


def main() -> None:
    """Main function to optimize hyperparameters w/ optuna!"""
    args = parse_arguments()
    experiment_config = ExperimentConfig.from_yaml(args.config)

    model_dir = (
        experiment_config.root_dir / "models" / experiment_config.experiment_name
    )
    optuna_dir = model_dir / "optuna"
    dir_check_make(optuna_dir)

    # create a study object with Hyperband Pruner
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=MIN_RESOURCE,
            max_resource=EPOCHS,
            reduction_factor=REDUCTION_FACTOR,
        ),
    )
    study.optimize(
        lambda trial: objective(trial, experiment_config, args),
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # display results
    print("Study statistics:\n")
    print(f"Number of finished trials: {len(study.trials)}\n")
    print(f"Number of pruned trials: {len(pruned_trials)}\n")
    print(f"Number of complete trials: {len(complete_trials)}\n")

    # explicitly print best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"\t{key}: {value}")

    # save results
    df = study.trials_dataframe().drop(
        ["datetime_start", "datetime_complete", "duration"], axis=1
    )  # exclude datetime columns
    df = df.loc[df["state"] == "COMPLETE"]  # keep only results that did not prune
    df = df.drop("state", axis=1)  # exclude state column
    df = df.sort_values("value")  # sort based on accuracy
    df.to_csv(optuna_dir / "optuna_results.csv", index=False)

    # display results in a dataframe
    print(f"\nOverall Results (ordered by accuracy):\n {df}")

    # find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(
        study, target=None
    )

    # display the most important hyperparameters
    print("\nMost important hyperparameters:")
    for key, value in most_important_parameters.items():
        print("  {}:{}{:.2f}%".format(key, (15 - len(key)) * " ", value * 100))

    # plot and save importances to file
    optuna.visualization.plot_optimization_history(study).write_image(
        f"{optuna_dir}/history.png"
    )
    optuna.visualization.plot_param_importances(study=study).write_image(
        f"{optuna_dir}/importances.png"
    )
    optuna.visualization.plot_slice(study=study).write_image(f"{optuna_dir}/slice.png")


if __name__ == "__main__":
    main()
