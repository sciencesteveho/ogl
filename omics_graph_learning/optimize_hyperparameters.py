# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to handle automated hyperparameter tuning via Optuna. We use the
resource conscious Hyperband pruner."""


import argparse
from typing import Any, Dict, Tuple

import optuna
from optuna.trial import TrialState
import plotly  # type: ignore
import torch

from config_handlers import ExperimentConfig
from gnn_architecture_builder import build_gnn_architecture
from graph_to_pytorch import GraphToPytorch
from train_gnn import _set_optimizer
from train_gnn import _set_scheduler
from train_gnn import build_gnn_architecture
from train_gnn import calculate_training_steps
from train_gnn import prep_loader
from train_gnn import test
from train_gnn import train
from utils import dir_check_make
from utils import setup_logging

# helpers
EPOCHS = 200
MIN_RESOURCE = 3
REDUCTION_FACTOR = 3
N_TRIALS = 200
RANDOM_SEED = 42


def setup_device() -> torch.device:
    """Check for GPU and set device accordingly."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        return torch.device("cuda:0")
    return torch.device("cpu")


def suggest_hyperparameters(
    trial: optuna.Trial,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Suggest hyperparameters for the Optuna trial.

    Hyperparameters are based off a mix of ancedotal results and from Tönshoff
    et al. (2023) - "Where Did the Gap Go? Reassessing the Long-Range Graph
    Benchmark."
    """
    model = trial.suggest_categorical(
        "model", ["GCN", "GraphSAGE", "PNA", "GAT", "UniMPTransformer", "DeeperGCN"]
    )

    model_params = {
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
    }

    train_params = {
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
        model_params["heads"] = trial.suggest_int("heads", low=1, high=4, step=1)

    # set convolutional layers
    if model == "DeeperGCN":
        model_params["gnn_layers"] = trial.suggest_int(
            "gnn_layers", low=6, high=32, step=2
        )
    else:
        model_params["gnn_layers"] = trial.suggest_int(
            "gnn_layers", low=1, high=8, step=1
        )

    # add task specific mlp if not DeeperGCN
    if model != "DeeperGCN":
        model_params["task_specific_mlp"] = trial.suggest_categorical(
            "task_specific_mlp", [True, False]
        )
        model_params["skip_connection"] = trial.suggest_categorical(
            "skip_connection", ["shared_source", "distinct_source", None]
        )

    # set positional encodings
    model_params["positional_encoding"] = trial.suggest_categorical(
        "positional_encoding", [True, False]
    )

    return model_params, train_params


def objective(
    trial: optuna.Trial, experiment_config: ExperimentConfig, args: argparse.Namespace
) -> torch.Tensor:
    """Objective function to be optimized by Optuna."""
    # set gpu
    device = setup_device()

    # get trial hyperparameters
    model_params, train_params = suggest_hyperparameters(trial)
    gnn_layers = model_params["gnn_layers"]
    avg_connectivity = train_params["avg_connectivity"]
    batch_size = train_params["batch_size"]
    learning_rate = train_params["learning_rate"]
    optimizer_type = train_params["optimizer_type"]
    scheduler_type = train_params["scheduler_type"]

    # load graph data
    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=args.split_name,
        regression_target=args.target,
        positional_encoding=model_params["positional_encoding"],
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

    # define and build model
    model = build_gnn_architecture(
        in_size=data.x.shape[1],
        out_channels=1,
        **model_params,
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
        model_params=model.parameters(),
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
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters with Optuna."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="expression_median_only",
        choices=[
            "expression_median_only",
            "expression_media_and_foldchange",
            "difference_from_average",
            "foldchange_from_average",
            "protein_targets",
            "rna_seq",
        ],
    )
    parser.add_argument(
        "--split_name",
        type=str,
    )
    args = parser.parse_args()
    experiment_config = ExperimentConfig.from_yaml(args.config)

    model_dir = (
        experiment_config.root_dir / "models" / experiment_config.experiment_name
    )
    optuna_dir = model_dir / "optuna"
    dir_check_make(optuna_dir)

    logger = setup_logging(
        optuna_dir / f"{experiment_config.experiment_name}_optuna.log"
    )

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
    logger.info("Study statistics:\n")
    logger.info(f"Number of finished trials: {len(study.trials)}\n")
    logger.info(f"Number of pruned trials: {len(pruned_trials)}\n")
    logger.info(f"Number of complete trials: {len(complete_trials)}\n")

    # explicitly print best trial
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"Value: {trial.value}")
    logger.info("Best params:")
    for key, value in study.best_params.items():
        logger.info(f"\t{key}: {value}")

    # save results
    df = study.trials_dataframe().drop(
        ["datetime_start", "datetime_complete", "duration"], axis=1
    )  # exclude datetime columns
    df = df.loc[df["state"] == "COMPLETE"]  # keep only results that did not prune
    df = df.drop("state", axis=1)  # exclude state column
    df = df.sort_values("value")  # sort based on accuracy
    df.to_csv(optuna_dir / "optuna_results.csv", index=False)

    # display results in a dataframe
    logger.info(f"\nOverall Results (ordered by accuracy):\n {df}")

    # find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(
        study, target=None
    )

    # display the most important hyperparameters
    logger.info("\nMost important hyperparameters:")
    for key, value in most_important_parameters.items():
        logger.info("  {}:{}{:.2f}%".format(key, (15 - len(key)) * " ", value * 100))

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
