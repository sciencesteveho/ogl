# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to handle automated hyperparameter tuning via Optuna. We use the
resource conscious Hyperband pruner."""


import argparse
import logging
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Dict, Tuple, Union

import filelock
import numpy as np
import optuna
from optuna.trial import TrialState
import plotly  # type: ignore
from scipy.stats import pearsonr  # type: ignore
from sqlalchemy.exc import OperationalError
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LRScheduler
import torch_geometric  # type: ignore

from config_handlers import ExperimentConfig
from gnn_architecture_builder import build_gnn_architecture
from graph_to_pytorch import GraphToPytorch
from schedulers import OptimizerSchedulerHandler
from train_gnn import build_gnn_architecture
from train_gnn import GNNTrainer
from train_gnn import prep_loader
from utils import _tensor_out_to_array
from utils import dir_check_make
from utils import PyGDataChecker
from utils import setup_logging

# helpers for trial
EPOCHS = 30
MIN_RESOURCE = 3
REDUCTION_FACTOR = 3
N_TRIALS = 100
RANDOM_SEED = 42
MAX_RETRIES = 5
RETRY_DELAY = 1


def get_remaining_walltime(logger: logging.Logger) -> Union[int, None]:
    """Get remaining walltime in seconds."""
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", "$SLURM_JOB_ID"], capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "TimeLeft=" in line:
                time_left = line.split("TimeLeft=")[1].split()[0]
                days, time = (
                    time_left.split("-") if "-" in time_left else ("0", time_left)
                )
                hours, mins, secs = time.split(":")
                return (
                    int(days) * 86400 + int(hours) * 3600 + int(mins) * 60 + int(secs)
                )
    except Exception as e:
        logger.warning(f"Failed to get remaining walltime: {e}")
    return None


def setup_device(local_rank: int = 0) -> torch.device:
    """Set up device based on available resources."""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return torch.device(f"cuda:{local_rank}")
        else:
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
        "shared_mlp_layers": trial.suggest_int(
            "shared_mlp_layers", low=1, high=3, step=1
        ),
        "dropout_rate": trial.suggest_float(
            "dropout_rate", low=0.0, high=0.5, step=0.1
        ),
        "attention_task_head": trial.suggest_categorical(
            "attention_task_head",
            [True, False],
        ),
        "positional_encoding": trial.suggest_categorical(
            "positional_encoding", [True, False]
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
        "batch_size": trial.suggest_int("batch_size", low=32, high=544, step=64),
        "avg_connectivity": trial.suggest_categorical(
            "avg_connectivity", [True, False]
        ),
    }

    # set heads and embedding size for attention-based models
    if model in ["GAT", "UniMPTransformer"]:
        model_params["heads"] = trial.suggest_int("heads", low=1, high=4, step=1)
        model_params["embedding_size"] = trial.suggest_int(
            "embedding_size", low=32, high=320, step=32
        )
    else:
        model_params["embedding_size"] = trial.suggest_int(
            "embedding_size", low=32, high=640, step=32
        )

    # set convolutional layers
    if model == "DeeperGCN":
        model_params["gnn_layers"] = trial.suggest_int(
            "gnn_layers", low=6, high=24, step=2
        )
    else:
        model_params["gnn_layers"] = trial.suggest_int(
            "gnn_layers", low=2, high=10, step=1
        )

    # no skip connections for DeeperGCN, as they are inherent to the layers
    if model != "DeeperGCN":
        model_params["residual"] = trial.suggest_categorical(
            "residual", ["shared_source", "distinct_source", None]
        )

    return model_params, train_params


def train_and_evaluate(
    trial: optuna.Trial,
    trainer: GNNTrainer,
    train_loader: torch_geometric.data.DataLoader,
    val_loader: torch_geometric.data.DataLoader,
    scheduler: LRScheduler,
    logger: logging.Logger,
) -> Tuple[float, float]:
    """Run training and evaluation."""
    patience = 5
    best_r = -float("inf")
    best_rmse = float("inf")
    early_stop_counter = 0

    for epoch in range(EPOCHS + 1):
        # train
        loss = trainer.train(train_loader=train_loader, epoch=epoch)
        logger.info(f"Epoch {epoch}, Loss: {loss}")

        if np.isnan(loss):
            logger.info(f"Trial {trial.number} pruned at epoch {epoch} due to NaN loss")
            raise optuna.exceptions.TrialPruned()

        # validation
        rmse, predictions, targets = trainer.evaluate(
            data_loader=val_loader,
            epoch=epoch,
            mask="val",
        )

        predictions = _tensor_out_to_array(predictions, 0)
        targets = _tensor_out_to_array(targets, 0)

        # calculate metrics on validation set
        r, p_val = pearsonr(predictions, targets)
        logger.info(f"Validation Pearson's R: {r}, p-value: {p_val}, RMSE: {rmse}")

        if np.isnan(rmse):
            logger.info(f"Trial {trial.number} pruned at epoch {epoch} due to NaN RMSE")
            raise optuna.exceptions.TrialPruned()

        # early stopping
        if r > best_r:
            best_r = r
            best_rmse = rmse
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # report for pruning
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(rmse)

        trial.report(r, epoch)

        # handle pruning based on the intermediate value.
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

    return best_r, best_rmse


def objective(
    trial: optuna.Trial,
    experiment_config: ExperimentConfig,
    args: argparse.Namespace,
    logger: logging.Logger,
    rank: int,
    device: torch.device,
) -> float:
    """Objective function to be optimized by Optuna."""
    try:
        # get trial hyperparameters
        model_params, train_params = suggest_hyperparameters(trial)
        gnn_layers = model_params["gnn_layers"]
        avg_connectivity = train_params["avg_connectivity"]
        batch_size = train_params["batch_size"]
        learning_rate = train_params["learning_rate"]
        optimizer_type = train_params["optimizer_type"]
        scheduler_type = train_params["scheduler_type"]

        # log trial hyperparameters
        logger.info("=" * 50)
        logger.info(f"Starting Trial {trial.number} with parameters:")
        logger.info(f"Model Params: {model_params}")
        logger.info(f"Train Params: {train_params}")
        logger.info("=" * 50)
        logger.handlers[0].flush()

        # load graph data
        data = GraphToPytorch(
            experiment_config=experiment_config,
            split_name=args.split_name,
            regression_target=args.target,
            positional_encoding=model_params["positional_encoding"],
        ).make_data_object()

        # check data integreity
        PyGDataChecker.check_pyg_data(data)

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
            train_dataset=train_loader,
            **model_params,
        )
        model = model.to(device)

        # set up optimizer
        total_steps, warmup_steps = OptimizerSchedulerHandler.calculate_training_steps(
            train_loader=train_loader,
            batch_size=batch_size,
            epochs=100,
        )

        optimizer = OptimizerSchedulerHandler.set_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            model_params=model.parameters(),
        )
        scheduler = OptimizerSchedulerHandler.set_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            training_steps=total_steps,
        )

        # initialize trainer
        trainer = GNNTrainer(
            model=model,
            device=device,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
        )

        # training loop
        _, best_rmse = train_and_evaluate(
            trial, trainer, train_loader, val_loader, scheduler, logger
        )

        trial.set_user_attr("best_rmse", best_rmse)  # save best rmse
        return best_rmse
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            trial.report(float("inf"), EPOCHS)
            logger.info(f"Trial {trial.number} pruned due to CUDA out of memory")
            return float("inf")
        raise e


def run_optimization(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    logger: logging.Logger,
    optuna_dir: Path,
) -> None:
    """Run the optimization process per GPU."""

    # initialize distributed training, if detected
    device = setup_device(rank)
    logger.info(f"Process {rank}/{world_size} starting optimization on device {device}")

    if world_size > 1:
        try:
            # set environment variables for distributed training
            os.environ["MASTER_ADDR"] = socket.gethostbyname(socket.gethostname())
            os.environ["MASTER_PORT"] = "29500"  # default port for PyTorch

            logger.info(
                f"Initializing process group with rank {rank}, world_size {world_size}"
            )
            logger.info(
                f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}"
            )

            # try NCCL first & fall back to GLOO if NCCL is not available
            try:
                dist.init_process_group("nccl", rank=rank, world_size=world_size)
                logger.info("Initialized process group with NCCL backend")
            except RuntimeError:
                logger.warning(
                    "NCCL initialization failed, falling back to GLOO backend"
                )
                dist.init_process_group("gloo", rank=rank, world_size=world_size)
                logger.info("Initialized process group with GLOO backend")
        except Exception as e:
            logger.error(f"Failed to initialize distributed process group: {str(e)}")
            raise

    experiment_config = ExperimentConfig.from_yaml(args.config)

    # create a study with Hyperband Pruner
    storage_url = f"sqlite:///{optuna_dir}/optuna_study.db"
    study_name = "distributed_optimization"
    lock_file = f"{optuna_dir}/optuna_study.lock"

    # use filelock to prevent simultaneous access to the database
    with filelock.FileLock(lock_file, timeout=60):
        for attempt in range(MAX_RETRIES):
            try:
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    load_if_exists=True,
                    direction="maximize",
                    pruner=optuna.pruners.HyperbandPruner(
                        min_resource=MIN_RESOURCE,
                        max_resource=EPOCHS,
                        reduction_factor=REDUCTION_FACTOR,
                    ),
                )
                logger.info(f"Created or loaded existing study: {study_name}")
                break
            except (OperationalError, KeyError) as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Error accessing study (attempt {attempt + 1}): {e}"
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(
                        f"Failed to create or load study after {MAX_RETRIES} attempts"
                    )
                    raise

    n_trials = N_TRIALS if world_size == 1 else N_TRIALS // world_size

    start_time = time.time()
    for trial_num in range(n_trials):
        remaining_time = get_remaining_walltime(logger)
        if remaining_time is not None and remaining_time <= 10800:  # 3 hours
            logger.info("Less than 3 hours remaining, stopping optimization")
            break

        study.optimize(
            lambda trial: objective(
                trial, experiment_config, args, logger, rank, device
            ),
            n_trials=n_trials,
            gc_after_trial=True,
        )

        # check time after trials
        elapsed_time = time.time() - start_time
        logger.info(
            f"Completed trial {trial_num + 1}/{n_trials}. Elapsed time: {elapsed_time:.2f} seconds"
        )

    # synchronize all processes
    if world_size > 1:
        dist.barrier()


def display_results(
    study: optuna.Study, optuna_dir: Path, logger: logging.Logger
) -> None:
    """Display the results of the Optuna study."""
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

    # determine the number of available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = num_gpus if num_gpus > 0 else 1

    # set up directories
    experiment_config = ExperimentConfig.from_yaml(args.config)
    model_dir = (
        experiment_config.root_dir / "models" / experiment_config.experiment_name
    )
    optuna_dir = model_dir / "optuna"
    dir_check_make(optuna_dir)

    # set up logging
    logger = setup_logging(
        str(optuna_dir / f"{experiment_config.experiment_name}_optuna.log")
    )

    # run optimization
    logger.info(f"Starting optimization process with {world_size} processes")
    logger.info(f"Configuration: {args}")
    if world_size > 1:
        try:
            mp.spawn(
                run_optimization,
                args=(world_size, args, logger, optuna_dir),
                nprocs=world_size,
                join=True,
            )
        except Exception as e:
            logger.error(f"Error in multiprocessing spawn: {str(e)}")
            raise
    else:
        run_optimization(0, world_size, args, logger, optuna_dir)

    # display results after optimization is over
    if world_size == 1 or (world_size > 1 and dist.get_rank() == 0):
        study = optuna.load_study(
            study_name="flexible_optimization", storage="sqlite:///optuna_study.db"
        )
        display_results(study, optuna_dir, logger)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
