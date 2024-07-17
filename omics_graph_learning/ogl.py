#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Main script for running Omics Graph Learning"""


import argparse
import datetime
import os
import subprocess
import sys
from typing import List, Optional, Tuple

from config_handlers import ExperimentConfig
from utils import _dataset_split_name


def _log_progress(message: str) -> None:
    """Print a log message with timestamp to stdout"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{timestamp}] {message}\n")


def _run_command(command: str, get_output: bool = False) -> Optional[str]:
    """Runs a shell command."""
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
            shell=True,
        )
        if get_output:
            return result.stdout.strip()
        else:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running command: {command}")
        print(e.output)
        sys.exit(1)
    return None


def _check_file(filename: str) -> bool:
    """Check if a file already exists"""
    return os.path.isfile(filename)


def submit_slurm_job(job_script: str, args: str, dependency: Optional[str]) -> str:
    """Submits a SLURM job and returns its job ID."""
    dependency_addendum = f"--dependency=afterok:{dependency}" if dependency else ""
    command = f"sbatch {dependency_addendum} {job_script} {args}"
    job_id = _run_command(command, get_output=True)
    assert job_id is not None
    return job_id.split()[-1]  # Extract just the job ID


def parse_pipeline_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the entire pipeline."""
    parser = argparse.ArgumentParser(
        description="Wrapper script for submitting jobs to SLURM."
    )
    parser.add_argument(
        "-ey",
        "--experiment_yaml",
        type=str,
        help="Path to experiment YAML file",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Partition for SLURM scheduling",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[
            "GCN",
            "GraphSAGE",
            "PNA",
            "GAT",
            "UniMPTransformer",
            "DeeperGCN",
            "MLP",
        ],
        help="GNN model type",
        required=True,
    )
    parser.add_argument(
        "-t", "--target", type=str, help="Target for GNN training", required=True
    )
    parser.add_argument(
        "-tf",
        "--tpm_filter",
        type=float,
        default=1.0,
        help="TPM values to filter genes by",
        required=True,
    )
    parser.add_argument(
        "-pf",
        "--percent_of_samples_filter",
        type=float,
        default=0.2,
        help="Number of samples that tpm must hit for threshold filter",
        required=True,
    )
    parser.add_argument(
        "-gl",
        "--gnn_layers",
        type=int,
        default=2,
        help="Number of GNN layers",
        required=True,
    )
    parser.add_argument(
        "-ll",
        "--linear_layers",
        type=int,
        default=3,
        help="Number of linear layers",
        required=True,
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leakyrelu", "gelu"],
        help="Activation function to use. Options: `relu`, `leakyrelu`, `gelu` (default: relu).",
    )
    parser.add_argument("--dimensions", type=int, default="256")
    parser.add_argument("-l", "--residual", help="Use residual", action="store_true")
    parser.add_argument("--epochs", type=int, default="100")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "AdamW"],
        help="Which optimizer to use for learning. Options: `Adam` or `AdamW` (default: Adam)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "linearwarmup"],
        help="Which scheduler to use for learning. Options: `plateau`, `cosine` or `linearwarmup`. (default: plateau)",
    )
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--total_random_edges", type=int, required=False, default=None)
    parser.add_argument("--graph_type", type=str, default="full")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--zero_nodes", action="store_true")
    parser.add_argument("--randomize_node_feats", action="store_true")
    parser.add_argument("--early_stop", action="store_true", default=False)
    parser.add_argument("--randomize_edges", action="store_true")
    parser.add_argument("--total_random_edges", type=int, default=False)
    parser.add_argument("--rna_seq", action="store_true")
    parser.add_argument("--gene_only_loader", action="store_true")
    return parser.parse_args()


def get_splits(
    slurm_dependency: str, args: argparse.Namespace, split_name: str = "split"
) -> str:
    """Submit a SLURM job to get splits."""
    sbatch_command = f"sbatch --parsable --dependency=afterok:{slurm_dependency} get_training_targets.sh {args.experiment_yaml} {args.tpm_filter} {args.percent_of_samples_filter} {split_name}"
    if args.rna_seq:
        sbatch_command += " --rna_seq"
    return _run_command(command=sbatch_command, get_output=True) or ""


def _get_file_paths(
    config: ExperimentConfig, args: argparse.Namespace, split_name: str
) -> Tuple[str, str]:
    """Construct file paths for graphs to check if files exist"""
    experiment_name = config.experiment_name
    graph_dir = config.graph_dir
    final_graph = os.path.join(
        graph_dir,
        split_name,
        f"{experiment_name}_{args.graph_type}_{split_name}_graph_scaled.pkl",
    )
    intermediate_graph = os.path.join(
        graph_dir,
        f"{experiment_name}_{args.graph_type}_graph.pkl",
    )
    return final_graph, intermediate_graph


def run_node_and_edge_generation(
    config: ExperimentConfig, args: argparse.Namespace
) -> List[str]:
    """Run node and edge generation jobs."""
    tissues = config.tissues
    partition_specific_script = (
        "pipeline_node_and_edge_generation_mem.sh"
        if args.partition == "EM"
        else "pipeline_node_and_edge_generation.sh"
    )
    pipeline_a_ids = []
    for tissue in tissues:
        job_id = submit_slurm_job(
            job_script=partition_specific_script,
            args=f"{args.experiment_yaml} omics_graph_learning/configs/{tissue}.yaml",
            dependency=None,
        )
        pipeline_a_ids.append(job_id)
    return pipeline_a_ids


def run_graph_concatenation(pipeline_a_ids: List[str], args: argparse.Namespace) -> str:
    """Run graph concatenation job."""
    constructor = "concat.sh"
    return submit_slurm_job(
        job_script=constructor,
        args=f"full {args.experiment_yaml}",
        dependency=":".join(pipeline_a_ids),
    )


def create_scalers(
    split_id: str, args: argparse.Namespace, split_name: str
) -> List[str]:
    """Create scalers for node features."""
    slurmids = []
    for num in range(39):
        job_id = submit_slurm_job(
            job_script="make_scaler.sh",
            args=f"{num} full {args.experiment_yaml} {split_name}",
            dependency=split_id,
        )
        slurmids.append(job_id)
    return slurmids


def scale_node_features(
    slurmids: List[str], args: argparse.Namespace, split_name: str
) -> str:
    """Run node feature scaling job."""
    return submit_slurm_job(
        job_script="scale_node_feats.sh",
        args=f"full {args.experiment_yaml} {split_name}",
        dependency=":".join(slurmids),
    )


def prepare_gnn_training_args(args: argparse.Namespace, split_name: str) -> str:
    """Prepare arguments for GNN training."""
    bool_flags = " ".join(
        [
            f"--{flag}"
            for flag in [
                "residual",
                "zero_nodes",
                "randomize_node_feats",
                "early_stop",
                "randomize_edges",
                "rna_seq",
            ]
            if getattr(args, flag)
        ]
    )

    train_args = (
        f"--experiment_config {args.experiment_yaml} "
        f"--model {args.model} "
        f"--target {args.target} "
        f"--gnn_layers {args.gnn_layers} "
        f"--linear_layers {args.linear_layers} "
        f"--activation {args.activation} "
        f"--dimensions {args.dimensions} "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.learning_rate} "
        f"--optimizer {args.optimizer} "
        f"--dropout {args.dropout} "
        f"--graph_type {args.graph_type} "
        f"--split_name {split_name} "
        f"{bool_flags}"
    )

    if args.heads:
        train_args += f" --heads {args.heads}"
    if args.total_random_edges:
        train_args += f" --total_random_edges {args.total_random_edges}"

    return train_args


def submit_gnn_job(
    args: argparse.Namespace, split_name: str, dependency: Optional[str]
) -> None:
    """Submit GNN training job."""
    train_args = prepare_gnn_training_args(args, split_name)
    submit_slurm_job(job_script="train_gnn.sh", args=train_args, dependency=dependency)
    _log_progress("GNN training job submitted.")


def main() -> None:
    """Run OGL pipeline, from data parsing to graph constructuion to GNN
    training with checks to avoid redundant computation."""
    # parse arguments and config
    args = parse_pipeline_arguments()
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)

    # get splitname
    split_name = _dataset_split_name(
        test_chrs=experiment_config.test_chrs,
        val_chrs=experiment_config.val_chrs,
        tpm_filter=args.tpm_filter,
        percent_of_samples_filter=args.percent_of_samples_filter,
    )
    if args.rna_seq:
        split_name += "_rna_seq"

    # store final and intermediate graph paths to check if files exist
    final_graph, intermediate_graph = _get_file_paths(
        config=experiment_config, args=args, split_name=split_name
    )

    _log_progress(f"Checking for final graph: {final_graph}")
    if not _check_file(final_graph):
        _log_progress(
            f"Final graph not found. Checking for intermediate graph: {intermediate_graph}"
        )
        if not _check_file(intermediate_graph):
            _log_progress("No intermediates found. Running entire pipeline!")

            pipeline_a_ids = run_node_and_edge_generation(experiment_config, args)
            _log_progress("Node and edge generation job submitted.")

            construct_id = run_graph_concatenation(pipeline_a_ids, args)
            _log_progress("Graph concatenation job submitted.")

            split_id = get_splits(construct_id, args, split_name)
            _log_progress("Dataset split job submitted.")
        else:
            _log_progress(
                "Intermediate graph found. Submitting jobs for dataset split, scaler, and training."
            )
            split_id = get_splits("-1", args, split_name)

        slurmids = create_scalers(split_id, args, split_name)
        _log_progress("Scaler jobs submitted.")

        scale_id = scale_node_features(slurmids, args, split_name)
        _log_progress("Node feature scaling job submitted.")

        submit_gnn_job(args, split_name, scale_id)
    else:
        _log_progress("Final graph found. Going straight to GNN training.")
        submit_gnn_job(args, split_name, None)


if __name__ == "__main__":
    main()
