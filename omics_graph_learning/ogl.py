#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Main script for running Omics Graph Learning"""


import argparse
import datetime
import os
import subprocess
import sys
from typing import Dict, Optional

import yaml  # type: ignore


def log_progress(message: str) -> None:
    """Print a log message with timestamp to stdout"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{timestamp}] {message}")


def parse_pipeline_arguments() -> argparse.ArgumentParser:
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
        help="Activation function to use. Options: relu, leakyrelu, gelu (default: relu).",
    )
    parser.add_argument("--dimensions", type=int, default="256")
    parser.add_argument("-l", "--residual", help="Use residual", action="store_true")
    parser.add_argument("--epochs", type=int, default="100")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Which optimizer to use for learning. Options: AdamW or Adam (default: AdamW)",
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
    return parser


def run_command(command: str, get_output: bool = False) -> Optional[str]:
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


def check_file(filename: str) -> bool:
    """Check if a file already exists"""
    return os.path.isfile(filename)


def parse_yaml(experiment_yaml: str) -> Dict[str, str]:
    """Reads YAML and returns as a dictionary."""
    with open(experiment_yaml, "r") as f:
        return yaml.safe_load(f)


def submit_slurm_job(job_script: str, dependency, args):
    """Sub,its a SLURM job and returns its job ID."""
    dependency_addendum = f"--dependency=afterok:{dependency}" if dependency else ""
    command = f"sbatch {dependency_addendum} {job_script} {args}"
    job_id = run_command(command, get_output=True)
    return job_id.split()[-1]  # Extract just the job ID


def get_splits(slurm_dependency):
    """Submit a SLURM job to get splits."""
    sbatch_command = f"sbatch --parsable --dependency=afterok:{slurm_dependency} get_training_targets.sh {args.experiment_yaml} {args.tpm_filter} {args.percent_of_samples_filter} {split_name}"
    if args.rna_seq:
        sbatch_command += " --rna_seq"
    return run_command(sbatch_command, get_output=True)


def main():
    # start by parsing args
    parser = parse_pipeline_arguments()
    args = parser.parse_args()


if __name__ == "__main__":
    main()


# # Set conda environment
# os.system("module load anaconda3/2022.10")
# os.system("conda activate /ocean/projects/bio210019p/stevesho/ogl")

# # =============================================================================
# # Main logic
# # =============================================================================
# # =============================================================================
# # Main script logic begins here
# # =============================================================================

# # Parse arguments
# parser = argparse.ArgumentParser()

# # Add all arguments here like it's done in example above
# # ...

# args = parser.parse_args()

# # Load environment modules and activate conda environment
# run_command("module load anaconda3/2022.10")
# run_command("conda activate /ocean/projects/bio210019p/stevesho/ogl")

# # Parse experiment YAML configuration
# config = parse_yaml(args.experiment_yaml)
# working_directory = config["working_directory"]
# experiment_name = config["experiment_name"]
# tissues = config["tissues"]

# # Construct file paths and check for their existence
# split_name = (
#     ...
# )  # This will be generated based on the script logic in actual conversion
# final_graph = os.path.join(
#     working_directory,
#     experiment_name,
#     "graphs",
#     split_name,
#     f"{experiment_name}_{args.graph_type}_{split_name}_graph_scaled.pkl",
# )

# # Example of checking if files exist and executing different SLURM jobs based on that
# if not check_file(final_graph):
#     # Do something if final graph does not exist
#     intermediate_graph = ...

#     if not check_file(intermediate_graph):
#         # Run the entire pipeline if intermediate graph does not exist
#         # Submit SLURM jobs for parsing nodes and edges, etc.
#         pass
#     else:
#         # Run parts of the pipeline, e.g., getting splits
#         split_id = get_splits("-1")
# else:
#     # Final graph exists, go straight to training
#     log_progress("Final graph found. Going straight to GNN training.")
#     submit_slurm_job(train, args=train_args)

# # Note that you will need to convert other parts of the bash script using similar patterns as shown above.
