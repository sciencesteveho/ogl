#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Main script for running the Omics Graph Learning pipeline, built for running
on a HPC via the SLURM scheduler."""


import argparse
import os
import sys
from typing import List, Optional, Tuple

import pytest

from config_handlers import ExperimentConfig
from constants import NodePerturbation
from utils import _check_file
from utils import _dataset_split_name
from utils import _log_progress
from utils import _run_command
from utils import submit_slurm_job


def run_tests() -> bool:
    """Run all unit tests to ensure pipeline code is functioning correctly.

    Returns:
        bool: `True` if all tests pass, `False` otherwise
    """
    exit_code = pytest.main([])
    return exit_code == 0  # pytest.ExitCode.OK is 0


class PipelineRunner:
    """Class for handling the entire pipeline, from data parsing to graph
    construction and GNN training.

    Attributes:
        config (ExperimentConfig): Experiment configuration
        args (argparse.Namespace): Parsed CLI arguments specific to the GNN
        experiment

    Methods:
    --------
    run_pipeline: Run the entire pipeline

    Examples:
    --------
    # Run the pipeline and train a model according to default params
    >>> python ogl.py \
        --partition [PARTITION] 
        --experiment_yaml [EXPERIMENT_YAML_PATH] \

    # Run the pipeline and train a model, overriding default params
    >>> python ogl.py \
        --partition [PARTITION] \
        --experiment_yaml [EXPERIMENT_YAML_PATH] \
        --model GAT \
        --heads 2

    # Run the pipeline and optimize hyperparameters instead of training a full
    model
    >>> python ogl.py \
        --partition [PARTITION] \
        --experiment_yaml [EXPERIMENT_YAML_PATH] \
        --optimize_params
    """

    def __init__(self, config: ExperimentConfig, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

    def _get_file_paths(self, split_name: str) -> Tuple[str, str]:
        """Construct file paths for graphs to check if files exist"""
        experiment_name = self.config.experiment_name
        graph_dir = self.config.graph_dir
        final_graph = os.path.join(
            graph_dir,
            split_name,
            f"{experiment_name}_{self.config.graph_type}_scaled.pkl",
        )
        intermediate_graph = os.path.join(
            graph_dir,
            f"{experiment_name}_{self.config.graph_type}_graph.pkl",
        )
        return final_graph, intermediate_graph

    def get_train_test_val_split(self, slurm_dependency: str, split_name: str) -> str:
        """Submit a SLURM job to get splits."""
        sbatch_command = f"sbatch --parsable --dependency=afterok:{slurm_dependency} training_targets.sh {self.args.experiment_yaml} {self.args.tpm_filter} {self.args.percent_of_samples_filter} {split_name}"
        if self.args.target == "rna_seq":
            sbatch_command += " --rna_seq"
        return _run_command(command=sbatch_command, get_output=True) or ""

    def run_node_and_edge_generation(self, split_name: str) -> List[str]:
        """Run node and edge generation jobs."""
        tissues = self.config.tissues
        partition_specific_script = (
            "pipeline_node_and_edge_generation_mem.sh"
            if self.args.partition == "EM"
            else "pipeline_node_and_edge_generation.sh"
        )

        slurmids = []
        for tissue in tissues:
            args = f"{self.args.experiment_yaml} \
                {self.config.sample_config_dir}/{tissue}.yaml \
                {self.args.tpm_filter} \
                {self.args.percent_of_samples_filter} \
                {self.args.filter_mode} \
                {split_name} \
                {self.args.target}"

            if self.args.positional_encoding:
                args += " --positional_encoding"

            job_id = submit_slurm_job(
                job_script=partition_specific_script,
                args=args,
                dependency=None,
            )
            slurmids.append(job_id)
        return slurmids

    def run_graph_concatenation(self, slurmids: List[str], split_name: str) -> str:
        """Run graph concatenation job."""
        constructor = "concat.sh"
        return submit_slurm_job(
            job_script=constructor,
            args=f"{self.args.experiment_yaml} \
                {split_name}",
            dependency=":".join(slurmids),
        )

    def scale_node_features(self, slurmids: List[str], split_name: str) -> str:
        """Run node feature scaling job."""
        return submit_slurm_job(
            job_script="scale_features.sh",
            args=f"{self.args.experiment_yaml} \
                {split_name}",
            dependency=":".join(slurmids),
        )

    def prepare_gnn_training_args(
        self, args: argparse.Namespace, split_name: str
    ) -> str:
        """Prepare arguments for GNN training."""
        bool_flags = " ".join(
            [
                f"--{flag}"
                for flag in [
                    "early_stop",
                    "task_specific_mlp",
                    "positional_encoding",
                    "gene_only_loader",
                    "optimize_params",
                    "run_tests",
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
            f"--residual {args.residual} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size} "
            f"--learning_rate {args.learning_rate} "
            f"--optimizer {args.optimizer} "
            f"--scheduler {args.scheduler} "
            f"--dropout {args.dropout} "
            f"--split_name {split_name} "
        )

        if args.heads:
            train_args += f" --heads {args.heads}"
        if args.total_random_edges:
            if args.edge_perturbation == "randomize_edges":
                train_args += f" --total_random_edges {args.total_random_edges}"
            else:
                raise ValueError(
                    "`total_random_edges` should only be set when `randomize_edges` is True"
                )
        if args.node_perturbation:
            train_args += f" --node_perturbation {args.node_perturbation}"
        if args.edge_perturbation:
            train_args += f" --edge_perturbation {args.edge_perturbation}"
        return train_args + bool_flags

    def submit_gnn_job(self, split_name: str, dependency: Optional[str]) -> None:
        """Submit GNN training job."""
        train_args = self.prepare_gnn_training_args(self.args, split_name)
        if self.args.optimize_params:
            submit_slurm_job(
                job_script="optimize_params.hs",
                args=f"{self.args.experiment_yaml} {self.args.target} {split_name}",
                dependency=dependency,
            )
            _log_progress("Hyperparameter optimization job submitted.")
        else:
            submit_slurm_job(
                job_script="train_gnn.sh", args=train_args, dependency=dependency
            )
            _log_progress("GNN training job submitted.")

    def all_pipeline_jobs(self, intermediate_graph: str, split_name: str) -> None:
        """Submit all pipeline jobs if a final graph is not found."""
        _log_progress(
            f"Final graph not found. Checking for intermediate graph: {intermediate_graph}"
        )
        if not _check_file(intermediate_graph):
            run_id = self.graph_construction_jobs(split_name)
        else:
            _log_progress(
                "Intermediate graph found. Submitting jobs for dataset split, scaler, and training."
            )

        # construct_id = self.run_graph_concatenation(
        #     slurmids=slurmids, split_name=split_name
        # )

        # slurmids: list[str] = []
        # scale_id = self.scale_node_features(slurmids, split_name)
        # _log_progress("Node feature scaling job submitted.")

        # self.submit_gnn_job(split_name, scale_id, args.optimize_params)

    def graph_construction_jobs(self, split_name: str) -> str:
        """Submit jobs for node and edge generation, local context parsing, and
        graph construction."""
        _log_progress("No intermediates found. Running entire pipeline!")

        # slurmids = self.run_node_and_edge_generation(split_name=split_name)
        _log_progress("Node and edge generation job submitted.")

        # slurmids = []
        # construct_id = self.run_graph_concatenation(
        #     slurmids=slurmids, split_name=split_name
        # )
        # _log_progress("Graph concatenation job submitted.")

        return "placeholder"
        # return construct_id

    def run_pipeline(self) -> None:
        """Run the pipeline! Check for existing files and submit jobs as needed."""
        split_name = _dataset_split_name(
            test_chrs=self.config.test_chrs,
            val_chrs=self.config.val_chrs,
            tpm_filter=self.args.tpm_filter,
            percent_of_samples_filter=self.args.percent_of_samples_filter,
        )
        if self.args.target == "rna_seq":
            split_name += "_rna_seq"

        print(f"Starting process for: {split_name}")
        final_graph, intermediate_graph = self._get_file_paths(split_name=split_name)

        _log_progress(f"Checking for final graph: {final_graph}")
        if not _check_file(final_graph):
            self.all_pipeline_jobs(intermediate_graph, split_name)
        else:
            _log_progress("Final graph found. Going straight to GNN training.")
            self.submit_gnn_job(split_name, None)


def validate_args(args: argparse.Namespace) -> None:
    """Helper function to validate CLI arguments that have dependencies."""
    if args.partition not in ["RM", "EM"]:
        print("Error: --partition must be 'RM' or 'EM'")
        sys.exit(1)

    if args.target != "rna_seq" and args.filter_mode is None:
        print("Error: if target type is not `rna_seq`, --filter_mode is required")
        sys.exit(1)

    if args.model in ["GAT", "UniMPTransformer"] and args.heads is None:
        print(f"Error: --heads is required when model is {args.model}")
        sys.exit(1)

    if args.total_random_edges and args.edge_perturbation != "randomize_edges":
        print(
            "Error: if --total_random_edges is set, --edge_perturbation must be `randomize_edges`"
        )
        sys.exit(1)


def parse_pipeline_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the entire pipeline."""
    # set up list of perturbation options
    perturbation_choices: List[str] = [
        perturbation.name for perturbation in NodePerturbation
    ]

    parser = argparse.ArgumentParser(description="Omics Graph Learning Pipeline")
    parser.add_argument(
        "--experiment_yaml",
        type=str,
        required=True,
        help="Path to experiment YAML file",
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        choices=["RM", "EM"],
        help="Partition for SLURM scheduling",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
        choices=[
            "GCN",
            "GraphSAGE",
            "PNA",
            "GAT",
            "UniMPTransformer",
            "DeeperGCN",
            "MLP",
        ],
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
    parser.add_argument("--tpm_filter", type=float, default=1.0)
    parser.add_argument("--percent_of_samples_filter", type=float, default=0.2)
    parser.add_argument(
        "--filter_mode",
        type=str,
        help="Mode to filter genes, specifying within the target tissue or across all possible gtex tissues (e.g. `within` or `across`). This is required if the target type is not `rna_seq`",
        default="within",
    )
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--linear_layers", type=int, default=3)
    parser.add_argument(
        "--activation", type=str, default="relu", choices=["relu", "leakyrelu", "gelu"]
    )
    parser.add_argument("--task_specific_mlp", action="store_true", default=False)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument(
        "--residual",
        type=str,
        default=None,
        choices=["shared_source", "distinct_source", None],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["Adam", "AdamW"]
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "linear_warmup"],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--positional_encoding", action="store_true")
    parser.add_argument("--early_stop", action="store_true", default=True)
    parser.add_argument(
        "--node_perturbation",
        type=str,
        default=None,
        choices=perturbation_choices,
        help="Type of node based perturbation to apply. Choose from either `zero_node_feats`, `randomize_node_feats`, `randomize_node_feat_order`, or pick the name of a specific feat to perturb",
    )
    parser.add_argument(
        "--edge_perturbation",
        type=str,
        default=None,
        choices=["randomize_edges", "remove_all_edges", "remove_specific_edges"],
        help="Type of node based perturbation to apply. Choose from either `zero_node_feats`, `randomize_node_feats`, `randomize_node_feat_order`, or pick the name of a specific feat to perturb",
    )
    parser.add_argument("--total_random_edges", type=int, default=None)
    parser.add_argument("--gene_only_loader", action="store_true")
    parser.add_argument("--optimize_params", action="store_true")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run unit tests before executing the pipeline",
        default=False,
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    """Run OGL pipeline, from data parsing to graph constructuion to GNN
    training with checks to avoid redundant computation."""
    args = parse_pipeline_arguments()

    # run unit tests first
    # if args.run_tests:
    #     passed_tests = run_tests()
    #     if not passed_tests:
    #         print("Unit tests failed. Exiting.")
    #         sys.exit(1)

    # run OGL pipeline
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)
    pipe_runner = PipelineRunner(config=experiment_config, args=args)
    pipe_runner.run_pipeline()


if __name__ == "__main__":
    main()
