#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Main script for running the Omics Graph Learning pipeline, built for running
on a HPC via the SLURM scheduler."""


import argparse
import os
import sys
from typing import List, Optional, Tuple

from config_handlers import ExperimentConfig
from utils import _check_file
from utils import _dataset_split_name
from utils import _log_progress
from utils import _run_command
from utils import parse_yaml
from utils import submit_slurm_job


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
            f"{experiment_name}_{self.args.graph_type}_{split_name}_graph_scaled.pkl",
        )
        intermediate_graph = os.path.join(
            graph_dir,
            f"{experiment_name}_{self.args.graph_type}_graph.pkl",
        )
        return final_graph, intermediate_graph

    def get_splits(self, slurm_dependency: str, split_name: str = "split") -> str:
        """Submit a SLURM job to get splits."""
        sbatch_command = f"sbatch --parsable --dependency=afterok:{slurm_dependency} get_training_targets.sh {self.args.experiment_yaml} {self.args.tpm_filter} {self.args.percent_of_samples_filter} {split_name}"
        if self.args.target == "rna_seq":
            sbatch_command += " --rna_seq"
        return _run_command(command=sbatch_command, get_output=True) or ""

    def run_node_and_edge_generation(self) -> List[str]:
        """Run node and edge generation jobs."""
        tissues = self.config.tissues
        partition_specific_script = (
            "pipeline_node_and_edge_generation_mem.sh"
            if self.args.partition == "EM"
            else "pipeline_node_and_edge_generation.sh"
        )
        pipeline_a_ids = []
        for tissue in tissues:
            job_id = submit_slurm_job(
                job_script=partition_specific_script,
                args=f"{self.args.experiment_yaml} omics_graph_learning/configs/{tissue}.yaml",
                dependency=None,
            )
            pipeline_a_ids.append(job_id)
        return pipeline_a_ids

    def run_graph_concatenation(self, pipeline_a_ids: List[str]) -> str:
        """Run graph concatenation job."""
        constructor = "concat.sh"
        return submit_slurm_job(
            job_script=constructor,
            args=f"full {self.args.experiment_yaml}",
            dependency=":".join(pipeline_a_ids),
        )

    def create_scalers(self, split_id: str, split_name: str) -> List[str]:
        """Create scalers for node features."""
        slurmids = []
        for num in range(39):
            job_id = submit_slurm_job(
                job_script="make_scaler.sh",
                args=f"{num} full {self.args.experiment_yaml} {split_name}",
                dependency=split_id,
            )
            slurmids.append(job_id)
        return slurmids

    def scale_node_features(self, slurmids: List[str], split_name: str) -> str:
        """Run node feature scaling job."""
        return submit_slurm_job(
            job_script="scale_node_feats.sh",
            args=f"full {self.args.experiment_yaml} {split_name}",
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

    def submit_gnn_job(self, split_name: str, dependency: Optional[str]) -> None:
        """Submit GNN training job."""
        # train_args = self.prepare_gnn_training_args(self.args, split_name)
        # submit_slurm_job(
        #     job_script="train_gnn.sh", args=train_args, dependency=dependency
        # )
        # _log_progress("GNN training job submitted.")

    def all_pipeline_jobs(self, intermediate_graph: str, split_name: str) -> None:
        """Submit all pipeline jobs if a final graph is not found."""
        # _log_progress(
        #     f"Final graph not found. Checking for intermediate graph: {intermediate_graph}"
        # )
        # if not _check_file(intermediate_graph):
        #     split_id = self.graph_construction_jobs(split_name)
        # else:
        #     _log_progress(
        #         "Intermediate graph found. Submitting jobs for dataset split, scaler, and training."
        #     )
        #     split_id = self.get_splits("-1", split_name)

        # slurmids = self.create_scalers(split_id, split_name)
        # _log_progress("Scaler jobs submitted.")

        # scale_id = self.scale_node_features(slurmids, split_name)
        # _log_progress("Node feature scaling job submitted.")

        # self.submit_gnn_job(split_name, scale_id)

    def graph_construction_jobs(self, split_name: str) -> str:
        """Submit jobs for node and edge generation, local context parsing, and
        graph construction."""
        _log_progress("No intermediates found. Running entire pipeline!")

        pipeline_a_ids = self.run_node_and_edge_generation()
        _log_progress("Node and edge generation job submitted.")

        # construct_id = self.run_graph_concatenation(pipeline_a_ids)
        # _log_progress("Graph concatenation job submitted.")

        # split_id = self.get_splits(construct_id, split_name)
        # _log_progress("Dataset split job submitted.")
        return "placeholder"
        # return split_id

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

        final_graph, intermediate_graph = self._get_file_paths(split_name=split_name)

        _log_progress(f"Checking for final graph: {final_graph}")
        if not _check_file(final_graph):
            _log_progress("Final graph not found. Checking for intermediate graph.")
            self.all_pipeline_jobs(intermediate_graph, split_name)
        else:
            _log_progress("Final graph found. Going straight to GNN training.")
            self.submit_gnn_job(split_name, None)


def validate_args(args: argparse.Namespace) -> None:
    """Helper function to validate CLI arguments that have dependencies."""
    if args.partition not in ["RM", "EM"]:
        print("Error: --partition must be 'RM' or 'EM'")
        sys.exit(1)

    if args.model in ["GAT", "UniMPTransformer"] and args.heads is None:
        print(f"Error: --heads is required when model is {args.model}")
        sys.exit(1)

    if args.randomize_edges and args.total_random_edges is None:
        print("Error: --total_random_edges is required when --randomize_edges is set")
        sys.exit(1)


def parse_pipeline_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the entire pipeline."""
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
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--linear_layers", type=int, default=3)
    parser.add_argument(
        "--activation", type=str, default="relu", choices=["relu", "leakyrelu", "gelu"]
    )
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--residual", action="store_true")
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
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--graph_type", type=str, default="full")
    parser.add_argument("--zero_nodes", action="store_true")
    parser.add_argument("--randomize_node_feats", action="store_true")
    parser.add_argument("--randomize_edges", action="store_true")
    parser.add_argument("--total_random_edges", type=int, default=None)
    parser.add_argument("--gene_only_loader", action="store_true")
    parser.add_argument("--optimize_params", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    """Run OGL pipeline, from data parsing to graph constructuion to GNN
    training with checks to avoid redundant computation."""
    args = parse_pipeline_arguments()
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)
    pipe_runner = PipelineRunner(config=experiment_config, args=args)
    pipe_runner.run_pipeline()


if __name__ == "__main__":
    main()
