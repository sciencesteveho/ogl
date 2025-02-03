# sourcery skip: avoid-global-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Compare model performance metrics across various runs."""


import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

from omics_graph_learning.model_metrics import ModelMetrics
from omics_graph_learning.visualization import set_matplotlib_publication_parameters


def plot_optimization_history(
    optuna_log: str,
    output_dir: str,
) -> None:
    """Plot optimization history from an Optuna log file."""
    set_matplotlib_publication_parameters()

    # read data
    df = pd.read_csv(optuna_log)

    # set up plot
    plt.figure(figsize=(3, 2.2))
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    pastel_blue = "#5187bd"
    pastel_red = "#c46262"

    # scatter trials
    plt.scatter(
        df["number"],
        df["value"],
        color=pastel_blue,
        s=1.25,
        alpha=1,
        label="Objective Value",
    )

    # calculate running minimum
    running_min = df.sort_values("number")["value"].expanding().min()
    plt.plot(
        df.sort_values("number")["number"],
        running_min,
        color=pastel_red,
        label="Best Value",
        linewidth=1,
    )

    plt.xlabel("Trial")
    plt.ylabel("Objective value (loss)")
    plt.title("")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().set_xticks(np.arange(50, 201, 50))
    plt.gca().set_yticks(np.arange(0.95, 1.21, 0.05))

    # add legend inside the plot
    plt.legend(
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.02),
        frameon=False,
        labelcolor="#808080",
    )

    plt.xlim(0, 202.5)
    plt.ylim(0.9, 1.22)
    plt.tight_layout()
    plt.savefig("optuna_history.png", dpi=450)
    plt.clf()


def get_metric_dataframes(
    models: Dict[str, str], base_model_dir: Path, model_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get per-run and summary dataframes for a dictionary of models, adding a
    'Model_Group' column with readable names.
    """
    per_run_dfs = []
    summary_dfs = []

    for model_key, readable_name in models.items():
        model_path = base_model_dir / model_key
        if not model_path.exists():
            print(f"Warning: Model path {model_path} does not exist. Skipping.")
            continue

        metrics_obj = ModelMetrics(model_path)
        per_run_df = metrics_obj.get_per_run_df()
        summary_df = metrics_obj.get_summary_df()

        if model_type == "best":
            # remove '_replicate_num' suffix using regex
            base_model_key = re.sub(r"_replicate_\d+$", "", model_key)
            # fetch the readable name for the base model if it exists
            base_readable = models.get(base_model_key, base_model_key)
            per_run_df["Model_Group"] = base_readable
            summary_df["Model_Group"] = base_readable
        else:
            per_run_df["Model_Group"] = readable_name
            summary_df["Model_Group"] = readable_name

        per_run_dfs.append(per_run_df)
        summary_dfs.append(summary_df)

    if not per_run_dfs or not summary_dfs:
        print(f"No dataframes were created for model type '{model_type}'.")
        return pd.DataFrame(), pd.DataFrame()

    combined_per_run_df = pd.concat(per_run_dfs, ignore_index=True)
    combined_summary_df = pd.concat(summary_dfs, ignore_index=True)

    # group replicates by 'Model_Group' and calculate the mean
    # especially for 'best' models
    if model_type == "best":
        if numeric_cols := [
            col
            for col in combined_summary_df.columns
            if col not in ["Model", "Model_Group"]
            and pd.api.types.is_numeric_dtype(combined_summary_df[col])
        ]:
            grouped_summary = combined_summary_df.groupby(
                "Model_Group", as_index=False
            )[numeric_cols].mean()

            combined_summary_df = grouped_summary
        else:
            print("No numeric columns found to group by 'Model_Group'.")

    return combined_per_run_df, combined_summary_df


def compute_cohens_d(x: List[Union[int, float]], y: List[Union[int, float]]) -> float:
    """Compute Cohen's d for two independent samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    )
    # avoid division by zero if pooled_std = 0
    return 0.0 if pooled_std == 0 else (np.mean(x) - np.mean(y)) / pooled_std


def compare_best_worst_within_category(
    combined_metrics_df: pd.DataFrame,
) -> float:
    """Compute and display Cohen's d between the best and worst performing
    models within each category.
    """
    # mean performance per model group
    performance_grouped_df = (
        combined_metrics_df.groupby("Model_Group")["Validation_Pearson"]
        .mean()
        .reset_index()
    )

    # get best and worst models based on mean correlations
    best_model = performance_grouped_df.loc[
        performance_grouped_df["Validation_Pearson"].idxmax(), "Model_Group"
    ]
    worst_model = performance_grouped_df.loc[
        performance_grouped_df["Validation_Pearson"].idxmin(), "Model_Group"
    ]

    data_best = combined_metrics_df[combined_metrics_df["Model_Group"] == best_model][
        "Validation_Pearson"
    ].values
    data_worst = combined_metrics_df[combined_metrics_df["Model_Group"] == worst_model][
        "Validation_Pearson"
    ].values

    # convert to numpy arrays
    data_best = np.array(data_best)
    data_worst = np.array(data_worst)

    # calculate Cohen's d
    co_d = compute_cohens_d(data_best.tolist(), data_worst.tolist())
    print(f"  Best Model: {best_model} (Mean = {np.mean(data_best):.4f})")
    print(f"  Worst Model: {worst_model} (Mean = {np.mean(data_worst):.4f})")
    print(
        f"  Cohen's d: {co_d:.2f} "
        f"({'Large' if abs(co_d) >= 0.8 else 'Medium' if abs(co_d) >= 0.5 else 'Small'})"
    )
    return co_d


def main() -> None:
    """Generate figures for iterative model construction."""
    # plot optimization history
    # plot_optimization_history(optuna_log="optuna_results.csv", output_dir=".")

    alpha_models = {
        "k562_release_0.65": "α=0.65",
        # "k562_release_0.70": "α=0.70",
        "k562_release_0.75": "α=0.75",
        # "k562_release_0.80": "α=0.80",
        "k562_release_0.85": "α=0.85",
        # "k562_release_0.90": "α=0.90",
        "k562_release_0.95": "α=0.95",
        "k562_release_1.0": "α=1.0",
    }

    node_models = {
        "k562_release_replicate_3": "Baseline",
        "k562_release_cpgislands": "+ CpG islands",
        "k562_release_crms": "+ Cis-regulatory modules",
        "k562_release_ctcf": "+ CTCF cCREs",
        "k562_release_superenhancers": "+ Superenhancers",
        "k562_release_tfbindingsites": "+ TF binding footprints",
        # "k562_release_tss": "+ Transcription start sites",
        "k562_release_all_nodes": "+ All node types (combined)",
    }

    operator_models = {
        "k562_release_alpha_0.95_GAT": "GATv2",
        "k562_release_alpha_0.95_GCN": "GCN",
        "k562_release_alpha_0.95_GraphSAGE": "GraphSAGE",
        "k562_release_alpha_0.95_PNA": "PNA",
        "k562_release_alpha_0.95_UniMPTransformer": "UniMPTransformer",
    }

    graph_construction_models = {
        "k562_release_replicate_3": "Baseline",
        "k562_allcontacts_global_encode": "ENCODE-only regulatory catalogue",
        "k562_allcontacts_global_epimap": "EpiMap-only regulatory catalogue",
        # "k562_release_gene_gene": "+ Gene-gene interactions",
    }

    interaction_models = {
        "k562_release_replicate_3": "Baseline",
        "k562_release_all_interact": "+ miRNA and RBP interactions",
        "k562_release_all_nodes_and_interact": "+ All additional node and interact types",
        "k562_release_mirna": "+ miRNA interactions",
        "k562_release_rbp": "+ RNA-binding protein interactions",
    }

    batch_and_lr_models = {
        "k562_release_alpha_0.95_batch_128": "batch=128, LR=0.0005",
        "k562_release_alpha_0.95_batch_256": "batch=256, LR=0.0005",
        "k562_release_alpha_0.95_batch_32": "batch=32, LR=0.0005",
        "k562_release_alpha_0.95_lr_0.0001": "batch=64, LR=0.0001",
        "k562_release_alpha_0.95_lr_0.0003": "batch=64, LR=0.0003",
        "k562_release_alpha_0.95_lr_0.001_batch_32": "batch=32, LR=0.001",
        "k562_release_alpha_0.95_lr_0.005_batch_32": "batch=32, LR=0.005",
        "k562_release_replicate_3": "batch=16, LR=0.0005",
    }

    dropout_models = {
        "k562_release_alpha_0.95_dropout_0.1": "dropout 0.1",
        "k562_release_alpha_0.95_dropout_0.2": "dropout 0.2",
        "k562_release_alpha_0.95_GAT": "dropout 0.3",
        "k562_release_alpha_0.95_dropout_0.4": "dropout 0.4",
        "k562_release_alpha_0.95_dropout_0.5": "dropout 0.5",
    }

    local_models = {
        "k562_release_replicate_3": "Baseline",
        "k562_allcontacts_global_gene_gene": "Local only",
    }

    base_model_dir = Path("/Users/steveho/gnn_plots/figure_2/model_performance")

    model_categories = [
        ("Alpha", alpha_models, "alpha"),
        ("Node", node_models, "node"),
        ("Operator", operator_models, "operator"),
        ("Graph Construction", graph_construction_models, "graph_construction"),
        ("Interactions", interaction_models, "interaction"),
        ("Batch and Learning Rate", batch_and_lr_models, "batch_and_lr"),
        ("Dropout", dropout_models, "dropout"),
        ("Local", local_models, "local"),
    ]

    for category_name, models, model_type in model_categories:
        combined_metrics_df, performance_df = get_metric_dataframes(
            models, base_model_dir, model_type
        )
        # calculate Cohen's d between best and worst performing models
        co_d = compare_best_worst_within_category(combined_metrics_df)

        # add cohens_d to performance_df
        performance_df.attrs["cohens_d"] = co_d

        # save the combined and summary DataFrames to CSV files
        combined_metrics_df.to_csv(
            f"combined_model_metrics_{model_type}.csv", index=False
        )
        performance_df.to_csv(
            f"model_performance_summary_{model_type}.csv", index=False
        )
        # plot_iterative_model(model_type)


if __name__ == "__main__":
    main()
