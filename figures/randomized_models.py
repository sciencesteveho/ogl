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
from omics_graph_learning.model_metrics import ModelMetrics
from omics_graph_learning.visualization import set_matplotlib_publication_parameters
import pandas as pd  # type: ignore
import pingouin as pg  # type: ignore
from scipy.stats import f_oneway  # type: ignore
from scipy.stats import ttest_rel  # type: ignore
from scipy.stats import wilcoxon  # type: ignore
import seaborn as sns  # type: ignore
from statsmodels.stats.anova import AnovaRM  # type: ignore
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # type: ignore


class ModelMetrics:
    """Class to store model performance metrics and compute statistics across
    runs.

    Attributes:
        model_dir: Path to the model directory.
        per_run_df: DataFrame containing metrics for each run.
        summary_df: DataFrame containing mean and std for each metric.
    """

    def __init__(self, model_dir: Union[Path, str]) -> None:
        """Initialize the ModelMetrics object."""
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        self.model_dir = model_dir

        # collect metrics from all runs
        self.per_run_df = self._load_metrics()

        # compute summary statistics
        self.summary_df = self._compute_summary()

    def _load_metrics(self) -> pd.DataFrame:
        """Load metrics from model evaluation for all runs."""
        metrics = []

        for run_number in range(1, 4):
            metrics_file = self.model_dir / f"run_{run_number}" / "eval_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_json = json.load(f)
                    metrics.append(
                        {
                            "Model": self.model_dir.name,
                            "Run": run_number,
                            # Final test metrics
                            "Final_Test_Loss": metrics_json.get(
                                "Final test loss", np.nan
                            ),
                            "Final_Test_Pearson": metrics_json.get(
                                "Final test pearson", np.nan
                            ),
                            "Final_Test_RMSE": metrics_json.get(
                                "Final test RMSE", np.nan
                            ),
                            "Final_Test_Accuracy": metrics_json.get(
                                "Final test accuracy", np.nan
                            ),
                            # Validation metrics
                            "Validation_Loss": metrics_json.get(
                                "Validation loss", np.nan
                            ),
                            "Validation_Pearson": metrics_json.get(
                                "Validation pearson", np.nan
                            ),
                            "Validation_RMSE": metrics_json.get(
                                "Validation RMSE", np.nan
                            ),
                            "Validation_Accuracy": metrics_json.get(
                                "Validation accuracy", np.nan
                            ),
                            # Bootstrap metrics (test)
                            "Bootstrap_Pearson": metrics_json.get(
                                "Bootstrap pearson", np.nan
                            ),
                            "CI_lower_test": metrics_json.get("CI lower test", np.nan),
                            "CI_upper_test": metrics_json.get("CI upper test", np.nan),
                            "CI_Lower": metrics_json.get("CI lower", np.nan),
                            "CI_Upper": metrics_json.get("CI upper", np.nan),
                            # Bootstrap metrics (validation)
                            "Bootstrap_Pearson_Val": metrics_json.get(
                                "Bootstrap pearson val", np.nan
                            ),
                            "CI_Lower_Val": metrics_json.get("CI lower val", np.nan),
                            "CI_Upper_Val": metrics_json.get("CI upper val", np.nan),
                        }
                    )
            else:
                print(f"Metrics file not found: {metrics_file}")

        if not metrics:
            # if no metrics were loaded, return a DataFrame with NaNs
            return pd.DataFrame(
                {
                    "Model": [self.model_dir.name],
                    "Run": [np.nan],
                    "Final_Test_Loss": [np.nan],
                    "Final_Test_Pearson": [np.nan],
                    "Final_Test_RMSE": [np.nan],
                    "Final_Test_Accuracy": [np.nan],
                    "Validation_Loss": [np.nan],
                    "Validation_Pearson": [np.nan],
                    "Validation_RMSE": [np.nan],
                    "Validation_Accuracy": [np.nan],
                    "Bootstrap_Pearson": [np.nan],
                    "CI_Lower": [np.nan],
                    "CI_Upper": [np.nan],
                    "Bootstrap_Pearson_Val": [np.nan],
                    "CI_Lower_Val": [np.nan],
                    "CI_Upper_Val": [np.nan],
                }
            )

        # create DataFrame from the list of metrics
        return pd.DataFrame(metrics)

    def _compute_summary(self) -> pd.DataFrame:
        """Compute mean and std for each metric."""
        mean_metrics = self.per_run_df.mean(numeric_only=True)
        std_metrics = self.per_run_df.std(numeric_only=True, ddof=1)

        summary_data = {
            "Model": self.model_dir.name,
            # Final test
            "Final_Test_Loss_Mean": mean_metrics.get("Final_Test_Loss", np.nan),
            "Final_Test_Loss_Std": std_metrics.get("Final_Test_Loss", np.nan),
            "Final_Test_Pearson_Mean": mean_metrics.get("Final_Test_Pearson", np.nan),
            "Final_Test_Pearson_Std": std_metrics.get("Final_Test_Pearson", np.nan),
            "Final_Test_RMSE_Mean": mean_metrics.get("Final_Test_RMSE", np.nan),
            "Final_Test_RMSE_Std": std_metrics.get("Final_Test_RMSE", np.nan),
            "Final_Test_Accuracy_Mean": mean_metrics.get("Final_Test_Accuracy", np.nan),
            "Final_Test_Accuracy_Std": std_metrics.get("Final_Test_Accuracy", np.nan),
            "CI_lower_test_mean": mean_metrics.get("CI_lower_test", np.nan),
            "CI_upper_test_mean": mean_metrics.get("CI_upper_test", np.nan),
            # Validation
            "Validation_Loss_Mean": mean_metrics.get("Validation_Loss", np.nan),
            "Validation_Loss_Std": std_metrics.get("Validation_Loss", np.nan),
            "Validation_Pearson_Mean": mean_metrics.get("Validation_Pearson", np.nan),
            "Validation_Pearson_Std": std_metrics.get("Validation_Pearson", np.nan),
            "Validation_RMSE_Mean": mean_metrics.get("Validation_RMSE", np.nan),
            "Validation_RMSE_Std": std_metrics.get("Validation_RMSE", np.nan),
            "Validation_Accuracy_Mean": mean_metrics.get("Validation_Accuracy", np.nan),
            "Validation_Accuracy_Std": std_metrics.get("Validation_Accuracy", np.nan),
            # Bootstrap (test)
            "CI_Lower_Mean": mean_metrics.get("CI_Lower", np.nan),
            "CI_Lower_Std": std_metrics.get("CI_Lower", np.nan),
            "CI_Upper_Mean": mean_metrics.get("CI_Upper", np.nan),
            "CI_Upper_Std": std_metrics.get("CI_Upper", np.nan),
            "Bootstrap_Pearson_Mean": mean_metrics.get("Bootstrap_Pearson", np.nan),
            "Bootstrap_Pearson_Std": std_metrics.get("Bootstrap_Pearson", np.nan),
            # Bootstrap (validation)
            "Bootstrap_Pearson_Val_Mean": mean_metrics.get(
                "Bootstrap_Pearson_Val", np.nan
            ),
            "Bootstrap_Pearson_Val_Std": std_metrics.get(
                "Bootstrap_Pearson_Val", np.nan
            ),
            "CI_Lower_Val_Mean": mean_metrics.get("CI_Lower_Val", np.nan),
            "CI_Lower_Val_Std": std_metrics.get("CI_Lower_Val", np.nan),
            "CI_Upper_Val_Mean": mean_metrics.get("CI_Upper_Val", np.nan),
            "CI_Upper_Val_Std": std_metrics.get("CI_Upper_Val", np.nan),
        }

        return pd.DataFrame([summary_data])

    def get_per_run_df(self) -> pd.DataFrame:
        """Return the DataFrame containing metrics for each run."""
        return self.per_run_df

    def get_summary_df(self) -> pd.DataFrame:
        """Return the DataFrame containing the mean and std for the model."""
        return self.summary_df


FEATURES = {
    5: "Size",
    6: "GC-content",
    7: "ATAC",
    8: "CNV",
    9: "CpG methylation",
    10: "CTCF",
    11: "DNase",
    12: "H3K27ac",
    13: "H3K27me3",
    14: "H3K36me3",
    15: "H3K4me1",
    16: "H3K4me2",
    17: "H3K4me3",
    18: "H3K79me2",
    19: "H3K9ac",
    20: "H3K9me3",
    21: "Indels",
    22: "LINE",
    23: "Long terminal repeats",
    24: "Microsatellites",
    25: "PhastCons",
    26: "POLR2A",
    27: "PolyA sites",
    28: "RAD21",
    29: "RBP binding sites",
    30: "Recombination rate",
    31: "Rep G1b",
    32: "Rep G2",
    33: "Rep S1",
    34: "Rep S2",
    35: "Rep S3",
    36: "Rep S4",
    37: "RNA repeat",
    38: "Simple repeats",
    39: "SINE",
    40: "SMC3",
    41: "SNP",
}


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
        combined_metrics_df.groupby("Model_Group")["Final_Test_Pearson"]
        .mean()
        .reset_index()
    )

    # get best and worst models based on mean correlations
    best_model = performance_grouped_df.loc[
        performance_grouped_df["Final_Test_Pearson"].idxmax(), "Model_Group"
    ]
    worst_model = performance_grouped_df.loc[
        performance_grouped_df["Final_Test_Pearson"].idxmin(), "Model_Group"
    ]

    data_best = combined_metrics_df[combined_metrics_df["Model_Group"] == best_model][
        "Final_Test_Pearson"
    ].values
    data_worst = combined_metrics_df[combined_metrics_df["Model_Group"] == worst_model][
        "Final_Test_Pearson"
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
    randomized_node_features = {
        "k562_release_replicate_3": "Baseline",
        "k562_release_randomize_all_feats": "All features randomized",
        "k562_release_randomize_node_5": "Size randomized",
        "k562_release_randomize_node_6": "GC-content randomized",
        "k562_release_randomize_node_7": "ATAC randomized",
        "k562_release_randomize_node_8": "CNV randomized",
        "k562_release_randomize_node_9": "CpG methylation randomized",
        "k562_release_randomize_node_10": "CTCF randomized",
        "k562_release_randomize_node_11": "DNase randomized",
        "k562_release_randomize_node_12": "H3K27ac randomized",
        "k562_release_randomize_node_13": "H3K27me3 randomized",
        "k562_release_randomize_node_14": "H3K36me3 randomized",
        "k562_release_randomize_node_15": "H3K4me1 randomized",
        "k562_release_randomize_node_16": "H3K4me2 randomized",
        "k562_release_randomize_node_17": "H3K4me3 randomized",
        "k562_release_randomize_node_18": "H3K79me2 randomized",
        "k562_release_randomize_node_19": "H3K9ac randomized",
        "k562_release_randomize_node_20": "H3K9me3 randomized",
        "k562_release_randomize_node_21": "Indels randomized",
        "k562_release_randomize_node_22": "LINE randomized",
        "k562_release_randomize_node_23": "Long terminal repeats randomized",
        "k562_release_randomize_node_24": "Microsatellites randomized",
        "k562_release_randomize_node_25": "PhastCons randomized",
        "k562_release_randomize_node_26": "POLR2A randomized",
        "k562_release_randomize_node_27": "PolyA sites randomized",
        "k562_release_randomize_node_28": "RAD21 randomized",
        "k562_release_randomize_node_29": "RBP binding sites randomized",
        "k562_release_randomize_node_30": "Recombination rate randomized",
        "k562_release_randomize_node_31": "Rep G1b randomized",
        "k562_release_randomize_node_32": "Rep G2 randomized",
        "k562_release_randomize_node_33": "Rep S1 randomized",
        "k562_release_randomize_node_34": "Rep S2 randomized",
        "k562_release_randomize_node_35": "Rep S3 randomized",
        "k562_release_randomize_node_36": "Rep S4 randomized",
        "k562_release_randomize_node_37": "RNA repeat randomized",
        "k562_release_randomize_node_38": "Simple repeats randomized",
        "k562_release_randomize_node_39": "SINE randomized",
        "k562_release_randomize_node_40": "SMC3 randomized",
        "k562_release_randomize_node_41": "SNP randomized",
    }

    base_model_dir = Path("/Users/steveho/gnn_plots/figure_2/model_performance")

    model_categories = [
        (
            "Randomized Node Features",
            randomized_node_features,
            "randomized_node_features",
        ),
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

    # Get all models including the all-features-randomized model
    full_plot_df = performance_df.copy()

    # Sort by performance (from highest to lowest) but keep all-random for the end
    non_random_df = full_plot_df[
        full_plot_df["Model"] != "k562_release_randomize_all_feats"
    ].copy()
    all_random_df = full_plot_df[
        full_plot_df["Model"] == "k562_release_randomize_all_feats"
    ].copy()

    # Sort the non-random models
    non_random_df = non_random_df.sort_values(
        "Final_Test_Pearson_Mean", ascending=False
    )

    # Combine the sorted non-random with the all-random at the end
    plot_df = pd.concat([non_random_df, all_random_df])

    # Create a color array, default is steelblue for all bars
    colors = ["#db95a5"] * len(plot_df)

    # Find the position of the baseline model after sorting
    baseline_position = plot_df.index[
        plot_df["Model"] == "k562_release_replicate_3"
    ].tolist()
    if baseline_position:
        # Get the position in the sorted dataframe
        baseline_position_idx = plot_df.index.get_loc(baseline_position[0])
        # Set the color for the baseline model
        colors[baseline_position_idx] = "#76c76d"

    # Find the position of the all-random model
    all_random_position = plot_df.index[
        plot_df["Model"] == "k562_release_randomize_all_feats"
    ].tolist()
    if all_random_position:
        # Get the position in the sorted dataframe
        all_random_position_idx = plot_df.index.get_loc(all_random_position[0])
        # Set the color for the all-random model
        colors[all_random_position_idx] = "darkgreen"

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 3))

    # Calculate asymmetric errors for confidence intervals
    # For lower bound: mean - lower_CI
    # For upper bound: upper_CI - mean
    yerr = np.zeros((2, len(plot_df)))

    for i, idx in enumerate(plot_df.index):
        # Check if this is the baseline model (which has different column names)
        if plot_df.loc[idx, "Model"] == "k562_release_replicate_3":
            # For baseline, use "CI lower test" and "CI upper test" columns
            if (
                "CI_lower_test_mean" in plot_df.columns
                and "CI_upper_test_mean" in plot_df.columns
            ):
                if not pd.isna(plot_df.loc[idx, "CI_lower_test_mean"]) and not pd.isna(
                    plot_df.loc[idx, "CI_upper_test_mean"]
                ):
                    yerr[0, i] = (
                        plot_df.loc[idx, "Final_Test_Pearson_Mean"]
                        - plot_df.loc[idx, "CI_lower_test_mean"]
                    )
                    yerr[1, i] = (
                        plot_df.loc[idx, "CI_upper_test_mean"]
                        - plot_df.loc[idx, "Final_Test_Pearson_Mean"]
                    )
                else:
                    # If CIs not available, use standard deviation as fallback
                    yerr[0, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]
                    yerr[1, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]
            else:
                # If columns don't exist, use standard deviation
                yerr[0, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]
                yerr[1, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]
        else:
            # For other models, use "CI_Lower_Mean" and "CI_Upper_Mean" columns
            if (
                "CI_Lower_Mean" in plot_df.columns
                and "CI_Upper_Mean" in plot_df.columns
            ):
                if not pd.isna(plot_df.loc[idx, "CI_Lower_Mean"]) and not pd.isna(
                    plot_df.loc[idx, "CI_Upper_Mean"]
                ):
                    yerr[0, i] = (
                        plot_df.loc[idx, "Final_Test_Pearson_Mean"]
                        - plot_df.loc[idx, "CI_Lower_Mean"]
                    )
                    yerr[1, i] = (
                        plot_df.loc[idx, "CI_Upper_Mean"]
                        - plot_df.loc[idx, "Final_Test_Pearson_Mean"]
                    )
                else:
                    # If CIs not available, use standard deviation as fallback
                    yerr[0, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]
                    yerr[1, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]
            else:
                # If columns don't exist, use standard deviation
                yerr[0, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]
                yerr[1, i] = plot_df.loc[idx, "Final_Test_Pearson_Std"]

    # Plot vertical bars with Pearson correlation values and asymmetric error bars
    bars = ax.bar(
        range(len(plot_df)),
        plot_df["Final_Test_Pearson_Mean"],
        color=colors,
        width=0.7,
        yerr=yerr,
        error_kw=dict(ecolor="black", capsize=2, capthick=0.5, elinewidth=0.5),
    )

    # Set x-tick positions and labels
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["Model_Group"], rotation=90, fontsize=7)

    # Customize appearance
    ax.set_ylabel("Predicted vs actual expression\nPearson correlation", fontsize=7)

    # Set y-axis limits to maintain the same range as the current plot
    y_min = 0.875
    y_max = 0.895

    # Set axis limits to keep the same range as your current plot
    ax.set_ylim(y_min, y_max)

    # Get the value for all features randomized
    all_random_value = plot_df.loc[
        plot_df["Model"] == "k562_release_randomize_all_feats",
        "Final_Test_Pearson_Mean",
    ].values[0]

    # Get CI bounds for annotation
    all_random_idx = plot_df.index[
        plot_df["Model"] == "k562_release_randomize_all_feats"
    ][0]
    if "CI_Lower_Mean" in plot_df.columns and "CI_Upper_Mean" in plot_df.columns:
        if not pd.isna(plot_df.loc[all_random_idx, "CI_Lower_Mean"]) and not pd.isna(
            plot_df.loc[all_random_idx, "CI_Upper_Mean"]
        ):
            all_random_ci_lower = plot_df.loc[all_random_idx, "CI_Lower_Mean"]
            all_random_ci_upper = plot_df.loc[all_random_idx, "CI_Upper_Mean"]
            ci_annotation = f"$\it{{r}}$ = {all_random_value:.3f}"
        else:
            all_random_std = plot_df.loc[all_random_idx, "Final_Test_Pearson_Std"]
            if not pd.isna(all_random_std):
                ci_annotation = f"$\it{{r}}$ = {all_random_value:.3f}"
            else:
                ci_annotation = f"$\it{{r}}$ = {all_random_value:.3f}"
    else:
        all_random_std = plot_df.loc[all_random_idx, "Final_Test_Pearson_Std"]
        if not pd.isna(all_random_std):
            ci_annotation = f"$\it{{r}}$ = {all_random_value:.3f}"
        else:
            ci_annotation = f"$\it{{r}}$ = {all_random_value:.3f}"

    # Add annotation for the all features randomized value
    if len(all_random_position) > 0:
        # The position of the all-random bar
        x_pos = len(plot_df) - 1

        # Add an annotation with an arrow pointing to the all-random bar
        ax.annotate(
            ci_annotation,
            xy=(x_pos, y_min),
            xytext=(x_pos, y_min + 0.0055),
            fontsize=7,
            ha="center",
            va="bottom",
            arrowprops=dict(arrowstyle="->", linewidth=0.5),
        )

    # Remove grid
    ax.grid(False)

    # Adjust tick parameters
    ax.tick_params(axis="both", which="major", labelsize=7, width=0.5)
    ax.tick_params(axis="both", which="minor", labelsize=7, width=0.5)

    # Adjust spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # decrease space between bar and y axis
    plt.margins(x=0.01)

    # Tight layout to make sure everything fits
    plt.tight_layout()

    # Save the figure with high resolution
    plt.savefig("feature_performance_plot_with_ci.png", dpi=450, bbox_inches="tight")


if __name__ == "__main__":
    main()
