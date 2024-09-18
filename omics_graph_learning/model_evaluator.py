#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Statistically compare performance metrics across multiple models of varying
graph construction, architecture, and hyperparameters."""


import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import pingouin as pg  # type: ignore
from scipy.stats import ttest_rel  # type: ignore
from scipy.stats import wilcoxon  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.utils import resample  # type: ignore
import statsmodels.api as sm  # type: ignore
from statsmodels.stats.anova import AnovaRM  # type: ignore


class ModelEvaluator:
    """Class to model comparisons.

    Attributes:
        model_names: list of strings, names of the models.
        performances: dict, keys are model names, values are lists or arrays of
        performance metrics.

    Methods
    --------
    compute_statistics()
        Compute mean, standard deviation, and 95% confidence intervals via
        bootstrapping.
    bootstrap_ci(data, num_bootstrap=10000, ci=95)
        Compute bootstrap confidence intervals.
    perform_statistical_tests()
        Perform paired t-tests, Wilcoxon tests, calculate effect sizes, and
        repeated measures ANOVA.
    compute_cohens_d(x, y)
        Compute Cohen's d for paired samples.
    perform_repeated_measures_anova()
        Perform repeated measures ANOVA and post-hoc tests.
    visualize_results()
        Generate performance plots and heatmaps of statistical comparisons.
    report_results()
        Print statistical summaries and ANOVA results.

    Examples:
    --------
    >>> pass
    """

    def __init__(self, model_names, performances):
        """Initialize the ModelComparer with model names and their performance
        metrics.
        """
        self.model_names = model_names
        self.performances = performances
        self.stats = {}
        self.pairwise_results = {}
        self.compute_statistics()
        self.perform_statistical_tests()

    def compute_statistics(self) -> None:
        """Compute test set mean and standard deviation for each model."""
        self.means = {}
        self.stds = {}
        for model in self.model_names:
            data = np.array(self.performances[model])
            self.means[model] = np.mean(data)
            self.stds[model] = np.std(data, ddof=1)
            self.stats[model] = {
                "mean": self.means[model],
                "std": self.stds[model],
            }

    def load_bootstrap_ci(self) -> None:
        """Load precomputed bootstrap confidence intervals from model post
        evaluation.
        """
        self.cis = {}
        for model in self.model_names:
            # Assuming that each run has its own CI, compute the aggregate CI per model
            # For simplicity, we'll average the CIs across runs
            ci_lower_list = []
            ci_upper_list = []
            for run_number in range(1, 4):  # Assuming 3 runs
                ci_file = (
                    experiment_dir / model / f"run_{run_number}" / "eval_metrics.json"
                )
                if ci_file.exists():
                    with open(ci_file, "r") as f:
                        metrics = json.load(f)
                        # Assuming 'CI lower' and 'CI upper' are stored
                        ci_lower_list.append(metrics.get("CI lower", np.nan))
                        ci_upper_list.append(metrics.get("CI upper", np.nan))
                else:
                    print(f"CI file not found: {ci_file}")
            # Compute mean CI lower and upper
            mean_ci_lower = np.nanmean(ci_lower_list)
            mean_ci_upper = np.nanmean(ci_upper_list)
            self.cis[model] = (mean_ci_lower, mean_ci_upper)

    def perform_statistical_tests(self):
        """Perform paired t-tests, Wilcoxon tests, calculate effect sizes, and repeated measures ANOVA."""
        models = self.model_names
        n_models = len(models)
        self.p_values_ttest = np.full((n_models, n_models), np.nan)
        self.p_values_wilcoxon = np.full((n_models, n_models), np.nan)
        self.effect_sizes = np.full((n_models, n_models), np.nan)

        for i in range(n_models):
            for j in range(i + 1, n_models):
                model_i = models[i]
                model_j = models[j]
                data_i = np.array(self.performances[model_i])
                data_j = np.array(self.performances[model_j])

                # Paired t-test
                t_stat, p_val_ttest = ttest_rel(data_i, data_j)
                self.p_values_ttest[i, j] = p_val_ttest
                self.p_values_ttest[j, i] = p_val_ttest  # Symmetric

                # Wilcoxon signed-rank test
                try:
                    w_stat, p_val_wilcoxon = wilcoxon(data_i, data_j)
                except ValueError:
                    # Handle the case where all differences are zero
                    p_val_wilcoxon = 1.0
                self.p_values_wilcoxon[i, j] = p_val_wilcoxon
                self.p_values_wilcoxon[j, i] = p_val_wilcoxon  # Symmetric

                # Effect size (Cohen's d)
                cohen_d = self.compute_cohens_d(data_i, data_j)
                self.effect_sizes[i, j] = cohen_d
                self.effect_sizes[j, i] = cohen_d  # Symmetric

        # Repeated measures ANOVA
        self.perform_repeated_measures_anova()

    def compute_cohens_d(self, x, y):
        """Compute Cohen's d for paired samples."""
        diff = x - y
        n = len(diff)
        sd_diff = np.std(diff, ddof=1)
        mean_diff = np.mean(diff)
        cohen_d = mean_diff / sd_diff if sd_diff != 0 else 0.0
        return cohen_d

    def perform_repeated_measures_anova(self):
        """Perform repeated measures ANOVA and post-hoc tests."""
        # Prepare data
        subjects = np.arange(len(next(iter(self.performances.values()))))
        data = []
        for model in self.model_names:
            for i, perf in enumerate(self.performances[model]):
                data.append(
                    {"Subject": subjects[i], "Model": model, "Performance": perf}
                )
        df = pd.DataFrame(data)
        aovrm = AnovaRM(df, "Performance", "Subject", within=["Model"])
        res = aovrm.fit()
        self.anova_results = res

        # Post-hoc tests with Bonferroni correction
        self.posthoc_results = pg.pairwise_ttests(
            dv="Performance", within="Model", subject="Subject", data=df, padjust="bonf"
        )

    def visualize_results(self):
        """Generate performance plots and heatmaps of statistical comparisons."""
        # Performance plots with error bars
        models = self.model_names
        means = [self.means[model] for model in models]
        errors = [self.means[model] - self.cis[model][0] for model in models]

        plt.figure(figsize=(10, 6))
        plt.bar(models, means, yerr=errors, capsize=5, color="skyblue")
        plt.ylabel("Performance Metric (e.g., RMSE)")
        plt.title("Model Performance with 95% Confidence Intervals")
        plt.tight_layout()
        plt.show()

        # Heatmap of p-values (Paired t-test)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.p_values_ttest,
            annot=True,
            fmt=".3f",
            xticklabels=models,
            yticklabels=models,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
        )
        plt.title("Pairwise p-values (Paired t-test)")
        plt.tight_layout()
        plt.show()

        # Heatmap of Effect Sizes
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.effect_sizes,
            annot=True,
            fmt=".3f",
            xticklabels=models,
            yticklabels=models,
            cmap="viridis",
        )
        plt.title("Pairwise Effect Sizes (Cohen's d)")
        plt.tight_layout()
        plt.show()

    def report_results(self):
        """Print statistical summaries and ANOVA results."""
        # Print mean, std, and confidence intervals
        print("Model Performance Metrics:")
        for model in self.model_names:
            mean = self.means[model]
            std = self.stds[model]
            ci = self.cis[model]
            print(
                f"{model}: Mean = {mean:.4f}, Std = {std:.4f}, 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]"
            )

        # Print ANOVA results
        print("\nRepeated Measures ANOVA Results:")
        print(self.anova_results)

        # Print post-hoc test results
        print("\nPost-hoc Tests (Bonferroni corrected):")
        print(
            self.posthoc_results[
                ["A", "B", "T", "dof", "p-unc", "p-corr", "hedges"]
            ].to_string(index=False)
        )


def collect_performance_data(
    experiment_dir: Path, model_names: List[str], num_runs: int = 3
) -> Dict[str, List[float]]:
    """
    Collect performance data from the model training runs.

    Parameters:
    - experiment_dir: Path to the directory containing the experiment results.
    - model_names: List of model names.
    - num_runs: Number of runs (random seeds).

    Returns:
    - performances: dict, keys are model names, values are lists of performance metrics.
    """
    performances = {}
    for model_name in model_names:
        model_performances = []
        for run_number in range(1, num_runs + 1):
            metrics_file = (
                experiment_dir / model_name / f"run_{run_number}" / "eval_metrics.json"
            )
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    # Assuming 'Final test RMSE' is the performance metric
                    model_performances.append(metrics["Final test RMSE"])
            else:
                print(f"Metrics file not found: {metrics_file}")
        performances[model_name] = model_performances
    return performances
