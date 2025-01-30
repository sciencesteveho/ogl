#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Statistically compare performance metrics across multiple models of varying
graph construction, architecture, and hyperparameters."""


import json
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import pingouin as pg  # type: ignore
from scipy.stats import f_oneway  # type: ignore
from scipy.stats import ttest_rel  # type: ignore
from scipy.stats import wilcoxon  # type: ignore
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
