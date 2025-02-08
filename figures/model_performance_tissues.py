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


def get_metric_dataframes(models: Dict[str, str], base_model_dir: Path) -> pd.DataFrame:
    """Get per-run and summary dataframes for a dictionary of models, adding a
    'Model_Group' column with readable names.
    """
    per_run_dfs = []

    for model_key, readable_name in models.items():
        model_path = base_model_dir / model_key
        if not model_path.exists():
            print(f"Warning: Model path {model_path} does not exist. Skipping.")
            continue

        metrics_obj = ModelMetrics(model_path)
        per_run_df = metrics_obj.get_per_run_df()
        per_run_df["Model_Group"] = readable_name

        per_run_dfs.append(per_run_df)

    return pd.concat(per_run_dfs, ignore_index=True)


def main() -> None:
    """Generate figures for iterative model construction."""

    release_models = {
        "adrenal_release": "Adrenal",
        "aorta_release": "Aorta",
        "gm12878_release": "GM12878",
        "h1_esc_release": "H1-hESC",
        "hepg2_release": "HepG2",
        "hippocampus_release": "Hippocampus",
        "hmec_release": "HMEC",
        "imr90_release": "IMR90",
        "k562_release": "K562",
        "left_ventricle_release": "Left Ventricle",
        "liver_release": "Liver",
        "lung_release": "Lung",
        "mammary_release": "Mammary",
        "nhek_release": "NHEK",
        "ovary_release": "Ovary",
        "pancreas_release": "Pancreas",
        "skeletal_muscle_release": "Skeletal Muscle",
        "skin_release": "Skin",
        "small_intestine_release": "Small Intestine",
        "spleen_release": "Spleen",
        "cell_lines_release": "Cell Lines",
    }

    base_model_dir = Path("/Users/steveho/gnn_plots/figure_2/model_performance")

    metrics_df = get_metric_dataframes(release_models, base_model_dir)


if __name__ == "__main__":
    main()
