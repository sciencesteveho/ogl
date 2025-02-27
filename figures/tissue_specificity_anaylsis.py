# sourcery skip: avoid-global-variables, avoid-single-character-names-variables, docstrings-for-modules, no-long-functions, require-parameter-annotation, require-return-annotation
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde  # type: ignore
from scipy.stats import pearsonr  # type: ignore
from scipy.stats import spearmanr
import seaborn as sns  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from omics_graph_learning.visualization import set_matplotlib_publication_parameters


def analyze_tissue_specificity(effects_dict, effect_type="max"):
    """Analyze a tissue-specific dictionary of effects."""
    data = []
    col_name = "Max_Impact" if effect_type == "max" else "Avg_Impact"
    for tissue, elements in effects_dict.items():
        data.extend(
            {
                "Tissue": tissue,
                "Element": element,
                col_name: float(effect_info["fc"]),
                "Target_Gene": effect_info["gene"],
            }
            for element, effect_info in elements.items()
        )
    return pd.DataFrame(data)


def process_effects(effects, impact_col, top_n=1000):
    """Process an effects dictionary to generate a pivot matrix and related
    metrics.
    """
    effect_type = "max" if impact_col == "Max_Impact" else "avg"
    df = analyze_tissue_specificity(effects, effect_type=effect_type)

    # filter out non cCREs
    df = df[~df["Element"].str.contains("ENSG")]
    df = df[~df["Element"].str.contains("tad")]

    # pivot
    matrix = df.pivot_table(
        index="Element", columns="Tissue", values=impact_col, fill_value=0
    )
    matrix_abs = matrix.abs()
    matrix_norm = matrix_abs.divide(matrix_abs.max(axis=1), axis=0)

    # compute tau and element_max
    tau = (1 - matrix_norm).sum(axis=1) / (matrix_norm.shape[1] - 1)
    element_max = matrix_abs.max(axis=1)

    # select top elements
    top_elements = element_max.sort_values(ascending=False).head(top_n).index
    matrix_top = matrix.loc[top_elements].T

    return {
        "df": df,
        "matrix": matrix,
        "matrix_abs": matrix_abs,
        "matrix_norm": matrix_norm,
        "tau": tau,
        "element_max": element_max,
        "matrix_top": matrix_top,
    }


def plot_scatter(
    results,
    x_col,
    y_col,
    filename,
    figsize=(2.75, 1.75),
    correlation_method="spearman",
):
    """Plot a scatter plot with regression line and correlation stats."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x=x_col,
        y=y_col,
        data=results,
        alpha=0.4,
        s=2.5,
        ax=ax,
        linewidth=0,
        color="gray",
    )
    # remove x and y whitespace
    ax.margins(x=0, y=0)

    if correlation_method == "spearman":
        corr, p_value = spearmanr(results[x_col], results[y_col])
    else:
        corr, p_value = pearsonr(results[x_col], results[y_col])
    ax.text(
        0.05,
        0.85,
        rf"Spearman's $\rho$: {corr:.3f}",
        transform=ax.transAxes,
    )

    sns.regplot(
        x=x_col,
        y=y_col,
        data=results,
        scatter=False,
        ax=ax,
        line_kws=dict(
            color="indianred",
            linewidth=0.5,
            linestyle="--",
        ),
    )
    ax.set_xlabel("Tissue specificity (Tau)")
    ax.set_ylabel("Average predicted\n" r"log$_2$ fold change")

    fig.tight_layout()
    fig.savefig(filename, dpi=450)
    plt.close(fig)


def plot_histogram(
    data, column, filename, bins=30, figsize=(2, 2), bandwidth_factor=2.0
):
    """Plot a histogram for the specified column of the provided DataFrame."""
    fig, ax = plt.subplots(figsize=figsize)
    n, bin_edges, patches = ax.hist(
        data[column], bins=bins, alpha=0.5, edgecolor="white", linewidth=0.1
    )
    kde = gaussian_kde(
        data[column], bw_method=bandwidth_factor * gaussian_kde(data[column]).factor
    )

    x_grid = np.linspace(data[column].min(), data[column].max(), 1000)
    kde_values = kde(x_grid)
    scaling_factor = max(n) / max(kde_values)

    ax.plot(x_grid, kde_values * scaling_factor, color="steelblue", linewidth=0.5)

    ax.axvline(data[column].median(), color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Tissue specificity (Tau)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(filename, dpi=450)
    plt.close(fig)


if __name__ == "__main__":
    """Main script to generate tissue specificity analysis plots."""
    set_matplotlib_publication_parameters()

    # load effects
    with open("/Users/steveho/gnn_plots/perturb_effects/avg_effects.pkl", "rb") as f:
        avg_effects = pickle.load(f)
    with open("/Users/steveho/gnn_plots/perturb_effects/max_effects.pkl", "rb") as f:
        max_effects = pickle.load(f)

    max_results = process_effects(max_effects, impact_col="Max_Impact", top_n=1000)
    avg_results = process_effects(avg_effects, impact_col="Avg_Impact", top_n=1000)

    results_max = pd.DataFrame(
        {
            "Element": max_results["tau"].index,
            "Tau": max_results["tau"].values,
            "Max_Impact": max_results["element_max"].values,
        }
    )

    results_avg = pd.DataFrame(
        {
            "Element": avg_results["tau"].index,
            "Tau": avg_results["tau"].values,
            "Avg_Impact": avg_results["element_max"].values,
        }
    )

    # tissue specificity vs impact scatter plot
    scatter_filename = "max_impact_vs_specificity_scatter.png"
    plot_scatter(
        results_max,
        x_col="Tau",
        y_col="Max_Impact",
        filename=scatter_filename,
    )

    # same for avg
    scatter_filename = "avg_impact_vs_specificity_scatter.png"
    plot_scatter(
        results_avg,
        x_col="Tau",
        y_col="Avg_Impact",
        filename=scatter_filename,
    )

    # histogram distribution of Tau for all elements
    histogram_filename = "max_tau_distribution_histogram.png"
    plot_histogram(
        results_max,
        column="Tau",
        filename=histogram_filename,
    )

    # same for avg
    histogram_filename = "avg_tau_distribution_histogram.png"
    plot_histogram(
        results_avg,
        column="Tau",
        filename=histogram_filename,
    )
