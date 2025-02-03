#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure 3.

    1. Produces expected vs predicted plots for each sample.
    2. Produces expected vs predicted facet plot.
    3. Produces expected vs predicted correlation heatmap, where expected and
       predicted are measured as difference from the average expression.
"""


import csv
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

import matplotlib  # type: ignore
from matplotlib.collections import PolyCollection  # type: ignore
from matplotlib.colors import LogNorm  # type: ignore
import matplotlib.colors as mcolors
from matplotlib.figure import Figure  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from numpy import sqrt
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

from omics_graph_learning.interpret.interpret_utils import get_gene_idx_mapping
from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.interpret.interpret_utils import map_symbol
from omics_graph_learning.visualization import set_matplotlib_publication_parameters
from omics_graph_learning.visualization.training import plot_predicted_versus_expected


def _gtf_gene_chr_pairing(
    gtf: str,
) -> Dict[str, str]:
    """Get gene: chromosome dict from a gencode gtf file."""
    with open(gtf, newline="") as file:
        return {
            line[3]: line[0]
            for line in csv.reader(file, delimiter="\t")
            if line[0] not in ["chrX", "chrY", "chrM"]
        }


def _load_expression_data(base_path: Union[str, Path]) -> pd.DataFrame:
    """Loads predicted and expected data from multiple directories.

    Args:
        base_path: The base path containing the subdirectories.

    Returns:
        A DataFrame with 'predicted', 'expected', and 'plot_id' columns.
    """
    base_path = Path(base_path)
    data = []
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            try:
                predicted = np.load(subdir / "outs.npy")
                expected = np.load(subdir / "labels.npy")

                # Flatten the arrays if they are not 1D
                predicted = predicted.flatten()
                expected = expected.flatten()

                data.append(
                    pd.DataFrame(
                        {
                            "predicted": predicted,
                            "expected": expected,
                            "plot_id": subdir.name,  # Use subdirectory name as plot_id
                        }
                    )
                )
            except FileNotFoundError:
                print(f"Skipping {subdir} due to missing files.")
    return pd.concat(data, ignore_index=True)


def create_correlation_heatmap(
    df_all: pd.DataFrame,
    experiments_map: Dict[str, str],
    figsize: Tuple[float, float] = (3.0, 2.65),
) -> None:
    """Create correlation heatmap by taking the correlation between the
    predicted and observed expression as the difference from the average
    expression.
    """
    set_matplotlib_publication_parameters()
    colors = [
        "#08306b",
        "#083c9c",
        "#08519c",
        "#3182bd",
        "#6b94d6",
        "#6baed6",
        "#e6e6e6",
        "#fc9179",
        "#fb6a4a",
        "#fb504a",
        "#db2727",
        "#9c0505",
        "#99000d",
    ]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=100)
    experiments = list(experiments_map.keys())

    df_filtered = df_all[df_all["tissue"].isin(experiments)]

    # get common genes across samples
    gene_counts = df_filtered.groupby("gene_symbol")["tissue"].nunique()
    common_genes = gene_counts[gene_counts == len(experiments)].index
    df_filtered = df_filtered[df_filtered["gene_symbol"].isin(common_genes)]

    # calculate tissue average for each gene (in log2 space)
    gene_means_pred = df_filtered.groupby("gene_symbol")["prediction"].mean()
    gene_means_label = df_filtered.groupby("gene_symbol")["label"].mean()

    # merge the tissue averages with the dataframe
    df_filtered = df_filtered.merge(
        gene_means_pred.to_frame("mean_prediction"),
        left_on="gene_symbol",
        right_index=True,
    )
    df_filtered = df_filtered.merge(
        gene_means_label.to_frame("mean_label"), left_on="gene_symbol", right_index=True
    )

    # calculate fold changes
    df_filtered["prediction_fc"] = (
        df_filtered["prediction"] - df_filtered["mean_prediction"]
    )
    df_filtered["label_fc"] = df_filtered["label"] - df_filtered["mean_label"]

    # calculate correlation matrix
    correlation_matrix = np.zeros((len(experiments), len(experiments)))
    for i, tissue1 in enumerate(experiments):
        data1 = df_filtered[df_filtered["tissue"] == tissue1]
        for j, tissue2 in enumerate(experiments):
            if i == j:  # diagonal elements
                correlation_matrix[i, j] = np.corrcoef(
                    data1["prediction_fc"], data1["label_fc"]
                )[0, 1]
            else:  # off-diagonal elements
                data2 = df_filtered[df_filtered["tissue"] == tissue2]
                merged = pd.merge(
                    data1[["gene_symbol", "prediction_fc"]],
                    data2[["gene_symbol", "label_fc"]],
                    on="gene_symbol",
                )
                correlation_matrix[i, j] = np.corrcoef(
                    merged["prediction_fc"], merged["label_fc"]
                )[0, 1]

    # get display names in order
    correlation_matrix = correlation_matrix[:, ::-1]
    experiments = list(experiments_map.keys())
    display_names = [experiments_map[tissue] for tissue in experiments]

    # create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        xticklabels=display_names[::-1],
        yticklabels=display_names,
        cmap=custom_cmap,
        vmin=-0.4,
        vmax=0.4,
        linewidths=0.1,
        linecolor="white",
        square=True,
        center=0,
        cbar_kws={
            "shrink": 0.2,
            "aspect": 4.5,
            "label": "Correlation",
            "orientation": "vertical",
        },
    )

    plt.xlabel("Predicted cell-type specific expression")
    plt.ylabel("Observed cell-type specific expression")

    plt.gcf().axes[-1].tick_params(size=0)
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("Correlation", rotation=0, ha="left", va="center")
    cbar.yaxis.set_ticks([-0.4, 0, 0.4])
    cbar.yaxis.set_ticklabels(["-0.4", "0", "0.4"])
    cbar.yaxis.set_label_coords(-0.5, 1.35)
    plt.tick_params(axis="both", which="both", length=0)
    plt.xticks(rotation=90, ha="right")
    plt.tick_params(axis="x", which="major", pad=2.5)
    plt.yticks(rotation=0)

    plt.tight_layout()
    print(
        f"Number of genes present in all {len(experiments)} tissues: {len(common_genes)}"
    )
    plt.savefig("correlation_heatmap.png", dpi=450)


def plot_facet(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]],
    exclude_samples: Optional[List[str]] = None,
) -> None:
    """Plots predicted versus expected values for multiple experiments in a
    facet grid.
    """
    set_matplotlib_publication_parameters()

    titles = {
        "adrenal_release": "Adrenal",
        "aorta_release": "Aorta",
        "gm12878_release": "GM12878",
        "h1_esc_release": "H1-hESC",
        "hepg2_release": "HepG2",
        "hippocampus_release": "Hippocampus",
        "hmec_release": "HMEC",
        "imr90_release": "IMR90",
        "left_ventricle_release": "Left ventricle",
        "liver_release": "Liver",
        "lung_release": "Lung",
        "mammary_release": "Mammary",
        "nhek_release": "NHEK",
        "ovary_release": "Ovary",
        "pancreas_release": "Pancreas",
        "skeletal_muscle_release": "Skeletal muscle",
        "skin_release": "Skin",
        "small_intestine_release": "Small intestine",
        "spleen_release": "Spleen",
        "cell_lines_release": "Cell lines (combined)",
    }

    # filter out excluded samples if specified
    if exclude_samples:
        df = df[~df["plot_id"].isin(exclude_samples)]

    graph = sns.FacetGrid(
        df,
        col="plot_id",
        col_wrap=5,
        col_order=titles,
        height=1.15,
        aspect=0.95,
        sharex=False,
        sharey=False,
    )

    def plot_hexbin(x: np.ndarray, y: np.ndarray, **kwargs) -> PolyCollection:
        """Plot hexbin with log scale and linear regression line."""
        hb = plt.hexbin(
            x,
            y,
            gridsize=30,
            cmap="viridis",
            norm=LogNorm(),
            edgecolors="white",
            linewidths=0.025,
            mincnt=1,
            label="",
        )

        #  store current axis limits
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()

        # calculate and plot linear regression line
        slope, intercept, *_ = stats.linregress(x, y)
        x_fit = np.linspace(np.min(x) - 0.5, np.max(x) + 0.5, 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, color="indianred", linewidth=0.9, linestyle="--")

        # restore axis limits
        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)

        # add box and remove ticks
        plt.gca().spines["top"].set_visible(True)
        plt.gca().spines["right"].set_visible(True)
        plt.tick_params(axis="both", which="both", length=0)

        return hb

    # map hb to each facet
    graph.map(plot_hexbin, "predicted", "expected")

    # remove titles from subplots
    for ax, name in zip(graph.axes.flat, graph.col_names):

        # get data for this subplot
        subplot_data = df[df["plot_id"] == name]
        pearson_r = stats.pearsonr(subplot_data["predicted"], subplot_data["expected"])[
            0
        ]

        # add correlation to title
        display_name = titles.get(name, name)
        ax.set_title(
            f"{display_name}\n" + r"$\mathit{r}$ = " + f"{pearson_r:.4f}",
            size=7,
            y=0.98,
        )

    graph.figure.text(
        0.5125, -0.01, "Predicted Log2 Expression", ha="center", va="center", size=7
    )
    graph.figure.text(
        -0.01,
        0.5,
        "Expected Log2 Expression",
        ha="center",
        va="center",
        rotation=90,
        size=7,
    )
    graph.set_axis_labels("", "")
    graph.figure.subplots_adjust(right=0.925)
    cbar_ax = graph.figure.add_axes([1.025, 0.4, 0.03, 0.15])
    graph.figure.colorbar(graph.axes[0].collections[0], cax=cbar_ax)
    plt.tight_layout()

    if save_path:
        graph.savefig(
            f"{save_path}/performance_facet.png", dpi=450, bbox_inches="tight"
        )


def main() -> None:
    """Main function."""
    working_dir = "/Users/steveho/gnn_plots"
    gencode_file = (
        f"{working_dir}/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"
    )

    # load gencode to symbol mapping
    symbol_to_gencode = load_gencode_lookup(gencode_file)
    gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

    # load chr to gene mapping
    gtf_file = f"{working_dir}/graph_resources/local/gencode_v26_genes_only_with_GTEx_targets.bed"
    ensg_to_chr = _gtf_gene_chr_pairing(gtf_file)

    # change to symbol to chr
    symbol_to_chr = {
        gencode_to_symbol[k]: v
        for k, v in ensg_to_chr.items()
        if k in gencode_to_symbol
    }

    # get a list of genes in chr8 and chr9
    test_chrs = ["chr8", "chr9"]
    test_genes = [gene for gene, chr in symbol_to_chr.items() if chr in test_chrs]

    # make a dataframe to combine all the results
    df_all = pd.DataFrame()

    experiments = [
        "adrenal_release",
        "aorta_release",
        "gm12878_release",
        "h1_esc_release",
        "hepg2_release",
        "hippocampus_release",
        "hmec_release",
        "imr90_release",
        "k562_release",
        "left_ventricle_release",
        "liver_release",
        "lung_release",
        "mammary_release",
        "nhek_release",
        "ovary_release",
        "pancreas_release",
        "skeletal_muscle_release",
        "skin_release",
        "small_intestine_release",
        "spleen_release",
        "cell_lines_release",
    ]

    for experiment in experiments:
        # plot the expected vs predicted values for individual experiments
        model_out_dir = f"{working_dir}/figure_3/{experiment}"
        predicted = np.load(f"{model_out_dir}/outs.npy")
        expected = np.load(f"{model_out_dir}/labels.npy")
        rmse = sqrt(mean_squared_error(expected, predicted))

        # plot the expected vs predicted values for individual experiments
        plot_predicted_versus_expected(
            predicted=predicted,
            expected=expected,
            rmse=rmse,
            save_path=Path(model_out_dir),
            title=True,
        )

        if experiment != "cell_lines_release":
            # first, make a collective df
            graph_data_dir = f"{working_dir}/interpretation/{experiment}"
            idx_file = (
                f"{working_dir}/graph_resources/idxs/{experiment}_full_graph_idxs.pkl"
            )

            # load graph idxs
            with open(idx_file, "rb") as f:
                idxs = pickle.load(f)

            node_idx_to_gene_id, gene_indices = get_gene_idx_mapping(idxs)

            # get baseline predictions
            df = pd.read_csv(f"{graph_data_dir}/baseline_predictions_2_hop.csv")
            df["gene_id"] = df["node_idx"].map(node_idx_to_gene_id)
            df["gene_symbol"] = df["gene_id"].apply(
                lambda g: map_symbol(g, gencode_to_symbol=gencode_to_symbol)
            )
            df_test = df[df["gene_symbol"].isin(test_genes)]

            # add a column for the tissue name
            tissue_name = "_".join(experiment.split("_")[:-1])
            df_test["tissue"] = tissue_name

            # add df to the collective df
            df_all = pd.concat([df_all, df_test])

    # clear memory
    plt.clf()

    # create correlation heatmap
    experiments_corr = {
        "small_intestine": "Small intestine",
        "spleen": "Spleen",
        "skin": "Skin",
        "skeletal_muscle": "Skeletal muscle",
        "pancreas": "Pancreas",
        "ovary": "Ovary",
        "lung": "Lung",
        "left_ventricle": "Left ventricle",
        "hippocampus": "Hippocampus",
        "aorta": "Aorta",
        "adrenal": "Adrenal",
        "liver": "Liver",
        "mammary": "Mammary",
        "imr90": "IMR90",
        "nhek": "NHEK",
        "k562": "K562",
        "hmec": "HMEC",
        "hepg2": "HepG2",
        "h1_esc": "H1-hESC",
        "gm12878": "GM12878",
    }

    create_correlation_heatmap(df_all, experiments_corr, figsize=(3.5, 3.2))

    # plot expected vs predict facet
    base_path = Path(".")
    results_df = _load_expression_data(base_path)
    plot_facet(results_df, exclude_samples=["k562_release"], save_path=base_path)


if __name__ == "__main__":
    main()
