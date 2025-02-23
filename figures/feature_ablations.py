#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure node feature ablations.
"""

import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

from gseapy import dotplot  # type: ignore
import gseapy as gp  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore
import matplotlib.colors as mcolors  # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.visualization import set_matplotlib_publication_parameters

TISSUES = {
    "adrenal": "Adrenal",
    "aorta": "Aorta",
    "gm12878": "GM12878",
    "h1_esc": "H1-hESC",
    "hepg2": "HepG2",
    "hippocampus": "Hippocampus",
    "hmec": "HMEC",
    "imr90": "IMR90",
    "k562": "K562",
    "left_ventricle": "Left ventricle",
    "liver": "Liver",
    "lung": "Lung",
    "mammary": "Mammary",
    "nhek": "NHEK",
    "ovary": "Ovary",
    "pancreas": "Pancreas",
    "skeletal_muscle": "Skeletal muscle",
    "skin": "Skin",
    "small_intestine": "Small intestine",
    "spleen": "Spleen",
}


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


def load_node_feature_ablations() -> pd.DataFrame:
    """Load the node feature ablations.

    Returns:
        A dictionary where the keys are tissue names and the values are
        dictionaries where the keys are feature indices and the values are the
        mean fold change in the node feature ablation experiment.
    """
    data_frames = []
    for tissue, tissue_name in TISSUES.items():
        file = f"{tissue}_release/node_feature_perturbations.pkl"
        with open(file, "rb") as f:
            node_feat_ablation = pickle.load(f)
            df_sample = pd.DataFrame(
                node_feat_ablation.values(), index=node_feat_ablation.keys()
            ).T
            df_sample.index = pd.Index([tissue_name])
            data_frames.append(df_sample)

    # concatenate all the dataframes
    # change column idx names to proper feature names
    df = pd.concat(data_frames)
    df.columns = pd.Index([FEATURES.get(int(idx)) for idx in df.columns])
    return df


def load_double_node_feature_ablations() -> pd.DataFrame:
    """Load the node feature ablations for double perturbations."""
    data_frames = []
    for tissue, tissue_name in TISSUES.items():
        file = f"{tissue}_release/node_feature_perturbations_double.pkl"
        with open(file, "rb") as f:
            node_feat_ablation = pickle.load(f)
            df_sample = pd.DataFrame(
                node_feat_ablation.values(), index=node_feat_ablation.keys()
            ).T
            df_sample.index = pd.Index([tissue_name])
            data_frames.append(df_sample)

    # concatenate all the dataframes
    # change column idx names to proper feature names
    df = pd.concat(data_frames)
    df.columns = pd.Index(
        [f"{FEATURES.get(idx[0])} & {FEATURES.get(idx[1])}" for idx in df.columns]
    )
    return df


def plot_node_feature_ablations(
    df: pd.DataFrame, savename: str, log2: bool = True
) -> None:
    """Plot the node features ablations as an annotated heatmap, where the
    heatmap has violin plots to illustrate distribution of fold changes.
    """

    set_matplotlib_publication_parameters()

    # create figure and axes
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)
    ax_violin = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # get different colors for each feature
    n_features = len(df.columns)
    feature_colors = plt.cm.tab20(np.linspace(0, 1, n_features))  # type: ignore

    # melt df into long format
    df_melt = df.reset_index().melt(
        id_vars="index", var_name="Feature", value_name="FoldChange"
    )

    # alphabetical order
    feature_order = list(df.columns)

    sns.violinplot(
        data=df_melt,
        x="Feature",
        y="FoldChange",
        order=feature_order,
        ax=ax_violin,
        cut=0,
        inner="box",
        density_norm="width",
        linewidth=0.5,
        palette=list(feature_colors),
        width=0.7,
        saturation=1.0,
        inner_kws=dict(box_width=2),
        hue="Feature",
        legend=False,
    )

    # set violin labels and ticks
    ax_violin.set_xlabel("")
    ax_violin.set_ylabel("")
    # ax_violin.set_ylabel("Log fold change", va="right")
    ax_violin.set_xticklabels([])
    ax_violin.grid(False)

    # adjust violin plot box
    for spine in ["top", "right", "bottom", "left"]:
        ax_violin.spines[spine].set_linewidth(0.5)
        ax_violin.spines[spine].set_color("black")

    # custom heatmap colors
    colors = [
        "#08306b",
        "#083c9c",
        "#08519c",
        "#3182bd",
        "#7cabf7",
        "#81cffc",
        "#ffffff",
        "#fc9179",
        "#fb6a4a",
        "#fb504a",
        "#db2727",
        "#9c0505",
        "#99000d",
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=100)

    # plot heatmap
    vmax = max(abs(df.max().max()), abs(df.min().min()))
    heatmap_label = r"Log$_2$ fold change" if log2 else "Difference from additivity"
    sns.heatmap(
        df,
        cmap=cmap,
        center=0,
        # vmax=vmax,
        # vmin=-vmax,
        ax=ax_heat,
        cbar_kws={
            "label": heatmap_label,
            "shrink": 0.3,
            "aspect": 6.5,
            "orientation": "vertical",
            "pad": 0.012,
        },
        xticklabels=feature_order,
        linewidths=0.1,
        linecolor="white",
        square=True,
    )

    # create initial labels with proper rotation
    # move them down vertically to avoid overlap with colored squares
    ax_heat.set_xticklabels(feature_order, rotation=90, ha="center", va="top", y=-0.005)

    # add colored squares to the labels
    for label, color in zip(ax_heat.get_xticklabels(), feature_colors):
        text = label.get_text()
        pos = label.get_position()
        label._text = f"{text}"
        ax_heat.text(
            pos[0],
            pos[1] + 0.04,
            "■",
            fontsize=10,
            color=color,
            transform=label.get_transform(),
            rotation=90,
            va=label.get_va(),
            ha=label.get_ha(),
        )

    # align plots
    heatmap_pos = ax_heat.get_position()
    violin_pos = ax_violin.get_position()
    ax_violin.set_position(
        [heatmap_pos.x0, violin_pos.y0, heatmap_pos.width, violin_pos.height]
    )

    # save
    plt.savefig(f"{savename}.png", dpi=450, bbox_inches="tight")
    plt.clf()


def plot_single_enrichment_after_ablation(
    node_ablations: Dict[int, List[Tuple[str, float]]],
    gencode_to_symbol: Dict[str, str],
    idx: int,
    sample: str,
    top_n: int = 100,
) -> None:
    """Get the top n genes after node feature ablation."""
    set_matplotlib_publication_parameters()
    outpath = "/Users/steveho/gnn_plots/interpretation/individual_enrichment"

    gene_sets = [
        "Reactome_Pathways_2024",
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
    ]

    # set up colors
    colors = ["#E6F3FF", "#2171B5"]  # Light to dark
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # get top_n_genes
    top = node_ablations[idx][:top_n]

    # convert to gene symbols
    top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, _ in top]

    # run enrichr
    for gene_set in gene_sets:
        gmt = f"/Users/steveho/gnn_plots/interpretation/gmt/{gene_set}.gmt"
        print(f"Running enrichr for {gmt}...")

        try:
            enrichr_result = gp.enrichr(
                gene_list=top_genes,
                gene_sets=gmt,
                organism="Human",
                outdir=None,
            )
        except AttributeError:
            print(f"Error with {gene_set} for {sample}")
            continue

        # get significant terms
        significant = enrichr_result.results[
            enrichr_result.results["Adjusted P-value"] < 0.05
        ]
        df = significant.copy()
        df["neg_log_p"] = -np.log10(df["Adjusted P-value"])
        # plt.figure(figsize=(5, 1.25))

        # adjust name if GO
        # split the df "Term" column by "(GO:" and take the first part
        if "GO" in gene_set:
            df["Term"] = df["Term"].apply(lambda x: x.split("(GO:")[0])

        top5 = df.nlargest(5, "neg_log_p").copy()
        if len(top5) == 0:
            print(f"No significant terms for {gene_set} in {sample}")
            continue

        top5 = top5.sort_values("neg_log_p", ascending=False)

        color_vals = np.linspace(1, 0, len(top5))
        bar_colors = cmap(color_vals)

        ax = plt.gca()
        plt.barh(
            top5["Term"],
            top5["neg_log_p"],
            color=bar_colors,
            height=0.55,
        )

        # add significance line
        plt.axvline(
            -np.log10(0.05),
            color="black",
            linestyle="--",
            alpha=0.80,
            label="Adjusted p=0.05",
            linewidth=0.5,
            ymin=0,
            ymax=0.875,
        )

        plt.xlabel(r"-log$_{10}$ adjusted $\it{p}$")
        plt.gca().invert_yaxis()
        plt.gca().margins(y=0.15)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.5)

        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", width=0.5)

        plt.tight_layout()
        plt.savefig(f"{outpath}/{sample}_{FEATURES[idx]}_{gene_set}.png", dpi=450)
        plt.clf()


def collate_enrichment_results(
    gencode_to_symbol: Dict[str, str],
    idx: int,
    top_n: int = 100,
) -> None:
    """Get the top-n genes after node feature ablation and run Enrichr.
    Returns three DataFrames (Reactome, GO_BP, GO_MF) with up to top-5
    significant terms per sample (labeled by the 'Sample' column).
    """
    outpath = "/Users/steveho/gnn_plots/interpretation/collate_enrichment"

    # empty dfs to store results
    reactome_df = pd.DataFrame()
    go_bp_df = pd.DataFrame()
    go_mf_df = pd.DataFrame()

    gene_sets = [
        "Reactome_Pathways_2024",
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
    ]

    # run enrichment analysis for each gene set
    for gene_set in gene_sets:
        gmt = f"/Users/steveho/gnn_plots/interpretation/gmt/{gene_set}.gmt"
        print(f"Running enrichr for {gmt}...")

        for tissue, _ in TISSUES.items():
            node_ablations = pickle.load(
                open(f"{tissue}_release/node_feature_top_genes.pkl", "rb")
            )
            top = node_ablations[idx][:top_n]
            top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, _ in top]

            try:
                enrichr_result = gp.enrichr(
                    gene_list=top_genes,
                    gene_sets=gmt,
                    organism="Human",
                    outdir=None,
                )
            except AttributeError:
                print(f"Error with {gene_set} for {tissue}")
                continue

            significant = enrichr_result.results[
                enrichr_result.results["Adjusted P-value"] < 0.05
            ]

            if significant.empty:
                print(f"No significant terms for {gene_set} in {tissue}")
                continue

            df = significant.copy()
            df["neg_log_p"] = -np.log10(df["Adjusted P-value"])

            # remove GO term IDs
            if "GO" in gene_set:
                df["Term"] = df["Term"].apply(lambda x: x.split("(GO:")[0])

            # get top 5 terms
            top5 = df.nlargest(5, "neg_log_p").copy()
            top5.sort_values("neg_log_p", ascending=False, inplace=True)

            # add sample name
            top5["Sample"] = tissue

            # append to final dfs
            if gene_set == "Reactome_Pathways_2024":
                reactome_df = pd.concat([reactome_df, top5], ignore_index=True)
            elif gene_set == "GO_Biological_Process_2023":
                go_bp_df = pd.concat([go_bp_df, top5], ignore_index=True)
            elif gene_set == "GO_Molecular_Function_2023":
                go_mf_df = pd.concat([go_mf_df, top5], ignore_index=True)

    reactome_df.to_csv(f"{outpath}/reactome_{FEATURES[idx]}.csv", index=False)
    go_bp_df.to_csv(f"{outpath}/go_bp_{FEATURES[idx]}.csv", index=False)
    go_mf_df.to_csv(f"{outpath}/go_mf_{FEATURES[idx]}.csv", index=False)


def main() -> None:
    """Main function to generate ablation figures."""
    # make node_ablation heatmap figures
    df = load_node_feature_ablations()
    plot_node_feature_ablations(df, savename="node_feature_ablations")

    double_df = load_double_node_feature_ablations()
    plot_node_feature_ablations(double_df, savename="double_node_feature_ablations")

    # combine df and double df
    df = pd.concat([df, double_df], axis=1)

    # get atac combinations
    # string_matches = ['ATAC', 'H3', 'CpG', 'DNase', 'Micro']
    atac_columns = [col for col in df.columns if "ATAC" in col]
    atac_df = df[atac_columns]
    plot_node_feature_ablations(atac_df, savename="atac_node_feature_ablations")

    # get H3K27ac combinations
    h3k27ac_columns = [
        col for col in df.columns if "H3K27ac" in col and "ATAC" not in col
    ]
    h3k27ac_df = df[h3k27ac_columns]
    plot_node_feature_ablations(h3k27ac_df, savename="h3k27ac_node_feature_ablations")

    # get all histones
    # get H3K27ac combinations
    h3_columns = [col for col in df.columns if "H3" in col and "ATAC" not in col]
    h3_df = df[h3_columns]
    plot_node_feature_ablations(h3_df, savename="h3_node_feature_ablations")

    # additive analysis
    double_cols = [c for c in h3_df.columns if " & " in c]
    synergy_df = df[double_cols]

    for combo_col in double_cols:
        single1, single2 = combo_col.split(" & ")

        # observed combined effect
        observed = h3_df[combo_col]

        # predicted additive effect = sum of each single
        predicted = h3_df[single1] + h3_df[single2]

        # synergy (positive => synergy, negative => antagonism)
        synergy_col_name = combo_col + " (synergy)"
        synergy_df[synergy_col_name] = observed - predicted

    # only keep synergy columns
    synergy_df = synergy_df[[c for c in synergy_df.columns if "synergy" in c]]

    # remove (synergy) from column names
    synergy_df.columns = [c.replace(" (synergy)", "") for c in synergy_df.columns]
    plot_node_feature_ablations(synergy_df, savename="h3_additive", log2=False)

    # plot scatter
    for combo_col in double_cols:
        single1, single2 = combo_col.split(" & ")
        observed = h3_df[combo_col]
        predicted = h3_df[single1] + h3_df[single2]

        # Make a small scatter for each pair, 1 point per tissue
        plt.figure(figsize=(2, 2))
        plt.scatter(predicted, observed, c="dodgerblue", alpha=0.7, s=7.5, linewidth=0)

        # Add identity line for reference
        min_val = min(predicted.min(), observed.min())
        max_val = max(predicted.max(), observed.max())
        plt.plot(
            [min_val, max_val], [min_val, max_val], "k--", linewidth=0.5, c="darkgray"
        )

        # remove whitespace
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.xlabel("Predicted additive effect")
        plt.ylabel("Observed combined effect")
        plt.title(f"{single1} + {single2}")
        plt.tight_layout()
        plt.savefig(f"{combo_col}_scatter.png", dpi=450)
        plt.clf()
        plt.close()

    # additive analysis for ATAC
    atac_double_cols = [c for c in atac_df.columns if " & " in c]
    atac_synergy_df = df[atac_double_cols]

    for combo_col in atac_double_cols:
        single1, single2 = combo_col.split(" & ")

        # observed combined effect
        observed = df[combo_col]

        # predicted additive effect = sum of each single
        predicted = df[single1] + df[single2]

        # synergy (positive => synergy, negative => antagonism)
        synergy_col_name = combo_col + " (synergy)"
        atac_synergy_df[synergy_col_name] = observed - predicted

    # only keep synergy columns
    atac_synergy_df = atac_synergy_df[
        [c for c in atac_synergy_df.columns if "synergy" in c]
    ]

    # remove (synergy) from column names
    atac_synergy_df.columns = [
        c.replace(" (synergy)", "") for c in atac_synergy_df.columns
    ]
    plot_node_feature_ablations(atac_synergy_df, savename="atac_additive", log2=False)

    # plot scatter
    for combo_col in atac_double_cols:
        single1, single2 = combo_col.split(" & ")
        observed = df[combo_col]
        predicted = df[single1] + df[single2]

        # Make a small scatter for each pair, 1 point per tissue
        plt.figure(figsize=(2, 2))
        plt.scatter(predicted, observed, c="dodgerblue", alpha=0.7, s=7.5, linewidth=0)

        # Add identity line for reference
        min_val = min(predicted.min(), observed.min())
        max_val = max(predicted.max(), observed.max())
        plt.plot(
            [min_val, max_val], [min_val, max_val], "k--", linewidth=0.5, c="darkgray"
        )

        # remove whitespace
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.xlabel("Predicted additive effect")
        plt.ylabel("Observed combined effect")
        plt.title(f"{single1} + {single2}")
        plt.tight_layout()
        plt.savefig(f"{combo_col}_scatter.png", dpi=450)
        plt.clf()
        plt.close()

    # plot facet
    n_rows = 4
    n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5, 4))
    for i, combo_col in enumerate(double_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        single1, single2 = combo_col.split(" & ")

        observed = h3_df[combo_col]
        predicted = h3_df[single1] + h3_df[single2]

        ax.scatter(predicted, observed, c="dodgerblue", alpha=0.7, s=4, linewidth=0)

        min_val = min(predicted.min(), observed.min())
        max_val = max(predicted.max(), observed.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "k--", linewidth=0.5, c="darkgray"
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        ax.set_title(f"{single1} + {single2}")

    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supxlabel("Predicted additive effect", y=0.04)
    fig.supylabel("Observed combined effect", x=0.04)

    plt.tight_layout()
    plt.savefig("facet_scatter_plots.png", dpi=450)
    plt.clf()
    plt.close()

    # make individual GO enrichment figures
    working_dir = "/Users/steveho/gnn_plots"
    gencode_file = (
        f"{working_dir}/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"
    )

    # load gencode to symbol mapping
    symbol_to_gencode = load_gencode_lookup(gencode_file)
    gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

    # for tissue, tissue_name in TISSUES.items():
    #     node_ablations = pickle.load(
    #         open(f"{tissue}_release/node_feature_top_genes.pkl", "rb")
    #     )
    #     for idx in FEATURES:
    #         plot_single_enrichment_after_ablation(
    #             node_ablations=node_ablations,
    #             gencode_to_symbol=gencode_to_symbol,
    #             idx=idx,
    #             sample=tissue,
    #         )
    for idx in FEATURES:
        collate_enrichment_results(
            gencode_to_symbol=gencode_to_symbol,
            idx=idx,
        )

    # # get top 100 genes for
    # top = genes[12]

    # # convert to gene symbols
    # top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, fc in top]
    # print(top_genes)

    # get common genes..(?)

    # get most differentially affected genes..(?)

    # load GO collated results
    set_matplotlib_publication_parameters()
    df = pd.read_csv("reactome_H3K4me1.csv")
    df["sample_display"] = df["Sample"].map(TISSUES)

    # sort terms by neg_log_p
    df = df.sort_values("neg_log_p", ascending=False).copy()

    # count number of genes
    df["gene_count"] = df["Genes"].str.split(";").str.len()


if __name__ == "__main__":
    main()


# deprecated GSEA code
# # get top_n_genes
# top = node_ablations[idx]

# # convert to absolute fc
# top = [(gene, abs(fc)) for gene, fc in top]

# # remove genes with 0 fold change
# top = [x for x in top if x[1] != 0]

# # convert to gene symbols
# top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, _ in top]

# # pre-rank df for gsea
# ranked_df = pd.DataFrame(top, columns=["Gene", "FC"])
# ranked_df["Gene"] = ranked_df["Gene"].apply(
#     lambda x: gencode_to_symbol.get(x.split("_")[0])
# )

# # run gsea
# pre_res = gp.prerank(
#     rnk=ranked_df,
#     gene_sets="GO_Molecular_Function_2023",
#     threads=4,
#     permutation_num=1000,
#     outdir=None,
#     seed=42,
#     verbose=True,
# )

# ax = dotplot(
#     pre_res.res2d,
#     column="FDR q-val",
#     title="Reactome Pathways",
#     figsize=(40, 8),
#     cutoff=0.25,
# )
# ax.figure.savefig("test_gsea_reactome.png", dpi=300)


# import math
# import seaborn as sns
# import matplotlib.pyplot as plt

# n_rows = 4
# n_cols = 2

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5, 4))
# for i, combo_col in enumerate(double_cols[:8]):
#     row = i // n_cols
#     col = i % n_cols
#     ax = axes[row, col]
#     single1, single2 = combo_col.split(" & ")

#     observed = h3_df[combo_col]
#     predicted = h3_df[single1] + h3_df[single2]

#     ax.scatter(predicted, observed, c="dodgerblue", alpha=0.7, s=4, linewidth=0)

#     min_val = min(predicted.min(), observed.min())
#     max_val = max(predicted.max(), observed.max())
#     ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=0.5, c="darkgray")

#     ax.set_xlim(min_val, max_val)
#     ax.set_ylim(min_val, max_val)

#     ax.set_title(f"{single1} + {single2}")

# for ax in axes.flat:
#     ax.set_xlabel("")
#     ax.set_ylabel("")

# fig.supxlabel("Predicted additive effect", y=0.04)
# fig.supylabel("Observed combined effect", x=0.04)

# plt.tight_layout()
# plt.savefig("facet_scatter_plots.png", dpi=450)
# plt.clf()
# plt.close()
