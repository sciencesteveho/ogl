# sourcery skip: avoid-global-variables
from collections import Counter
import pickle
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools
import scipy.stats as stats  # type: ignore
import seaborn as sns  # type: ignore
import torch

from figures.saliency import get_old_indices_for_type
from figures.saliency import sort_by_type

interp_path = "/Users/steveho/gnn_plots/interpretation"
idx_path = "/Users/steveho/gnn_plots/graph_resources/idxs"


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

# look at LINE / enhancers
# look at CTCF enhancers
# look at H3K9me3


# open prediction df
gene_type_counter = Counter()
pattern = re.compile(r'gene_type\s+"([^"]+)"')

# get pybedtools gencode
gencodefile = "/Users/steveho/gnn_plots/graph_resources/local/gencode_v26_genes_only_with_GTEx_targets.bed"

for tissue in TISSUES:
    # for tissue in ["k562"]:
    # tissue = 'hepg2'
    feat_idx = 11
    tissue = "pancreas"
    outpath = f"{interp_path}/{tissue}_release"
    saliency_map = f"{outpath}/scaled_saliency_map.pt"
    idx_file = f"{idx_path}/{tissue}_release_full_graph_idxs.pkl"
    dfpath = f"/Users/steveho/gnn_plots/interpretation/{tissue}_release/best_predictions_2_hop.csv"

    # for node_type in ["promoters", "enhancers", "genes", "dyadic"]:
    for node_type in ["enhancers"]:
        node_type = "enhancers"
        # for node_type in ["genes"]:
        with open(idx_file, "rb") as f:
            idxs = pickle.load(f)

        type_specific_idxs = get_old_indices_for_type(idxs, node_type)

        idx_to_node = {v: k for k, v in idxs.items()}  # map from int -> node name

        # 2) Sort + reorder by type
        old_to_new, separation_points = sort_by_type(idxs)
        saliency = torch.load(saliency_map)

        # convert to numpy if needed
        saliency_np = (
            saliency.detach().cpu().numpy() if hasattr(saliency, "detach") else saliency
        )

        # reorder
        saliency_reordered = np.zeros_like(saliency_np)
        for old_idx, new_idx in old_to_new.items():
            saliency_reordered[new_idx] = saliency_np[old_idx]

        # map selected original indices -> new positions
        selected_mapping = [
            (old_to_new[i], i) for i in type_specific_idxs if i in old_to_new
        ]
        if not selected_mapping:
            raise ValueError(
                "None of the selected indices were found in the index mapping."
            )

        # sort by new (reordered) index
        selected_mapping.sort(key=lambda x: x[0])
        new_indices, original_labels = zip(*selected_mapping)

        # dictionary from new index -> original node name
        new_idx_to_node = {
            new: idx_to_node[old] for new, old in zip(new_indices, original_labels)
        }

        # subset saliency according to new_indices
        saliency_selected = saliency_reordered[np.array(new_indices), :]

        row_indices = np.where(saliency_selected[:, feat_idx] == 1)[0]

        # saliency_selected has shape (46346, num_features)
        row_sums = np.sum(saliency_selected, axis=1)

        # sort row indices by descending row sum
        sorted_indices = np.argsort(row_sums)[::-1]

        # pick the top 10000 rows
        top_indices = sorted_indices[:10000]

        # get the original labels
        top_labels = [new_idx_to_node[new_indices[row_idx]] for row_idx in top_indices]
        top_labels = [label.split(f"_{tissue}")[0] for label in top_labels]

        # top_labels = [label.split("_")[0] for label in top_labels]

        # open k-hops
        # focused on subset where difference in prediction <= 0.5 log2tpm
        # prtfile = f"/Users/steveho/gnn_plots/interpretation/{tissue}_release/connected_component_perturbations_2_hop.pkl"
        # with open(prtfile, "rb") as f:
        #     perts = pickle.load(f)

        # perts = perts["single"]
        # subset_nodes = set(top_labels)
        # genes_of_interest = set()  # store deduped gene keys here

        # for top_gene, sub_dict in perts.items():
        #     for sub_key in sub_dict.keys():
        #         if sub_key in subset_nodes:
        #             genes_of_interest.add(top_gene)
        #             break

        # genes_of_interest = list(genes_of_interest)
        # genes_of_interest = [gene.split("_")[0] for gene in genes_of_interest]

        # genes_of_interest = top_labels

        # genes = pybedtools.BedTool(gencodefile)
        # genes_filtered = genes.filter(lambda x: x[3] in genes_of_interest).saveas()

        # for feature in genes_filtered:
        #     annotation = feature[9]
        #     if match := pattern.search(annotation):
        #         gene_type = match[1]
        #         gene_type_counter[gene_type] += 1

        # for gene_type, count in gene_type_counter.items():
        #     print(f"{gene_type}: {count}")

        # get protein_coding count and non-pr

        # subset_new_indices = np.array(new_indices)[row_indices]
        # subset_original_labels = [
        #     new_idx_to_node[idx] for idx in subset_new_indices
        # ]

        # subset_compare = set(compare_labels)
        # genes_of_interest_compare = set()
        # for top_gene, sub_dict in perts.items():
        #     for sub_key in sub_dict.keys():
        #         if sub_key in subset_compare:
        #             genes_of_interest_compare.add(top_gene)
        #             break

        # genes_of_interest_compare = list(genes_of_interest_compare)

        # print(f"Node feature: {nodefeat}")
        # print("Genes of interest:", len(genes_of_interest))
        # print("Genes of interest compare:", len(genes_of_interest_compare))
        # df = pd.read_csv(dfpath)

        # # 1) Subset df for genes of interest
        # df_high = df[df["gene_id"].isin(genes_of_interest)].copy()
        # df_high["status"] = (
        #     "High contributing"  # or label, whichever name you prefer
        # )

        # # 2) subset for genes of interest in the compare set
        # df_compare = df[df["gene_id"].isin(genes_of_interest_compare)].copy()
        # df_compare["status"] = "Low contributing"

        # # random.sample can be used on the DataFrame index if you want full control,
        # # but pandas has a built-in .sample() method:
        # num_to_sample = len(df_high)
        # df_compare = df_compare.sample(n=num_to_sample, random_state=42)

        # # 3) If you only want 'prediction' and the new label in the final DataFrames:
        # df_high = df_high[["gene_id", "prediction", "status"]]
        # df_compare = df_compare[["gene_id", "prediction", "status"]]

        # df_plot = pd.concat([df_high, df_compare], ignore_index=True)

        # salient_vals = df_plot.loc[
        #     df_plot["status"] == "High contributing", "prediction"
        # ]
        # low_vals = df_plot.loc[
        #     df_plot["status"] == "Low contributing", "prediction"
        # ]
        # stat, pvalue = stats.mannwhitneyu(
        #     salient_vals, low_vals, alternative="two-sided"
        # )
        # # if not significant, skip
        # if pvalue > 0.05:
        #     print(f"Node feature {nodefeat} not significant.")
        #     continue

        # pvalue_formatted = f"{pvalue:.2e}"
        # base, exponent = pvalue_formatted.split("e")
        # exponent = int(exponent)  # remove leading zeros if any

        # plt.figure(figsize=(2.25, 2.25))
        # sns.violinplot(
        #     data=df_plot, x="status", y="prediction", palette="Set2", linewidth=0.5
        # )

        # title_str = (
        #     f"{tissue_name} - {nodefeat}\n"
        #     + rf"Mann-Whitney $P$ = {base} $\times$ 10$^{{{exponent}}}$"
        # )
        # plt.title(title_str)
        # plt.ylabel(r"Predicted expression (Log$_2$TPM)")
        # plt.xlabel("")

        # plt.tight_layout()
        # plt.savefig(
        #     f"violin_{nodefeat}_{node_type}_{tissue}.png",
        #     dpi=450,
        #     bbox_inches="tight",
        # )
        # plt.clf()

        # plot saliency heatmap for selected indices
        saliency_subset = saliency_selected[row_indices, :]

        fig, ax = plt.subplots()
        sns.heatmap(saliency_subset, cmap="viridis", cbar=False, ax=ax)  # type: ignore

        # add colorbar
        cbar = plt.colorbar(
            ax.collections[0],
            shrink=0.25,
            aspect=7.5,
            ax=ax,
            location="left",
            pad=0.08,
        )
        cbar.outline.set_linewidth(0.5)  # type: ignore
        cbar.ax.set_title("Contribution", pad=10)

        # add feature labels
        n_cols = saliency_np.shape[1]
        xtick_positions = np.arange(5, n_cols) + 0.5
        xtick_labels = [FEATURES_name[i] for i in range(5, n_cols)]
        plt.xticks(xtick_positions, xtick_labels, rotation=90, ha="center")
        plt.yticks([])

        # add positional encoding bracket for first 5 features
        ax.text(
            2.3,
            -0.03,
            "positional\nencoding",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
        )

        # plot descriptors
        num_nodes = len(subset_original_labels)
        plt.xlabel("Features")
        plt.ylabel(f"{num_nodes:,} nodes")
        plt.title(
            f"{tissue} input x gradient saliency map ({node_type}) where {nodefeat} = 1"
        )

        plt.tight_layout()
        plt.savefig(
            f"{outpath}/saliency_{nodefeat}_{node_type}.png",
            dpi=450,
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()
