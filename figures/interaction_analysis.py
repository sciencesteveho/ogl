# sourcery skip: avoid-global-variables, avoid-single-character-names-variables, docstrings-for-modules, name-type-suffix, no-long-functions, require-parameter-annotation, require-return-annotation, sourcery skip: extract-duplicate-method

from collections import defaultdict
import os
import pickle

from matplotlib.colors import LinearSegmentedColormap  # type: ignore
from matplotlib.patches import Patch  # type: ignore
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram  # type: ignore
from scipy.cluster.hierarchy import linkage  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.cluster import AgglomerativeClustering  # type: ignore

from omics_graph_learning.visualization import set_matplotlib_publication_parameters

# Feature mapping from index to name - adjust based on your actual mapping
feature_mapping = {
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

# group features by biological function
feature_groups = {
    "Chromatin active": [
        "h3k27ac",
        "h3k4me3",
        "h3k4me2",
        "h3k4me1",
        "h3k9ac",
        "h3k79me2",
    ],
    "Chromatin repressive": [
        "h3k27me3",
        "h3k36me3",
        "h3k9me3",
    ],
    "Chromatin accessibility": [
        "atac",
        "dnase",
    ],
    "Chromatin achitecture": [
        "ctcf",
        "rad21",
        "smc3",
    ],
    "Methylation": [
        "cpg",
    ],
    "Sequence features": [
        "gc",
        "size",
    ],
    "Repeats": [
        "line",
        "ltr",
        "microsatellites",
        "simplerepeats",
        "sine",
        "rnarepeat",
    ],
    "Replication timing": [
        "repg1b",
        "repg2",
        "reps1",
        "reps2",
        "reps3",
        "reps4",
    ],
    "Variation": [
        "cnv",
        "indels",
        "snp",
        "recombination",
    ],
    "Transcription": [
        "polyasites",
        "rbpbindingsites",
        "polr2a",
    ],
    "Conservation": [
        "phastcons",
    ],
}


feature_categories = {
    "Chromatin active": [
        "H3K27ac",
        "H3K4me3",
        "H3K4me2",
        "H3K4me1",
        "H3K9ac",
        "H3K79me2",
    ],
    "Chromatin repressive": ["H3K9me3", "H3K27me3", "H3K36me3"],
    "Chromatin accessibility": ["ATAC", "DNase"],
    "Chromatin achitecture": ["CTCF", "RAD21", "SMC3"],
    "Methylation": ["CpG methylation"],
    "Sequence features": ["GC-content", "Size"],
    "Repeats": [
        "LINE",
        "SINE",
        "Microsatellites",
        "Long terminal repeats",
        "Simple repeats",
        "RNA repeat",
    ],
    "Replication timing": ["Rep G1b", "Rep G2", "Rep S1", "Rep S2", "Rep S3", "Rep S4"],
    "Variation": ["SNP", "Indels", "CNV", "Recombination rate"],
    "Transcription": [
        "POLR2A",
        "PolyA sites",
        "RBP binding sites",
    ],
    "Conservation": ["PhastCons"],
}

group_colors = {
    "Chromatin active": "#FF8A80",  # light red
    "Chromatin repressive": "#FFAB40",  # light orange
    "Chromatin accessibility": "#bdba68",  # light yellow
    "Chromatin achitecture": "#7ca671",  # light green
    "Methylation": "#5999b5",  # light blue
    "Sequence features": "#B388FF",  # light purple
    "Repeats": "#F48FB1",  # light pink
    "Replication timing": "#A7FFEB",  # light teal
    "Variation": "#FF80AB",  # light magenta
    "Transcription": "#EA80FC",  # light violet
    "Conservation": "#D1C4E9",  # light lavender
}


def convert_to_feature_matrix(tissue):
    """Convert the nested perturbation dictionary to a flat dataframe with
    gene, element, feature, and impact columns.
    """
    perturb_dir = "/Users/steveho/gnn_plots/interpretation"

    all_rows = []
    perturb_file = (
        f"{perturb_dir}/{tissue}_release/selected_component_perturbations.pkl"
    )

    try:
        with open(perturb_file, "rb") as f:
            perturbation_dict = pickle.load(f)

        # For each gene in this tissue's perturbation data
        for gene_id, reg_elements in perturbation_dict.items():
            # Extract the base gene name (assuming format like GENE_TISSUE or just GENE)
            # We'll use a more sophisticated approach to extract the gene name
            gene_base = gene_id

            # Look for known tissue suffixes in the gene_id
            for tissue_suffix in TISSUES.keys():
                if gene_id.endswith(f"_{tissue_suffix}"):
                    gene_base = gene_id[
                        : -(len(tissue_suffix) + 1)
                    ]  # Remove _tissue_suffix
                    break

            # Process each regulatory element and its features
            for element, features in reg_elements.items():
                for feature_idx, impact in features.items():
                    feature_idx = int(feature_idx)

                    if feature_idx in feature_mapping:
                        feature_name = feature_mapping[feature_idx]

                        all_rows.append(
                            {
                                "Gene": gene_base,
                                "Gene_ID": gene_id,
                                "Element": element,
                                "Tissue": TISSUES[tissue],
                                "Feature": feature_name,
                                "Feature_Index": feature_idx,
                                "Impact": impact,
                                "Absolute_Impact": abs(impact),
                            }
                        )

    except FileNotFoundError:
        print(f"Warning: Could not find perturbation file for tissue {tissue}")

    return pd.DataFrame(all_rows)


def analyze_feature_correlations(feature_df):
    """Calculate correlations between different feature impacts."""
    pivot_df = feature_df.pivot_table(
        index=["Gene_ID", "Element"], columns="Feature", values="Impact", fill_value=0
    )

    # correlation matrix
    corr_matrix = pivot_df.corr(method="pearson")

    # calculate p-values for correlations
    p_values = pd.DataFrame(
        np.zeros_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns
    )

    for i, feat1 in enumerate(corr_matrix.columns):
        for j, feat2 in enumerate(corr_matrix.columns):
            if i > j and (pivot_df[feat1].var() > 0 and pivot_df[feat2].var() > 0):
                r, p = pearsonr(pivot_df[feat1], pivot_df[feat2])
                p_values.loc[feat1, feat2] = p
                p_values.loc[feat2, feat1] = p  # symmetric matrix

    return corr_matrix, p_values, pivot_df


def reorder_matrix_by_groups(corr_matrix, feature_groups):
    """Reorder correlation matrix according to feature groups."""
    # create lowercase versions for case-insensitive matching
    corr_lower = corr_matrix.copy()
    corr_lower.index = corr_lower.index.str.lower()
    corr_lower.columns = corr_lower.columns.str.lower()
    avail_features_lower = list(corr_lower.index)

    # function to find best match
    def find_match(feature, available_features):
        if feature in available_features:
            return feature
        matches = [f for f in available_features if feature in f or f in feature]
        return matches[0] if matches else None

    # create ordered list based on groups
    ordered_indices = []
    group_boundaries = []
    current_idx = 0

    for group, features in feature_groups.items():
        group_start = current_idx
        for feature in features:
            match = find_match(feature, avail_features_lower)
            if match:
                # Get original case
                orig_idx = corr_matrix.index[avail_features_lower.index(match)]
                if orig_idx not in ordered_indices:
                    ordered_indices.append(orig_idx)
                    current_idx += 1

        if current_idx > group_start:
            group_boundaries.append((group, group_start, current_idx))

    # add any remaining features
    for feat in corr_matrix.index:
        if feat not in ordered_indices:
            ordered_indices.append(feat)

    # reorder matrix
    ordered_corr = corr_matrix.loc[ordered_indices, ordered_indices]
    return ordered_corr, group_boundaries


def find_feature_interactions(feature_df, threshold=0.4):
    """
    Find pairs of features that co-occur with strong effects.

    Returns two types of interactions:
    1. Synergistic: both features have effects in the same direction
    2. Antagonistic: features have effects in opposite directions
    """
    synergistic = []
    antagonistic = []

    # Group by element to find co-occurring features
    for (gene_id, element), group in feature_df.groupby(["Gene_ID", "Element"]):
        # Filter for features with significant impact
        significant = group[abs(group["Impact"]) > threshold]

        # Look at all pairs of features
        features = significant["Feature"].tolist()
        impacts = significant["Impact"].tolist()

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat1, feat2 = features[i], features[j]
                impact1, impact2 = impacts[i], impacts[j]

                strength = min(abs(impact1), abs(impact2))
                # check if both impacts are in the same direction
                if impact1 * impact2 > 0:
                    interaction_type = "Synergistic"
                    synergistic.append(
                        {
                            "Gene": gene_id,
                            "Element": element,
                            "Feature1": feat1,
                            "Feature2": feat2,
                            "Impact1": impact1,
                            "Impact2": impact2,
                            "Strength": strength,
                        }
                    )
                else:
                    interaction_type = "Antagonistic"
                    antagonistic.append(
                        {
                            "Gene": gene_id,
                            "Element": element,
                            "Feature1": feat1,
                            "Feature2": feat2,
                            "Impact1": impact1,
                            "Impact2": impact2,
                            "Strength": strength,
                        }
                    )

    synergistic_df = pd.DataFrame(synergistic).sort_values("Strength", ascending=False)
    antagonistic_df = pd.DataFrame(antagonistic).sort_values(
        "Strength", ascending=False
    )

    return synergistic_df, antagonistic_df


def analyze_feature_groups(correlation_matrix):
    """Analyze correlations between feature groups"""
    group_corr = pd.DataFrame(
        index=feature_categories.keys(), columns=feature_categories.keys()
    )

    for g1, feats1 in feature_categories.items():
        for g2, feats2 in feature_categories.items():
            # filter for features that are in both the correlation matrix and feature groups
            feats1_filtered = [f for f in feats1 if f in correlation_matrix.columns]
            feats2_filtered = [f for f in feats2 if f in correlation_matrix.columns]

            if (
                feats1_filtered and feats2_filtered
            ):  # check if we have features to compare
                # extract the submatrix for these feature groups
                submatrix = correlation_matrix.loc[feats1_filtered, feats2_filtered]
                # calculate the average correlation
                group_corr.loc[g1, g2] = submatrix.values.mean()

    return group_corr


set_matplotlib_publication_parameters()
colors = ["#0000A0", "white", "#ff5757"]
cmap = LinearSegmentedColormap.from_list("custom_RdBu", colors, N=100)

# make synergestic and antagonistic df to combine all
synergistic_dfs = []
antagonistic_dfs = []

# convert data to df
for tissue in TISSUES.keys():
    feature_df = convert_to_feature_matrix(tissue=tissue)

    # get feature correlation matrix
    corr_matrix, p_values, pivot_df = analyze_feature_correlations(feature_df)
    ordered_corr, group_boundaries = reorder_matrix_by_groups(
        corr_matrix, feature_groups
    )

    # plot feature correlation heatmap
    plt.figure(figsize=(6, 6))
    mask = np.triu(np.ones_like(ordered_corr, dtype=bool), k=1)  # k=1 keeps diagonal
    sns.heatmap(
        ordered_corr,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.175, "aspect": 8},
        xticklabels=True,  # Ensure x labels are shown
        yticklabels=True,  # Ensure y labels are shown
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{tissue}_feature_correlation_matrix.png", dpi=450)
    plt.close()

    # analyze feature groups
    group_corr = analyze_feature_groups(corr_matrix)
    group_corr = group_corr.astype(float)

    plt.figure(figsize=(5, 5))
    sns.heatmap(
        group_corr,
        annot=True,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.25, "aspect": 6.5},
    )
    plt.title("Correlation between feature groups")
    plt.tight_layout()
    plt.savefig(f"{tissue}_feature_group_correlation.png", dpi=450)
    plt.close()

    # perform hierarchical clustering
    Z = linkage(corr_matrix, method="ward")
    plt.figure(figsize=(5.5, 4))
    dendro = dendrogram(Z, labels=corr_matrix.columns, leaf_font_size=7)
    plt.title(
        "Hierarchical clustering of features based\non correlation of predicted impact"
    )

    # remove all spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # rotate x labels 90 degrees and remove y ticks
    plt.xticks(rotation=90)
    plt.yticks([])

    # create a mapping from feature name to group
    feature_to_group = {}
    for group, features in feature_categories.items():
        for feat in features:
            feature_to_group[feat] = group

    # Add colored squares next to the labels instead of coloring the text
    for label in ax.get_xticklabels():
        feature = label.get_text()
        group = feature_to_group.get(feature)
        if group:
            # Get label position
            pos = label.get_position()
            transform = label.get_transform()

            # Add colored square marker next to the label
            ax.text(
                pos[0],  # x position
                0.03,  # Offset below the x-axis
                "■",  # Square character
                color=group_colors[group],
                transform=transform,
                rotation=0,
                va="top",
                ha="center",
            )

    # keep original labels black (default)
    for label in ax.get_xticklabels():
        label.set_color("black")

    legend_elements = [
        Patch(facecolor=color, label=group) for group, color in group_colors.items()
    ]

    # Add the legend to the plot
    plt.legend(
        handles=legend_elements,
        loc="upper left",  # Position at upper left
        bbox_to_anchor=(1.05, 0.75),  # Slightly outside the plot
        frameon=False,  # No frame around the legend
        handlelength=1,
        handleheight=1,
    )

    plt.tight_layout()
    plt.savefig(f"{tissue}_feature_clustering_dendrogram.png", dpi=450)
    plt.close()

    # find synergistic and antagonistic interactions
    synergistic_df, antagonistic_df = find_feature_interactions(feature_df)

    # add to synergistic and antagonistic dfs
    synergistic_dfs.append(synergistic_df)
    antagonistic_dfs.append(antagonistic_df)


# combine all synergistic and antagonistic dfs
synergistic_df = pd.concat(synergistic_dfs)
antagonistic_df = pd.concat(antagonistic_dfs)

# count and plot the most common interactions
syn_counts = (
    synergistic_df.groupby(["Feature1", "Feature2"]).size().reset_index(name="Count")
)
syn_counts = syn_counts.sort_values("Count", ascending=False)

ant_counts = (
    antagonistic_df.groupby(["Feature1", "Feature2"]).size().reset_index(name="Count")
)
ant_counts = ant_counts.sort_values("Count", ascending=False)


# define helper function to get category for a feature
def get_feature_category(feature):
    for category, features in feature_categories.items():
        if feature in features:
            return category
    return None


plt.figure(figsize=(3.25, 5.25))

# Filter for top 50
top_syn = syn_counts[syn_counts["Count"] > 1].head(50)

# Create the barplot with labels as "Feature1 + Feature2"
ax = sns.barplot(
    x="Count",
    y=top_syn.apply(lambda x: f"{x['Feature1']} + {x['Feature2']}", axis=1),
    data=top_syn,
)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.title("Top synergistic feature pairs")
plt.xlabel("Count")
plt.ylabel("")
plt.tight_layout()
plt.savefig("top_synergistic_pairs.png", dpi=450)
plt.close()

# -------------------------------------------
# Antagonistic pairs
# -------------------------------------------
plt.figure(figsize=(3.25, 5.25))

# Filter for top 50
top_ant = ant_counts[ant_counts["Count"] > 1].head(50)

# Create the barplot with labels as "Feature1 vs Feature2"
ax = sns.barplot(
    x="Count",
    y=top_ant.apply(lambda x: f"{x['Feature1']} vs {x['Feature2']}", axis=1),
    data=top_ant,
)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.title("Top antagonistic feature pairs")
plt.xlabel("Count")
plt.ylabel("")
plt.tight_layout()
plt.savefig("top_antagonistic_pairs.png", dpi=450)
plt.close()


# identify all unique features across both dataframes
all_features = set(synergistic_df["Feature1"].unique()).union(
    set(synergistic_df["Feature2"].unique()),
    set(antagonistic_df["Feature1"].unique()),
    set(antagonistic_df["Feature2"].unique()),
)

# assign category to each feature
feature_to_category = {}
for feature in all_features:
    assigned = False
    for category, cat_features in feature_categories.items():
        for cat_feature in cat_features:
            if cat_feature in feature:
                feature_to_category[feature] = category
                assigned = True
                break
        if assigned:
            break
    if not assigned:
        feature_to_category[feature] = "Other"

# create ordered list of features by category
ordered_features = []
for category in feature_categories.keys():
    category_features = [
        f for f in all_features if feature_to_category.get(f) == category
    ]
    ordered_features.extend(sorted(category_features))
# ddd any "Other" features at the end
ordered_features.extend(
    sorted([f for f in all_features if feature_to_category.get(f) == "Other"])
)

# create interaction matrix
interaction_matrix = pd.DataFrame(0, index=ordered_features, columns=ordered_features)

# populate synergistic interactions (positive values)
for _, row in synergistic_df.iterrows():
    feature1, feature2 = row["Feature1"], row["Feature2"]
    if feature1 in ordered_features and feature2 in ordered_features:
        interaction_matrix.loc[feature1, feature2] += 1
        interaction_matrix.loc[feature2, feature1] += 1  # Mirror for symmetry

# populate antagonistic interactions (negative values)
for _, row in antagonistic_df.iterrows():
    feature1, feature2 = row["Feature1"], row["Feature2"]
    if feature1 in ordered_features and feature2 in ordered_features:
        interaction_matrix.loc[feature1, feature2] -= 1
        interaction_matrix.loc[feature2, feature1] -= 1  # Mirror for symmetry

# create figure
plt.figure(figsize=(6, 6))

# calculate scalar min and max values for colormap
v_max = np.abs(interaction_matrix.values).max()
v_min = -v_max

# plot heatmap
ax = sns.heatmap(
    interaction_matrix,
    cmap=cmap,
    center=0,
    vmin=v_min,
    vmax=v_max,
    square=True,
    cbar_kws={"label": "Net interactions", "shrink": 0.25, "aspect": 6.5},
)

# calculate category boundaries
category_boundaries = {}
start_idx = 0
current_category = None

for i, feature in enumerate(ordered_features):
    category = feature_to_category.get(feature)
    if category != current_category:
        if current_category is not None:
            category_boundaries[current_category] = (start_idx, i - 1)
        current_category = category
        start_idx = i
# add the last category
if current_category is not None:
    category_boundaries[current_category] = (start_idx, len(ordered_features) - 1)

# draw rectangles around categories
for category, (start, end) in category_boundaries.items():
    width = end - start + 1
    # Draw rectangle around the category
    rect = Rectangle(
        (start, start),
        width,
        width,
        fill=False,
        edgecolor="black",
        lw=0.75,
        clip_on=False,
        alpha=0.75,
        linestyle="dashed",
    )
    ax.add_patch(rect)

    mid_point = start + width / 2
    plt.text(
        mid_point,
        -1,
        category,
        ha="left",
        va="top",
        color="gray",
        rotation=90,
        rotation_mode="anchor",
    )

# set tick parameters
plt.xticks(
    np.arange(len(ordered_features)) + 0.5,
    ordered_features,
    rotation=90,
    ha="right",
)
plt.yticks(np.arange(len(ordered_features)) + 0.5, ordered_features, rotation=0)

plt.tight_layout()

# Save high-resolution figure
plt.savefig("feature_interaction_heatmap.png", dpi=450, bbox_inches="tight")
