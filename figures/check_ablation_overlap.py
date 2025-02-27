# sourcery skip: avoid-global-variables
from collections import Counter
import pickle
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools
from scipy.stats import chi2_contingency  # type: ignore
from scipy.stats import fisher_exact  # type: ignore
import scipy.stats as stats  # type: ignore
import seaborn as sns  # type: ignore
import statsmodels.stats.multitest as smm  # type: ignore
import torch

from omics_graph_learning.visualization import set_matplotlib_publication_parameters

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


def extract_subgraphs(k_hop_subgraph):
    """
    Given a k-hop subgraph represented as a nested dictionary, returns a simplified dictionary
    where each key is a gene and its value is a list of associated node names.

    Parameters:
        k_hop_subgraph (dict): A dictionary where keys are genes and values are dictionaries
                               of node: metadata pairs.

    Returns:
        dict: A dictionary mapping each gene to a list of node names.
    """
    return {
        gene.split("_")[0]: [
            item.removesuffix("_h1_esc") for item in list(neighbors.keys())
        ]
        for gene, neighbors in k_hop_subgraph.items()
    }


interp_path = "/Users/steveho/gnn_plots/interpretation"
idx_path = "/Users/steveho/gnn_plots/graph_resources/idxs"
resource_dir = "/Users/steveho/gnn_plots/graph_resources/local"
gencode_path = "/Users/steveho/gnn_plots/graph_resources/local/gencode_v26_genes_only_with_GTEx_targets.bed"
microsat_path = (
    "/Users/steveho/gnn_plots/graph_resources/local/microsatellites_hg38.bed"
)
gencode = pybedtools.BedTool(gencode_path)
microsat = pybedtools.BedTool(microsat_path)

with open(
    "/Users/steveho/gnn_plots/interpretation/h1_esc_release/node_feature_top_genes.pkl",
    "rb",
) as f:
    test = pickle.load(f)

top_micro_genes = test[24][:1000]
bottom_micro_genes = test[24][-1000:]
top_genes = [x[0].split("_")[0] for x in top_micro_genes]

gencode_df = pd.read_csv(gencode_path, sep="\t", header=None, comment="#")
gencode_df = gencode_df.iloc[:, 0:4]  # keep only chr, start, end, geneID
gencode_df.columns = ["chr", "start", "end", "gene_id"]

# Often the GENCODE gene IDs include version numbers like ENSG000001234.5, so if
# your top_genes list uses IDs without the version, remove it:

bottom_micro_genes = test[24][-1000:]
bottom_genes = [x[0].split("_")[0] for x in bottom_micro_genes]

top_gene_set = set(top_genes)
bottom_gene_set = set(bottom_genes)

top_bed = gencode_df[gencode_df["gene_id"].isin(top_gene_set)].copy()
bottom_bed = gencode_df[gencode_df["gene_id"].isin(bottom_gene_set)].copy()

# Convert to BedTool objects
top_bedtool = pybedtools.BedTool.from_dataframe(
    top_bed[["chr", "start", "end", "gene_id"]]
)
bottom_bedtool = pybedtools.BedTool.from_dataframe(
    bottom_bed[["chr", "start", "end", "gene_id"]]
)


microsat_bedtool = pybedtools.BedTool(microsat_path)

# Intersect top genes
ms_intersect_top = top_bedtool.intersect(microsat_bedtool, wa=True, wb=True)
# Intersect bottom genes
ms_intersect_bottom = bottom_bedtool.intersect(microsat_bedtool, wa=True, wb=True)

# ----------------------------------------------------
# 4) COUNT MICROSATELLITE MOTIFS IN EACH GENE SET
# ----------------------------------------------------
top_ms_types = []
for feature in ms_intersect_top:
    # feature is a pybedtools Interval. The first 4 columns come from the gene bed,
    # the last 4 columns come from the microsat bed.
    # So feature[7] should be the microsatellite "name" or motif, like "16xGT".
    top_ms_types.append(feature[7])

bottom_ms_types = []
for feature in ms_intersect_bottom:
    bottom_ms_types.append(feature[7])

# Count frequencies
counter_top = Counter(top_ms_types)
counter_bottom = Counter(bottom_ms_types)

print("Top Genes Microsatellite Counts (most common):")
for motif, cnt in counter_top.most_common(10):
    print(f"{motif}: {cnt}")

print("\nBottom Genes Microsatellite Counts (most common):")
for motif, cnt in counter_bottom.most_common(10):
    print(f"{motif}: {cnt}")


all_motifs = sorted(set(counter_top.keys()) | set(counter_bottom.keys()))
df = pd.DataFrame(index=all_motifs, columns=["top_count", "bottom_count"])
df["top_count"] = df.index.map(counter_top)
df["bottom_count"] = df.index.map(counter_bottom)
df = df.fillna(0).astype(int)


# For a global test, you could do a chi-square test on the entire contingency table:
#   "top_count" vs "bottom_count" across all motifs
from scipy.stats import chi2_contingency

contingency = df[["top_count", "bottom_count"]].values
chi2, pval, dof, expected = chi2_contingency(contingency)
print("\nChi-square test across all microsatellite types:")
print(f"chi2={chi2:.2f}, p-value={pval:.2e}, dof={dof}")


# Get top 10 most common motifs overall
top_motifs = df.sum(axis=1).sort_values(ascending=False).head(10).index

# Store p-values
p_values = []
odds_ratios = []

for motif in top_motifs:
    top_count = df.loc[motif, "top_count"]
    bottom_count = df.loc[motif, "bottom_count"]
    total_top = df["top_count"].sum()
    total_bottom = df["bottom_count"].sum()

    contingency_table = [
        [top_count, total_top - top_count],
        [bottom_count, total_bottom - bottom_count],
    ]

    odds_ratio, p_value = fisher_exact(contingency_table)
    odds_ratios.append(odds_ratio)
    p_values.append(p_value)

# Multiple testing correction (FDR)
rejected, corrected_pvals, _, _ = smm.multipletests(p_values, method="fdr_bh")

# Display results
for motif, or_, pval, corr_pval, sig in zip(
    top_motifs, odds_ratios, p_values, corrected_pvals, rejected
):
    print(
        f"Motif: {motif}, OR={or_:.2f}, p-value={pval:.4f}, corrected p={corr_pval:.4f}, significant={sig}"
    )

# Top Genes Microsatellite Counts (most common):
# 18xAC: 41
# 15xTG: 38
# 20xAC: 35
# 22xAC: 35
# 17xAC: 34
# 16xTG: 33
# 19xTG: 33
# 15xAC: 33
# 18xTG: 31
# 20xTG: 31

# Bottom Genes Microsatellite Counts (most common):
# 16xAC: 49
# 15xAC: 39
# 20xAC: 38
# 18xTG: 38
# 17xAC: 35
# 18xAC: 34
# 15xTG: 33
# 19xAC: 32
# 17xGT: 27
# 21xAC: 26

# Chi-square test across all microsatellite types:
# chi2=192.44, p-value=3.02e-01, dof=183

# Motif: 16xAC, OR=0.58, p-value=0.0225, corrected p=0.2253, significant=False
# Motif: 18xAC, OR=1.14, p-value=0.6395, corrected p=0.7948, significant=False
# Motif: 20xAC, OR=0.86, p-value=0.5536, corrected p=0.7948, significant=False
# Motif: 15xAC, OR=0.79, p-value=0.3399, corrected p=0.7042, significant=False
# Motif: 15xTG, OR=1.08, p-value=0.8099, corrected p=0.8099, significant=False
# Motif: 17xAC, OR=0.91, p-value=0.7154, corrected p=0.7948, significant=False
# Motif: 18xTG, OR=0.76, p-value=0.2734, corrected p=0.7042, significant=False
# Motif: 19xAC, OR=0.76, p-value=0.3521, corrected p=0.7042, significant=False
# Motif: 21xAC, OR=1.12, p-value=0.6897, corrected p=0.7948, significant=False
# Motif: 19xTG, OR=1.35, p-value=0.2820, corrected p=0.7042, significant=False

# looking at genes not enough, let's look at the subgraph nodes
resource_dir = "/Users/steveho/gnn_plots/graph_resources/local"
gencode_path = "/Users/steveho/gnn_plots/graph_resources/local/gencode_v26_genes_only_with_GTEx_targets.bed"
enhancer_path = (
    "/Users/steveho/gnn_plots/graph_resources/local/enhancer_epimap_screen_overlap.bed"
)
dyadic_path = (
    "/Users/steveho/gnn_plots/graph_resources/local/dyadic_epimap_screen_overlap.bed"
)
promoter_path = (
    "/Users/steveho/gnn_plots/graph_resources/local/promoter_epimap_screen_overlap.bed"
)
microsat_path = (
    "/Users/steveho/gnn_plots/graph_resources/local/microsatellites_hg38.bed"
)

tissue = "h1_esc"
prtfile = f"/Users/steveho/gnn_plots/interpretation/{tissue}_release/connected_component_perturbations_2_hop.pkl"
with open(prtfile, "rb") as f:
    perts = pickle.load(f)

perts = perts["single"]

# get subgraphs
subgraph = extract_subgraphs(perts)

# how many top and bottom genes have subgraphs
top_genes_with_subgraph = set(top_genes) & set(subgraph.keys())
bottom_genes_with_subgraph = set(bottom_genes) & set(subgraph.keys())

print(len(top_genes_with_subgraph), len(bottom_genes_with_subgraph))

# keep top 100
top_genes_with_subgraph = list(top_genes_with_subgraph)[:100]

# get bottom 100, bottom first / lowest first
bottom_genes_with_subgraph = list(bottom_genes_with_subgraph)[-100:]

# combine all nodes in top 100
top_nodes = set()
for gene in top_genes_with_subgraph:
    top_nodes.update(subgraph[gene])

# combine all nodes in bottom 100
bottom_nodes = set()
for gene in bottom_genes_with_subgraph:
    bottom_nodes.update(subgraph[gene])

gencode_df = pd.read_csv(gencode_path, sep="\t", header=None, comment="#")
enhancer_df = pd.read_csv(enhancer_path, sep="\t", header=None, comment="#")
dyadic_df = pd.read_csv(dyadic_path, sep="\t", header=None, comment="#")
promoter_df = pd.read_csv(promoter_path, sep="\t", header=None, comment="#")
gencode_df = gencode_df.iloc[:, 0:4]  # keep only chr, start, end, geneID
enhancer_df = enhancer_df.iloc[:, 0:4]
dyadic_df = dyadic_df.iloc[:, 0:4]
promoter_df = promoter_df.iloc[:, 0:4]

# combine all into node_df
node_df = pd.concat([gencode_df, enhancer_df, dyadic_df, promoter_df])
node_df.columns = ["chr", "start", "end", "node_id"]

top_bed = node_df[node_df["node_id"].isin(top_nodes)].copy()
bottom_bed = node_df[node_df["node_id"].isin(bottom_nodes)].copy()

# Convert to BedTool objects
top_bedtool = pybedtools.BedTool.from_dataframe(
    top_bed[["chr", "start", "end", "node_id"]]
)
bottom_bedtool = pybedtools.BedTool.from_dataframe(
    bottom_bed[["chr", "start", "end", "node_id"]]
)

microsat_bedtool = pybedtools.BedTool(microsat_path)

# Intersect top genes
ms_intersect_top = top_bedtool.intersect(microsat_bedtool, wa=True, wb=True)
# Intersect bottom genes
ms_intersect_bottom = bottom_bedtool.intersect(microsat_bedtool, wa=True, wb=True)

# ----------------------------------------------------
# 4) COUNT MICROSATELLITE MOTIFS IN EACH GENE SET
# ----------------------------------------------------
top_ms_types = []
for feature in ms_intersect_top:
    # feature is a pybedtools Interval. The first 4 columns come from the gene bed,
    # the last 4 columns come from the microsat bed.
    # So feature[7] should be the microsatellite "name" or motif, like "16xGT".
    top_ms_types.append(feature[7])

bottom_ms_types = []
for feature in ms_intersect_bottom:
    bottom_ms_types.append(feature[7])

# Count frequencies
counter_top = Counter(top_ms_types)
counter_bottom = Counter(bottom_ms_types)

print("Top Genes Microsatellite Counts (most common):")
for motif, cnt in counter_top.most_common(10):
    print(f"{motif}: {cnt}")

print("\nBottom Genes Microsatellite Counts (most common):")
for motif, cnt in counter_bottom.most_common(10):
    print(f"{motif}: {cnt}")


all_motifs = sorted(set(counter_top.keys()) | set(counter_bottom.keys()))
df = pd.DataFrame(index=all_motifs, columns=["top_count", "bottom_count"])
df["top_count"] = df.index.map(counter_top)
df["bottom_count"] = df.index.map(counter_bottom)
df = df.fillna(0).astype(int)

total_top = df["top_count"].sum()
total_bottom = df["bottom_count"].sum()

odds_ratios = []
p_values = []
for motif in df.index:
    top_count = df.loc[motif, "top_count"]
    bottom_count = df.loc[motif, "bottom_count"]
    contingency_table = [
        [top_count, total_top - top_count],
        [bottom_count, total_bottom - bottom_count],
    ]
    or_val, p_val = fisher_exact(contingency_table)
    odds_ratios.append(or_val)
    p_values.append(p_val)

# Multiple-testing correction (FDR)
rejected, fdr_vals, _, _ = smm.multipletests(p_values, method="fdr_bh")

# Add new columns directly to df
df["odds_ratio"] = odds_ratios
df["p_value"] = p_values
df["fdr"] = fdr_vals
df["significant"] = rejected
df["log2_or"] = np.log2(df["odds_ratio"].replace(0, np.nan))
df["neg_log10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))


# For a global test, you could do a chi-square test on the entire contingency table:
#   "top_count" vs "bottom_count" across all motifs


# contingency = df[["top_count", "bottom_count"]].values
# chi2, pval, dof, expected = chi2_contingency(contingency)
# print("\nChi-square test across all microsatellite types:")
# print(f"chi2={chi2:.2f}, p-value={pval:.2e}, dof={dof}")


# # Get top 10 most common motifs overall
# top_motifs = df.sum(axis=1).sort_values(ascending=False).head(193).index

# # Store p-values
# p_values = []
# odds_ratios = []

# for motif in df.index:
#     top_count = df.loc[motif, "top_count"]
#     bottom_count = df.loc[motif, "bottom_count"]

#     contingency_table = [
#         [top_count, total_top - top_count],
#         [bottom_count, total_bottom - bottom_count],
#     ]

#     or_val, p_val = fisher_exact(contingency_table)
#     odds_ratios.append(or_val)
#     p_values.append(p_val)

# # Multiple-testing correction (FDR)
# rejected, fdr_vals, _, _ = smm.multipletests(p_values, method="fdr_bh")

# # Add these stats as new columns to df
# df["odds_ratio"] = odds_ratios
# df["p_value"] = p_values
# df["fdr"] = fdr_vals  # corrected p-value
# df["significant"] = rejected

# # Compute log2(OR) and −log10(FDR); protect against zero or negative
# # values by clipping them to avoid -inf or inf.
# df["log2_or"] = np.log2(df["odds_ratio"].replace(0, np.nan))
# df["neg_log10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))

# df.head()

# # Display results
# for motif, or_, pval, corr_pval, sig in zip(
#     top_motifs, odds_ratios, p_values, corrected_pvals, rejected
# ):
#     print(
#         f"Motif: {motif}, OR={or_:.2f}, p-value={pval:.4f}, corrected p={corr_pval:.4f}, significant={sig}"
#     )

# Chi-square test across all microsatellite types:
# chi2=276.98, p-value=5.75e-05, dof=192

# Top Genes Microsatellite Counts (most common):
# 17xAC: 64
# 15xAC: 62
# 16xTG: 55
# 19xAC: 53
# 22xAC: 51
# 18xGT: 48
# 18xAC: 44
# 20xAC: 43
# 16xAC: 41
# 17xGT: 40

# Bottom Genes Microsatellite Counts (most common):
# 17xAC: 41
# 18xAC: 38
# 19xTG: 37
# 15xAC: 36
# 21xGT: 36
# 19xGT: 35
# 22xTG: 34
# 16xAC: 32
# 23xAC: 32
# 20xTG: 31


# Motif: 17xAC, OR=1.11, p-value=0.6159, corrected p=1.0000, significant=False
# Motif: 15xAC, OR=1.23, p-value=0.3494, corrected p=0.9770, significant=False
# Motif: 19xAC, OR=1.22, p-value=0.4321, corrected p=0.9846, significant=False
# Motif: 18xAC, OR=0.82, p-value=0.3660, corrected p=0.9770, significant=False
# Motif: 20xAC, OR=0.98, p-value=1.0000, corrected p=1.0000, significant=False
# Motif: 16xAC, OR=0.91, p-value=0.7188, corrected p=1.0000, significant=False
# Motif: 16xTG, OR=2.34, p-value=0.0015, corrected p=0.0997, significant=False
# Motif: 22xAC, OR=1.75, p-value=0.0388, corrected p=0.6818, significant=False
# Motif: 20xGT, OR=1.01, p-value=1.0000, corrected p=1.0000, significant=False
# Motif: 18xGT, OR=1.72, p-value=0.0459, corrected p=0.6818, significant=False
# Motif: 19xTG, OR=0.53, p-value=0.0149, corrected p=0.5763, significant=False
# Motif: 19xGT, OR=0.60, p-value=0.0553, corrected p=0.7119, significant=False
# Motif: 21xGT, OR=0.56, p-value=0.0297, corrected p=0.6818, significant=False
# Motif: 18xTG, OR=1.04, p-value=1.0000, corrected p=1.0000, significant=False
# Motif: 20xTG, OR=0.75, p-value=0.3047, corrected p=0.9770, significant=False
# Motif: 22xTG, OR=0.60, p-value=0.0519, corrected p=0.7119, significant=False
# Motif: 17xGT, OR=1.30, p-value=0.3635, corrected p=0.9770, significant=False
# Motif: 21xAC, OR=1.13, p-value=0.6973, corrected p=1.0000, significant=False
# Motif: 23xAC, OR=0.61, p-value=0.0645, corrected p=0.7207, significant=False
# Motif: 17xTG, OR=0.78, p-value=0.3533, corrected p=0.9770, significant=False
# Motif: 21xTG, OR=1.21, p-value=0.5778, corrected p=1.0000, significant=False
# Motif: 15xTG, OR=1.15, p-value=0.7657, corrected p=1.0000, significant=False
# Motif: 18xCA, OR=1.34, p-value=0.3703, corrected p=0.9770, significant=False
# Motif: 19xCA, OR=1.17, p-value=0.6502, corrected p=1.0000, significant=False
# Motif: 17xCA, OR=0.59, p-value=0.0898, corrected p=0.7532, significant=False
# Motif: 22xGT, OR=3.81, p-value=0.0003, corrected p=0.0314, significant=True
# Motif: 16xGT, OR=0.78, p-value=0.4336, corrected p=0.9846, significant=False
# Motif: 24xAC, OR=1.37, p-value=0.4251, corrected p=0.9846, significant=False
# Motif: 23xGT, OR=0.30, p-value=0.0003, corrected p=0.0314, significant=True
# Motif: 21xCA, OR=0.93, p-value=0.8675, corrected p=1.0000, significant=False
