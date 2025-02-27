import pickle
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Assuming these imports work in your environment
from figures.saliency import get_old_indices_for_type
from figures.saliency import sort_by_type
from omics_graph_learning.visualization import set_matplotlib_publication_parameters

# Update these dictionaries/paths as needed for your environment
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


def load_saliency_and_indices(
    tissue: str,
    interp_path: str = "/Users/steveho/gnn_plots/interpretation",
    idx_path: str = "/Users/steveho/gnn_plots/graph_resources/idxs",
    release_suffix: str = "_release",
):
    """
    Loads:
      1) saliency map (as a PyTorch tensor or numpy array)
      2) dictionary of node -> index
      3) Reorders the saliency map by node type
    Returns:
      saliency_reordered (np.ndarray),
      idx_to_node (dict),    # Maps index -> node_name
      new_to_old (dict),     # Maps new index -> old index
      old_to_new (dict),     # Maps old index -> new index
      separation_points      # from sort_by_type (if needed)
    """
    outpath = f"{interp_path}/{tissue}{release_suffix}"
    saliency_map_path = f"{outpath}/scaled_saliency_map.pt"
    idx_file = f"{idx_path}/{tissue}{release_suffix}_full_graph_idxs.pkl"

    # Load saliency
    saliency = torch.load(saliency_map_path)
    saliency_np = (
        saliency.detach().cpu().numpy() if hasattr(saliency, "detach") else saliency
    )

    # Load index mapping
    with open(idx_file, "rb") as f:
        node_to_idx = pickle.load(f)  # str -> int
    idx_to_node = {v: k for k, v in node_to_idx.items()}

    # sort_by_type returns old_to_new, separation_points
    old_to_new, separation_points = sort_by_type(node_to_idx)

    # Prepare a new array with the same shape as original
    saliency_reordered = np.zeros_like(saliency_np)

    # new_to_old is just the inverse of old_to_new
    new_to_old = {}
    for old_i, new_i in old_to_new.items():
        saliency_reordered[new_i] = saliency_np[old_i]
        new_to_old[new_i] = old_i

    return saliency_reordered, idx_to_node, new_to_old, old_to_new, separation_points


def get_enhancer_indices(
    old_to_new: Dict[int, int], node_to_idx: Dict[str, int], node_type: str
):
    """
    Returns the new index positions for a given node_type (e.g., "enhancers").
    If you prefer the original indices, adjust accordingly.
    """
    # get_old_indices_for_type expects the original dictionary structure
    # so we need the old dictionary: node_to_idx is str->int, that's fine
    # We'll just compute the old indices for the node_type, then map them
    # to new indices.
    type_specific_old_idxs = get_old_indices_for_type(node_to_idx, node_type)

    # map from old -> new
    type_specific_new_idxs = []
    for old_idx in type_specific_old_idxs:
        if old_idx in old_to_new:
            type_specific_new_idxs.append(old_to_new[old_idx])
    return type_specific_new_idxs


def filter_for_feature(
    saliency_reordered: np.ndarray, new_indices: List[int], feat_idx: int
):
    """
    Among a set of node indices (new_indices), return only those where
    saliency_reordered[node_idx, feat_idx] == 1
    """
    # subset the relevant rows (enhancers) from saliency
    subset_sali = saliency_reordered[new_indices, :]
    # find row positions where feature = 1
    row_positions = np.where(subset_sali[:, feat_idx] == 1)[0]
    # subset back to new_indices
    final_indices = [new_indices[rp] for rp in row_positions]
    return final_indices


def get_common_enhancers_for_feature(
    tissue1: str, tissue2: str, feat_idx: int, node_type: str
) -> (List[str], List[str]):
    """
    1) Load saliency + reorder for each tissue
    2) Get new indices for enhancers
    3) Filter by feat_idx == 1
    4) Return the intersection of node names that appear in both tissues
    along with the respective new indices in each tissue.
    """
    # --- Tissue 1 ---
    sal1, idx_to_node1, new_to_old1, old_to_new1, sep1 = load_saliency_and_indices(
        tissue1
    )
    # We'll also need the original node->idx for tissue1 to pass to get_old_indices_for_type.
    # That is new_to_old1 (which is new->old) but we also want the original node->idx mapping:
    # we can invert idx_to_node1 to get node->idx if needed:
    node_to_idx1 = {v: k for k, v in idx_to_node1.items()}

    enh_new_idxs_1 = get_enhancer_indices(
        old_to_new1, node_to_idx1, node_type=node_type
    )
    # Filter for feat_idx == 1
    feat_enh_new_idxs_1 = filter_for_feature(sal1, enh_new_idxs_1, feat_idx)

    # map new index -> node name
    newidx_to_name_1 = {
        nidx: idx_to_node1[new_to_old1[nidx]] for nidx in feat_enh_new_idxs_1
    }
    # remove tissue identifier from node names
    newidx_to_name_1 = {
        nidx: re.sub(r"_[^_]+$", "", name) for nidx, name in newidx_to_name_1.items()
    }
    names_1 = set(newidx_to_name_1.values())

    # --- Tissue 2 ---
    sal2, idx_to_node2, new_to_old2, old_to_new2, sep2 = load_saliency_and_indices(
        tissue2
    )
    node_to_idx2 = {v: k for k, v in idx_to_node2.items()}

    enh_new_idxs_2 = get_enhancer_indices(
        old_to_new2, node_to_idx2, node_type=node_type
    )
    feat_enh_new_idxs_2 = filter_for_feature(sal2, enh_new_idxs_2, feat_idx)

    newidx_to_name_2 = {
        nidx: idx_to_node2[new_to_old2[nidx]] for nidx in feat_enh_new_idxs_2
    }
    newidx_to_name_2 = {
        nidx: re.sub(r"_[^_]+$", "", name) for nidx, name in newidx_to_name_2.items()
    }
    names_2 = set(newidx_to_name_2.values())

    # Intersection of node names
    common_names = names_1.intersection(names_2)

    # Filter each set of indices down to only those that appear in the intersection
    final_indices_1 = [
        nidx for nidx in feat_enh_new_idxs_1 if newidx_to_name_1[nidx] in common_names
    ]
    final_indices_2 = [
        nidx for nidx in feat_enh_new_idxs_2 if newidx_to_name_2[nidx] in common_names
    ]

    # Sort them consistently (e.g. by name) so both tissues match row ordering
    # Let's build a name->newidx map
    name_to_newidx_1 = {newidx_to_name_1[n]: n for n in final_indices_1}
    name_to_newidx_2 = {newidx_to_name_2[n]: n for n in final_indices_2}

    # sort by name for consistency
    common_names_sorted = sorted(list(common_names))

    # final new-indices lists in sorted order
    final_indices_1_sorted = [name_to_newidx_1[nm] for nm in common_names_sorted]
    final_indices_2_sorted = [name_to_newidx_2[nm] for nm in common_names_sorted]

    return (
        (final_indices_1_sorted, final_indices_2_sorted),
        (sal1, sal2),
        common_names_sorted,
    )


def compare_tissues_on_feature(
    tissue1: str, tissue2: str, node_type: str, feat_idx: int = 11, cmap="viridis"
):
    """
    Main function that:
      1) gets the new indices for the common enhancers (DNase=1 if feat_idx=11)
      2) plots them side by side as two heatmaps
    """
    (idxs_1, idxs_2), (sal1, sal2), common_names = get_common_enhancers_for_feature(
        tissue1, tissue2, feat_idx, node_type
    )

    # Subset the saliency arrays
    sal_subset_1 = sal1[idxs_1, :]
    sal_subset_2 = sal2[idxs_2, :]

    # Create a side-by-side figure with two subplots.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

    # --- Plot for tissue1 ---
    # Plot heatmap (without default colorbar)
    sns.heatmap(sal_subset_1, cmap=cmap, cbar=False, ax=axes[0])  # type: ignore

    # Add custom colorbar for tissue1
    cbar1 = plt.colorbar(
        axes[0].collections[0],
        shrink=0.25,
        aspect=7.5,
        ax=axes[0],
        location="left",
        pad=0.08,
    )
    cbar1.outline.set_linewidth(0.5)  # type: ignore
    cbar1.ax.set_title("Contribution", pad=10)

    # Add feature labels for tissue1
    n_cols = sal_subset_1.shape[1]
    xtick_positions = np.arange(5, n_cols) + 0.5
    xtick_labels = [FEATURES[i] for i in range(5, n_cols)]
    axes[0].set_xticks(xtick_positions)
    axes[0].set_xticklabels(xtick_labels, rotation=90, ha="center")
    axes[0].set_yticks([])  # remove y ticks

    # Add positional encoding bracket for the first 5 features
    axes[0].text(
        2.3,
        -0.03,
        "positional\nencoding",
        ha="center",
        va="top",
        transform=axes[0].get_xaxis_transform(),
    )

    # Add y-label and title for tissue1
    num_nodes1 = sal_subset_1.shape[0]
    axes[0].set_ylabel(f"{num_nodes1:,} nodes")
    axes[0].set_title(f"{TISSUES[tissue1]} input x gradient saliency map ({node_type})")

    # --- Plot for tissue2 ---
    # Plot heatmap (without default colorbar)
    sns.heatmap(sal_subset_2, cmap=cmap, cbar=False, ax=axes[1])  # type: ignore

    # Add custom colorbar for tissue2
    cbar2 = plt.colorbar(
        axes[1].collections[0],
        shrink=0.25,
        aspect=7.5,
        ax=axes[1],
        location="left",
        pad=0.08,
    )
    cbar2.outline.set_linewidth(0.5)  # type: ignore
    cbar2.ax.set_title("Contribution", pad=10)

    # Add feature labels for tissue2 (same as tissue1)
    n_cols2 = sal_subset_2.shape[1]
    xtick_positions2 = np.arange(5, n_cols2) + 0.5
    xtick_labels2 = [FEATURES[i] for i in range(5, n_cols2)]
    axes[1].set_xticks(xtick_positions2)
    axes[1].set_xticklabels(xtick_labels2, rotation=90, ha="center")
    axes[1].set_yticks([])  # remove y ticks

    # Add positional encoding bracket for the first 5 features in tissue2
    axes[1].text(
        2.3,
        -0.03,
        "positional\nencoding",
        ha="center",
        va="top",
        transform=axes[1].get_xaxis_transform(),
    )

    # Tissue2: Add title (y-label is shared so only set on left)
    num_nodes2 = sal_subset_2.shape[0]
    axes[1].set_ylabel(f"{num_nodes1:,} nodes")
    axes[1].set_title(f"{TISSUES[tissue2]} input x gradient saliency map ({node_type})")

    plt.tight_layout()
    # Save the figure; adjust the output path as needed.
    plt.savefig(
        f"comparison_{tissue1}_{tissue2}_feat{feat_idx}_{node_type}.png",
        dpi=450,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


# -----------------
# USAGE EXAMPLE:
# -----------------
if __name__ == "__main__":
    # Compare "liver" vs "pancreas" on DNase feature (feat_idx=11)
    set_matplotlib_publication_parameters()
    # compare_tissues_on_feature("liver", "pancreas", feat_idx=11, node_type="enhancers")
    # compare_tissues_on_feature("k562", "hepg2", feat_idx=11, node_type="enhancers")
    # compare_tissues_on_feature("k562", "gm12878", feat_idx=11, node_type="enhancers")
    # compare_tissues_on_feature("adrenal", "aorta", feat_idx=11, node_type="enhancers")
    # compare_tissues_on_feature("liver", "pancreas", feat_idx=11, node_type="promoters")
    # compare_tissues_on_feature("k562", "hepg2", feat_idx=11, node_type="promoters")
    # compare_tissues_on_feature("k562", "gm12878", feat_idx=11, node_type="promoters")
    # compare_tissues_on_feature("adrenal", "aorta", feat_idx=11, node_type="promoters")
    # compare_tissues_on_feature("liver", "pancreas", feat_idx=11, node_type="genes")
    # compare_tissues_on_feature("k562", "hepg2", feat_idx=11, node_type="genes")
    # compare_tissues_on_feature("k562", "gm12878", feat_idx=11, node_type="genes")
    # compare_tissues_on_feature("adrenal", "aorta", feat_idx=11, node_type="genes")
    compare_tissues_on_feature(
        "hmec",
        "imr90",
        feat_idx=11,
        node_type="enhancers",
    )
    compare_tissues_on_feature(
        "hmec",
        "imr90",
        feat_idx=11,
        node_type="promoters",
    )
    # compare_tissues_on_feature("liver", "lung", feat_idx=11, node_type="enhancers")
