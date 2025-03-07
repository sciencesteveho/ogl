"""Visualization utilities for the omics graph learning."""

import matplotlib.pyplot as plt  # type: ignore

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
    0: "Size",
    1: "GC-content",
    2: "ATAC",
    3: "CNV",
    4: "CpG methylation",
    5: "CTCF",
    6: "DNase",
    7: "H3K27ac",
    8: "H3K27me3",
    9: "H3K36me3",
    10: "H3K4me1",
    11: "H3K4me2",
    12: "H3K4me3",
    13: "H3K79me2",
    14: "H3K9ac",
    15: "H3K9me3",
    16: "Indels",
    17: "LINE",
    18: "Long terminal repeats",
    19: "Microsatellites",
    20: "PhastCons",
    21: "POLR2A",
    22: "PolyA sites",
    23: "RAD21",
    24: "RBP binding sites",
    25: "Recombination rate",
    26: "Rep G1b",
    27: "Rep G2",
    28: "Rep S1",
    29: "Rep S2",
    30: "Rep S3",
    31: "Rep S4",
    32: "RNA repeat",
    33: "Simple repeats",
    34: "SINE",
    35: "SMC3",
    36: "SNP",
}


def set_matplotlib_publication_parameters() -> None:
    """Set matplotlib parameters for publication."""
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 7,
            "figure.dpi": 450,
            "font.sans-serif": ["Arial", "Nimbus Sans"],
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
        }
    )


# not implemented yet
# __all__ = [""]
