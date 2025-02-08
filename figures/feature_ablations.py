#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code for figure 3.
"""

import pickle
from typing import Dict, List, Optional, Tuple, Union

import gseapy as gp  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from numpy import sqrt
import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import invert_symbol_dict
from omics_graph_learning.interpret.interpret_utils import load_gencode_lookup
from omics_graph_learning.visualization import set_matplotlib_publication_parameters

TISSUES = [
    # "adrenal",
    "aorta",
    "gm12878",
    # "h1_esc",
    "hepg2",
    "hippocampus",
    "hmec",
    "imr90",
    "k562",
    "left_ventricle",
    "liver",
    "lung",
    "mammary",
    "nhek",
    "ovary",
    "pancreas",
    "skeletal_muscle",
    "skin",
    "small_intestine",
    "spleen",
]

# Example list of file paths (replace with your actual paths or glob them)
file_paths = [f"{sample}_release/node_feature_perturbations.pkl" for sample in TISSUES]

# Read each file and store the results
data_frames = []
for i, fp in enumerate(file_paths, start=1):
    with open(fp, "rb") as f:
        d = pickle.load(f)

    # Convert to a DataFrame with a single row (one sample),
    # and columns are the node indices
    # In case keys are strings, convert them to int (or keep as string if you prefer)
    df_sample = pd.DataFrame(d.values(), index=d.keys()).T
    # Tag with sample name (row index)
    df_sample.index = [f"Sample_{i}"]

    data_frames.append(df_sample)

# Concatenate all samples into one DataFrame
df = pd.concat(data_frames)

# Sort columns by node index if needed (assuming numeric keys)
df.columns = df.columns.astype(int)  # convert columns to int if they are string
df = df.reindex(sorted(df.columns), axis=1)


# load the top x genes
# dictionary of idx: (gene, fc)
genes = pickle.load(open("node_feature_top_genes.pkl", "rb"))

working_dir = "/Users/steveho/gnn_plots"
gencode_file = (
    f"{working_dir}/graph_resources/local/gencode_to_genesymbol_lookup_table.txt"
)

# load gencode to symbol mapping
symbol_to_gencode = load_gencode_lookup(gencode_file)
gencode_to_symbol = invert_symbol_dict(symbol_to_gencode)

# get top 100 genes for
top = genes[15]

top = genes[18]
# convert to gene symbols
top_genes = [gencode_to_symbol.get(gene.split("_")[0]) for gene, fc in top]
print(top_genes)
