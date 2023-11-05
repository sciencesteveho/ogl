# Separate from automated testing, this script is used to ensure that the
# training targets are accurate

"""Tests that a random subset of targets match the median values in their
respective tissues."""

import numpy as np
import pickle
import random

from cmapPy.pandasGEXpress.parse_gct import parse

TISSUES = {
    "aorta": "Artery - Aorta",
    "hippocampus": "Brain - Hippocampus",
    "left_ventricle": "Heart - Left Ventricle",
    "liver": "Liver",
    "lung": "Lung",
    "mammary": "Breast - Mammary Tissue",
    "pancreas": "Pancreas",
    "skeletal_muscle": "Muscle - Skeletal",
    "skin": "Skin - Sun Exposed (Lower leg)",
    "small_intestine": "Small Intestine - Terminal Ileum",
}


def test_pickled_values_are_accurate(targets, true_df):
    for split in targets:
        test_targets = random.sample(list(targets[split]), 100)
        for entry in test_targets:
            target, tissue = entry.split("_", 1)
            true_median = true_df.loc[target, TISSUES[tissue]]
            print(f"log2 true median plus 0.25: {np.log2(true_median + 0.25)}")
            print(targets[split][entry][0])
            assert np.isclose(np.log2(true_median + 0.25), targets[split][entry][0])


def run_test():
    # load df with true median values
    true_df = parse(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct"
    ).data_df

    targets_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing"
    for targets in [
        "targets_random_assign.pkl",
        "targets_test_chr1.pkl",
        "targets_test_chr8-chr9_val_chr11-chr13.pkl",
    ]:
        with open(targets_dir + "/" + targets, "rb") as f:
            # open targets
            targets = pickle.load(f)

        # run test!
        test_pickled_values_are_accurate(
            targets=targets,
            true_df=true_df,
        )
