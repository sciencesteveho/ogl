#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Code to download and process GTEx gene expression data for training targets.
**Note that the protein expression data is available as XLSX files and thus are
not included in the script.

root_directory
├── shared_data
    └── targets
        ├── expression
        ├── matrices  <- place to store matrix level data
        └── tpm  <- place to store tissue level tpm
        
"""

import os
import pathlib
import pickle

from cmapPy.pandasGEXpress.parse_gct import parse  # type: ignore
from cmapPy.pandasGEXpress.write_gct import write  # type: ignore
import pandas as pd

DOWNLOADS_URLS = [
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
]

GENE_QUANTIFICATIONS = {
    "k562": "https://www.encodeproject.org/files/ENCFF611MXW/@@download/ENCFF611MXW.tsv",
    "imr90": "https://www.encodeproject.org/files/ENCFF019KLP/@@download/ENCFF019KLP.tsv",
    "gm12878": "https://www.encodeproject.org/files/ENCFF362RMV/@@download/ENCFF362RMV.tsv",
    "hepg2": "https://www.encodeproject.org/files/ENCFF103FSL/@@download/ENCFF103FSL.tsv",
    "h1-esc": "https://www.encodeproject.org/files/ENCFF910OBU/@@download/ENCFF910OBU.tsv",
    "hmec": "https://www.encodeproject.org/files/ENCFF292FVY/@@download/ENCFF292FVY.tsv",
    "nhek": "https://www.encodeproject.org/files/ENCFF223GBX/@@download/ENCFF223GBX.tsv",
    "hippocampus": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_brain_hippocampus.gct.gz",
    "lung": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_lung.gct.gz",
    "pancreas": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_pancreas.gct.gz",
    "skeletal_muscle": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_muscle_skeletal.gct.gz",
    "small_intestine": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_small_intestine_terminal_ileum.gct.gz",
    "liver": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_liver.gct.gz",
    "aorta": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_artery_aorta.gct.gz",
    "skin": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_skin_not_sun_exposed_suprapubic.gct.gz",
    "left_ventricle": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_heart_left_ventricle.gct.gz",
    "mammary": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_breast_mammary_tissue.gct.gz",
    "spleen": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_spleen.gct.gz",
    "ovary": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_ovary.gct.gz",
    "adrenal": "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/tpms-by-tissue/gene_tpm_2017-06-05_v8_adrenal_gland.gct.gz",
}


def _check_and_download(file: str, filename: str) -> None:
    """Download a given file if the file does not already exist"""
    if not os.path.exists(file):
        print(f"Downloading {filename}...")
        os.system(f"wget -O {filename} {file}")
    else:
        print(f"{file} already exists.")


def _tpm_median_across_all_tissues(
    median_across_all_file: pathlib.PosixPath,
    all_matrix_gct: str,
) -> None:
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT and
    saves it. Because the file is large and requires a lot of memory, we ran
    this separately from the produce_training_targets function and is only run once.

    Args:
        median_across_all_file (str): /path/to/median_across_all_file
        all_matrix_gct (str): /path/to/gtex gct file
    """
    try:
        if not median_across_all_file.exists():
            median_series = pd.Series(
                parse(all_matrix_gct).data_df.median(axis=1), name="all_tissues"
            ).to_frame()
            median_series.to_pickle(median_across_all_file, mode="xb")
        else:
            print("File already exists")
    except FileExistsError:
        print("File already exists!")


def standardize_tissue_name(tissue_name: str) -> str:
    """Convert to lowercase, replace spaces with underscores, and remove parentheses"""
    return tissue_name.casefold().replace(" ", "_").replace("(", "").replace(")", "")


def _write_tissue_level_gct(gct_file: str, output_dir: str) -> None:
    """Read the GCT and write out each individual column (tissue) as a separate GCT file."""
    gct_df = parse(gct_file)
    tissues = gct_df.col_metadata_df.columns

    for tissue in tissues:
        tissue_df = gct_df.data_df.loc[:, tissue].to_frame()
        new_gct = gct_df.clone(base_df=tissue_df)
        tissue_file_name = os.path.join(
            output_dir, f"{tissue.replace(' - ', '_').replace(' ', '_')}.gct"
        )
        write(new_gct, tissue_file_name)
        print(f"Created GCT file for {tissue}: {tissue_file_name}")


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
