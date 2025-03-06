#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to download and process GTEx gene expression data for training targets.
**Note that the protein expression data is available as XLSX files and thus are
not included in the script.

root_directory
├── shared_data
    └── targets
        ├── expression
        ├── matrices  <- place to store matrix level data
        └── tpm  <- place to store tissue level tpm

Additionally, this script will generate a dataframe that provides the median TPM
per gene across all samples within the GTEx V8 dataset (all samples meaning all
tissues as well), as well as the average TPM per gene across all samples.
"""

import argparse
import os
from pathlib import Path

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


def _check_and_download(path: Path, file: str) -> None:
    """Download a given file if the file does not already exist"""
    if not os.path.exists(file):
        os.system(f"wget -P {path} {file}")
    else:
        print(f"{file} already exists.")


def _gunzip_file(file: Path) -> None:
    """Gunzip a file"""
    os.system(f"gunzip {file}")


def _tpm_median_across_all_tissues(
    df: pd.DataFrame,
    median_across_all_file: Path,
) -> None:
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT and
    saves it.

    Args:
        median_across_all_file (str): /path/to/median_across_all_file
        gtex_tpm_gct (str): /path/to/gtex gct file (not median tpm, just tpm gct)
    """
    median_series = pd.Series(df.median(axis=1), name="all_tissues").to_frame()
    median_series.to_pickle(median_across_all_file)


def _avg_tpm_all_tissues(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Get the average (not median!) expression for each gene in GTEx across all
    samples. Formally, the average activity is the summed expression at each
    gene across all samples, divided by the total number of samples.

    Args:
        df: pd.DataFrame: dataframe with gene expression data
        output_dir: Path: output directory to save the average activity
        dataframe

    Returns:
        np.ndarray: array with average activity for each gene
    """
    sample_count = df.astype(bool).sum(axis=1)
    summed_activity = pd.Series(df.sum(axis=1), name="all_tissues").to_frame()
    summed_activity["average"] = (
        summed_activity.div(sample_count, axis=0).fillna(0).values
    )
    summed_activity.to_pickle(output_dir / "average_activity_df.pkl")


def standardize_tissue_name(tissue_name: str) -> str:
    """Convert to lowercase, replace spaces with underscores, and remove parentheses"""
    return tissue_name.casefold().replace(" ", "_").replace("(", "").replace(")", "")


def _write_tissue_level_gct(gtex_median_tpm_gct: Path, output_dir: Path) -> None:
    """Read the GCT and write out each individual column (tissue) as a separate GCT file."""
    gct_df = parse(gtex_median_tpm_gct)
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
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # set up directories
    target_dir = Path(args.target_dir)
    expression_dir = target_dir / "expression"
    matrix_dir = target_dir / "matrices"
    tpm_dir = target_dir / "tpm"

    # download matrix files
    for url in DOWNLOADS_URLS:
        _check_and_download(path=matrix_dir, file=url)
        _gunzip_file(matrix_dir / url.split("/")[-1])

    # download tissue level TPMs
    for _, url in GENE_QUANTIFICATIONS.items():
        _check_and_download(path=tpm_dir, file=url)
        _gunzip_file(tpm_dir / url.split("/")[-1])

    # load the GTEx TPM GCT
    gtex_tpm_gct = matrix_dir / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"
    gtex_tpm_df = parse(gtex_tpm_gct).data_df

    # make the median across all tissues matrix
    median_all_file = matrix_dir / "gtex_tpm_median_across_all_tissues.pkl"
    _tpm_median_across_all_tissues(
        df=gtex_tpm_df,
        median_across_all_file=median_all_file,
    )

    # make the average activity matrix
    _avg_tpm_all_tissues(
        df=gtex_tpm_df,
        output_dir=matrix_dir,
    )

    # write out individual tissue level gcts
    _write_tissue_level_gct(
        matrix_dir / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct",
        tpm_dir,
    )


if __name__ == "__main__":
    main()
