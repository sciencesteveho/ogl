#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""This code was used to download EpiMap repository peaks programmatically based
off of their BSS accession names"""

import argparse
import os
from typing import List, Tuple

import pandas as pd

ATTRIBUTES = [
    "ATAC-seq",
    "CTCF",
    "DNase-seq",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K79me2",
    "H3K9ac",
    "H3K9me3",
    "POLR2A",
    "RAD21",
    "SMC3",
]

EPIMAP_ACCESSIONS = {
    "k562": ["BSS00762"],
    "imr90": ["BSS00720"],
    "gm12878": ["BSS00439"],
    "hepg2": ["BSS00558"],
    "h1-esc": ["BSS00478"],
    "hmec": ["BSS01209"],
    "nhek": ["BSS00355"],
    "hippocampus": ["BSS00091", "BSS01125", "BSS01124"],
    "lung": ["BSS01869", "BSS01871", "BSS00150"],
    "pancreas": ["BSS01406", "BSS00123", "BSS01407"],
    "psoas": ["BSS01460", "BSS01462", "BSS01463"],
    "small_intestine": ["BSS01597", "BSS01588", "BSS01599"],
    "liver": ["BSS01519", "BSS01169", "BSS01159"],
    "aorta": ["BSS00088", "BSS00079", "BSS00080"],
    "skin": ["BSS01180", "BSS01182", "BSS01678"],
    "left_ventricle": ["BSS00512", "BSS00507", "BSS00513"],
    "mammary": ["BSS01209", "BSS01211", "BSS01213"],
    "spleen": ["BSS01631", "BSS01634", "BSS01628"],
    "ovary": ["BSS01401", "BSS01402", "BSS01403"],
    "adrenal": ["BSS00054", "BSS00046", "BSS00048"],
}

OBSERVED_URL_PREFIX = "https://epigenome.wustl.edu/epimap/data/observed/"
IMPUTED_URL_PREFIX = "https://epigenome.wustl.edu/epimap/data/imputed/"


def _load_epimap_tables(file: str) -> pd.DataFrame:
    """Loads in tables of EpiMap files"""
    return pd.read_csv(
        file,
        sep="\t",
        header=None,
        names=["accession", "mark", "filename"],
    )


def _get_epimap_files(
    accession: str, observed_df: pd.DataFrame, imputed_df: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """First checks the observed list to see if the files are present. If they
    are, it grabs them. Remaining files not observed are grabbed from the
    imputed marks."""
    observed = observed_df[
        (observed_df["accession"] == accession) & observed_df["mark"].isin(ATTRIBUTES)
    ]
    observed_marks = set(observed["mark"])
    imputed_marks = set(ATTRIBUTES) - observed_marks
    imputed = imputed_df[
        (imputed_df["accession"] == accession) & imputed_df["mark"].isin(imputed_marks)
    ]
    return (observed["filename"].tolist(), imputed["filename"].tolist())


def _format_urls(
    observed_filenames: List[str], imputed_filenames: List[str]
) -> List[str]:
    """Adds URL prefixes for full URLs to download"""
    observed_urls = [OBSERVED_URL_PREFIX + filename for filename in observed_filenames]
    imputed_urls = [IMPUTED_URL_PREFIX + filename for filename in imputed_filenames]
    return observed_urls + imputed_urls


def _download_files(urls: List[str], download_dir: str) -> None:
    """Downloads files to a specified directory"""
    os.makedirs(download_dir, exist_ok=True)
    for url in urls:
        os.system(f"wget -P {url} {download_dir}")


def _list_all_downloads(
    observed_df: pd.DataFrame, imputed_df: pd.DataFrame
) -> List[str]:
    """List all files that need to be downloaded. This code is used for a
    supplementary table."""
    all_files: List[str] = []
    accessions = [item for sublist in EPIMAP_ACCESSIONS.values() for item in sublist]
    for accession in accessions:
        files = _get_epimap_files(
            accession=accession, observed_df=observed_df, imputed_df=imputed_df
        )
        urls = _format_urls(*files)
        all_files.extend(urls)
    return all_files


def main() -> None:
    """Download epimap files for a specific tissue to a directory."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--observed_tsv",
        type=str,
        help="Path to the observed TSV file",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--imputed_tsv",
        type=str,
        help="Path to the imputed TSV file",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--tissue",
        type=str,
        help="Tissue name for downloading",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--download_dir",
        type=str,
        help="Directory to download files to",
        required=True,
    )
    args = parser.parse_args()

    observed_df = _load_epimap_tables(args.observed_tsv)
    imputed_df = _load_epimap_tables(args.imputed_tsv)
    for accession in EPIMAP_ACCESSIONS[args.tissue]:
        filenames = _get_epimap_files(
            accession=accession,
            observed_df=observed_df,
            imputed_df=imputed_df,
        )
        urls = _format_urls(*filenames)
        _download_files(urls, args.download_dir)
        print(
            f"Finished downloading {len(urls)} files to {args.download_dir} for {args.tissue}"
        )
    print("Finished downloading all files!")


if __name__ == "__main__":
    main()
