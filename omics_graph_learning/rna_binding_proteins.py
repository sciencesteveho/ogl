#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""_summary_ of project"""

import csv
import os
import pickle
from typing import Dict

import pybedtools  # type: ignore

RBP_PROTEINS = "/ocean/projects/bio210019p/stevesho/data/data_preparse/rbp_proteins.txt"
GENCODE = "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"


def genes_from_gencode(gencode_ref: pybedtools.BedTool) -> Dict[str, str]:
    """Returns a dict of gencode v26 genes, their ids and associated gene
    symbols
    """
    return {
        line[9].split(";")[3].split('"')[1]: line[3]
        for line in gencode_ref
        if line[0] not in ["chrX", "chrY", "chrM"]
    }


def main() -> None:
    """Main function"""
    genes = genes_from_gencode(pybedtools.BedTool(GENCODE))
    rbp_genes = [line[0] for line in csv.reader(open(RBP_PROTEINS), delimiter="\t")]

    common_genes = set(genes) & set(rbp_genes)


if __name__ == "__main__":
    main()
