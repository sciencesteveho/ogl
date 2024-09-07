#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to prepare the refTSS v4.1 annotation file for use in the graph. Gene
symbols are converted to ENSG IDs and the TSS file is parsed multiple times if
multiple genes are associated with the TSS."""


import argparse
from collections import defaultdict
import csv
import subprocess
from typing import Any, Dict, List, Tuple

import pybedtools  # type: ignore


def genes_from_gencode(gencode_ref: pybedtools.BedTool) -> Dict[str, str]:
    """Returns a dict of gencode v26 genes, their ids and associated gene
    symbols
    """
    return {
        line[9].split(";")[3].split('"')[1]: line[3]
        for line in gencode_ref
        if line[0] not in ["chrX", "chrY", "chrM"]
    }


def _reftss_cut_cols(file: str) -> None:
    """Cuts the first and eighth columns of the refTSS annotation file to
    produce a file with only the TSS and gene symbol"""
    cmd = f"cut -f1,8 {file} > {file}.cut"
    subprocess.run(cmd, stdout=None, shell=True)


def _tss_to_gene_tuples(file: str) -> List[Tuple[str, str]]:
    """Read refTSS annotation file and return a list of tuples matching TSS to
    each gene symbol."""
    with open(file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        return [(row[0], col) for row in reader for col in row[1].split(" ") if col]


def _tss_tuples_to_dict(
    tss_tuples: List[Tuple[str, str]],
    genesymbol_to_gencode: Dict[str, str],
) -> Dict[str, List[str]]:
    """Map tuples to a dictionary with the first element as the key"""
    key_mappings = defaultdict(list)
    for key, value in tss_tuples:
        if value in genesymbol_to_gencode:
            key_mappings[key].append(genesymbol_to_gencode[value])
    return key_mappings


def prepare_tss_file(
    tss_file: str, annotation_file: str, gencode_ref: str, outfile: str
) -> None:
    """Prepares parsed TSS file, where each TSS is linked with its gencode gene.
    If multiple genes are associated with the TSS, the tss is split across
    multiple lines.

    Args:
        tss_file (str): The path to the TSS file.
        annotation_file (str): The path to the annotation file.
        gencode_ref (str): The path to the GENCODE reference file.
        savedir (str): The directory to save the output file.

    Returns:
        None
    """
    _reftss_cut_cols(annotation_file)
    genesymbol_to_gencode = genes_from_gencode(pybedtools.BedTool(gencode_ref))
    tss = pybedtools.BedTool(tss_file)
    maps = _tss_tuples_to_dict(
        _tss_to_gene_tuples(f"{annotation_file}.cut"),
        genesymbol_to_gencode=genesymbol_to_gencode,
    )

    bed: List[Any] = []
    for line in tss:
        if line[3] in maps:
            bed.extend(
                [line[0], line[1], line[2], f"tss_{line[3]}_{value}"]
                for value in maps[line[3]]
            )
        else:
            bed.append([line[0], line[1], line[2], f"tss_{line[3]}"])

    bed = pybedtools.BedTool(bed).saveas(outfile)


def main() -> None:
    """Run main function"""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--tss_file", type=str)
    argparser.add_argument("--annotation_file", type=str)
    argparser.add_argument("--gencode_ref", type=str)
    argparser.add_argument("--outfile", type=str)
    args = argparser.parse_args()

    prepare_tss_file(
        tss_file=args.tss_file,
        annotation_file=args.annotation_file,
        gencode_ref=args.gencode_ref,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
