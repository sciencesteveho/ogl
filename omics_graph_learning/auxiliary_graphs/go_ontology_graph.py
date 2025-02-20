#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to parse a gene relationship graph from GO ontology annotations"""


import argparse
import csv
import datetime
import itertools
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pybedtools  # type: ignore


def log_progress(message: str) -> None:
    """Print a log message with timestamp to stdout"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{timestamp}] {message}")


def genes_from_gencode(gencode_ref: str) -> Dict[str, str]:
    """Returns a dict of gencode v26 genes, their ids and associated gene
    symbols
    """
    gencode_ref = pybedtools.BedTool(gencode_ref)  # type: ignore
    return {
        line[9].split(";")[3].split('"')[1]: line[3]
        for line in gencode_ref
        if line[0] not in ["chrM"]
    }


def _uniprot_to_gene_symbol(mapfile: Path) -> Dict[str, str]:
    """Get dictionary for mapping Uniprot IDs to Gencode IDs. We keep the first
    gene if there are multiple genes that uniprot maps to."""
    with open(mapfile) as file:
        return {row[0]: row[1] for row in csv.reader(file, delimiter="\t")}


def _uniprot_to_gencode(mapfile: Path, gencode_ref: str) -> Dict[str, str]:
    """Create a mapping from Uniprot to Gencode IDs. We only keep the genes that
    are present in gencode, otherwise the indexes won't be congruent."""
    gencode_mapper = genes_from_gencode(gencode_ref)
    mapper = _uniprot_to_gene_symbol(mapfile)
    return {
        key: gencode_mapper[value]
        for key, value in mapper.items()
        if value in gencode_mapper
    }


def get_go_annotations(go_gaf: Path) -> List[Tuple[str, str]]:
    """Create GO ontology graph"""
    with open(go_gaf, newline="", mode="r") as file:
        reader = csv.reader(file, delimiter="\t")
        return [
            (row[1], row[4])
            for row in reader
            if not row[0].startswith("!") and row[6] not in ["IEA", "IEP", "IC", "ND"]
        ]


def make_go_graph(
    edges: List[Tuple[str, str]], mapper: Dict[str, str]
) -> set[Tuple[str, str]]:
    """Convert edges to gene-to-gene edges, linking genes sharing a GO term"""
    all_edges = set()
    go_to_gene: Dict[str, List[str]] = {}
    for gene, go_term in edges:
        if go_term not in go_to_gene:
            go_to_gene[go_term] = []
        if gene in mapper:
            go_to_gene[go_term].append(mapper[gene])
    for linked_genes in go_to_gene.values():
        for gene_pair in itertools.combinations(linked_genes, 2):
            all_edges.add(gene_pair)
    return all_edges


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Create a graph from gene ontology")
    parser.add_argument("--working_dir", type=str, help="Working directory")
    parser.add_argument("--mapfile", type=str, help="Mapping file")
    parser.add_argument("--go_gaf", type=str, help="GO annotation file")
    parser.add_argument("--gencode_ref", type=str, help="Gencode reference file")
    args = parser.parse_args()
    log_progress(f"Starting with args: {args}")

    # set up vars and paths
    working_dir = Path(args.working_dir)
    mapfile = working_dir / args.mapfile
    go_gaf = working_dir / args.go_gaf
    final_graph = working_dir / "go_graph.txt"
    log_progress(f"Working directory: {working_dir}")
    log_progress(f"Mapfile: {mapfile}")
    log_progress(f"GO GAF file: {go_gaf}")
    log_progress(f"Final graph: {final_graph}")

    # get GO graph!
    mapper = _uniprot_to_gencode(mapfile, args.gencode_ref)
    log_progress(f"Mapper created w/ number of genes: {len(mapper)}")
    go_edges = get_go_annotations(go_gaf)
    log_progress(f"GO edges created w/ number of edges: {len(go_edges)}")
    go_graph = make_go_graph(go_edges, mapper)

    # write to a file and savels
    with open(final_graph, "w") as file:
        for edge in go_graph:
            file.write(f"{edge[0]}\t{edge[1]}\n")


if __name__ == "__main__":
    main()
