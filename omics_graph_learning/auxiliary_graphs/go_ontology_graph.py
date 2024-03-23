#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
# Create a graph of the Gene Ontology (GO) ontology
# Convert GO names to gencode IDs


"""_summary_ of project"""

import argparse
import csv
import itertools
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple

import pybedtools


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
    return {row[0]: row[1] for row in csv.reader(open(mapfile), delimiter="\t")}


def _uniprot_to_gencode(mapfile: Path, gencode_ref: str) -> Dict[str, str]:
    """Create a mapping from Uniprot to Gencode IDs. We only keep the genes that
    are present in gencode, otherwise the indexes won't be congruent."""
    gencode_mapper = genes_from_gencode(gencode_ref)
    mapper = _uniprot_to_gene_symbol(mapfile)
    return {
        key: gencode_mapper[value]
        for key, value in mapper.items()
        if value in gencode_ref
    }


def _create_go_graph(go_gaf: Path) -> List[Tuple[str, str]]:
    """Create GO ontology graph"""
    with open(go_gaf, newline="", mode="r") as file:
        reader = csv.reader(file, delimiter="\t")
        return [
            (row[1], row[4])
            for row in reader
            if not row[0].startswith("!") and row[6] != "IEA"
        ]


def _gene_to_gene_edges(
    edges: List[Tuple[str, str]], mapper: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Convert edges to gene-to-gene edges, but linking two genes if they share a GO term"""
    # Explicitly set vars
    go_to_gene: Dict[str, List[str]] = {}
    gene_edges: List[Tuple[str, str]] = []

    # make a list of genes for each go-term
    for gene, go_term in edges:
        if go_term not in go_to_gene:
            go_to_gene[go_term] = []
        if gene in mapper:
            go_to_gene[go_term].append(mapper[gene])

    # get all possible pairs for each go-term list
    for linked_genes in go_to_gene.values():
        gene_edges.extend(itertools.combinations(linked_genes, 2))
    return gene_edges


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Create a graph from gene ontology")
    parser.add_argument("--working_dir", type=str, help="Working directory")
    parser.add_argument("--mapfile", type=str, help="Mapping file")
    parser.add_argument("--go_gaf", type=str, help="GO annotation file")
    parser.add_argument("--gencode_ref", type=str, help="Gencode reference file")
    args = parser.parse_args()

    # set up vars and paths
    working_dir = Path(args.working_dir)
    mapfile = working_dir / args.mapfile
    go_gaf = working_dir / args.go_gaf
    final_graph = working_dir / "go_graph.txt"

    # get GO graph!
    mapper = _uniprot_to_gencode(mapfile, args.gencode_ref)
    go_edges = _create_go_graph(go_gaf)
    go_graph = _gene_to_gene_edges(go_edges, mapper)

    # write to a file and save
    with open(final_graph, "w") as file:
        for edge in go_graph:
            file.write(f"{edge[0]}\t{edge[1]}\n")


if __name__ == "__main__":
    main()
