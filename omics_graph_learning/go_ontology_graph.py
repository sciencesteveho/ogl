#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
# Create a graph of the Gene Ontology (GO) ontology
# Convert GO names to gencode IDs


"""_summary_ of project"""

import csv
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _gencode_genes(gencode_file: str) -> List[str]:
    """Get list of gencode genes"""
    df = pd.read_csv(gencode_file, sep="\t", header=None)
    return list(df[3].unique())


def _unipro_to_gencode_mapper(mapfile: Path) -> Dict[str, str]:
    """Get dictionary for mapping Uniprot IDs to Gencode IDs. We keep the first
    gene if there are multiple genes that uniprot maps to."""
    df = pd.read_csv(mapfile, sep="\t", header=None)
    mapper = {
        row[0]: row[18] for row in df.itertuples(index=False) if not pd.isna(row[18])
    }
    for key, value in mapper.items():
        if ";" in value:
            genes = value.split("; ")
            mapper[key] = genes[0]
    return mapper


def _create_go_graph(
    go_graph: pathlib.PosixPath, mapper: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Create GO ontology graph"""
    reader = csv.reader(open(go_graph, newline=""), delimiter="\t")
    edges = []
    for row in reader:
        if row[6] != "IEA":
            go_id = row[4]
            if go_id in mapper:
                db_object_id = row[1]
                edges.append((mapper[db_object_id], go_id))
    return edges


def _gene_to_gene_edges(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Convert edges to gene-to-gene edges, but linking two genes if they share a GO term"""
    # Explicitly set vars
    go_to_gene: Dict[str, List[str]] = {}
    gene_edges: List[Tuple[str, str]] = []

    for gene, go in edges:
        if go not in go_to_gene:
            go_to_gene[go] = []
        go_to_gene[go].append(gene)

    for genes in go_to_gene.values():
        for i, gene1 in enumerate(genes):
            gene_edges.extend((gene1, gene2) for gene2 in genes[i + 1 :])
    return gene_edges


def main() -> None:
    """Main function"""
    working_dir = Path(
        "/ocean/projects/bio210019p/stevesho/data/preprocess/auxiliary_graphs/go"
    )
    mapfile = working_dir / "HUMAN_9606_idmapping_selected.tab"
    go_graph = working_dir / "goa_human.gaf"
    gencode = "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"

    gencode_genes = _gencode_genes(gencode)
    mapper = _unipro_to_gencode_mapper(mapfile=mapfile)
    go_edges = _create_go_graph(go_graph, mapper)


if __name__ == "__main__":
    main()
