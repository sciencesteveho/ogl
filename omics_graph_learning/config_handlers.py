#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to handle the storing and sharing data from omics graph learning config
files."""

from dataclasses import dataclass
from dataclasses import field
import pathlib
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    baseloop_directory: pathlib.Path
    baseloops: str
    experiment_name: str
    feat_window: int
    gene_gene: bool
    interaction_types: Optional[Any]  # Replace Any with a more specific type if needed
    loop_resolution: float
    nodes: Optional[Any]  # Replace Any with a more specific type if needed
    regulatory: str
    tissues: List[str]
    average_activity_df: pathlib.Path
    config_dir: pathlib.Path
    expression_median_across_all: pathlib.Path
    expression_median_matrix: pathlib.Path
    expression_all_matrix: pathlib.Path
    gencode_gtf: pathlib.Path
    matrix_dir: pathlib.Path
    protein_abundance_matrix: pathlib.Path
    protein_abundance_medians: pathlib.Path
    tpm_dir: pathlib.Path
    test_chrs: List[str]
    val_chrs: List[str]
    working_directory: pathlib.Path
    # Default attributes if needed
    node_types: List[str] = field(
        default_factory=lambda: ["dyadic", "enhancers", "gencode", "promoters"]
    )


@dataclass
class TissueConfig:
    # Directories
    circuit_dir: pathlib.Path
    data_dir: pathlib.Path
    root_dir: pathlib.Path

    # Features
    features: Dict[str, str]

    # Interaction
    interaction: Dict[str, str]

    # Local
    local: Dict[str, str]

    # Methylation
    methylation: Dict[str, str]

    # References
    references: Dict[str, str]

    # Resources
    blacklist: pathlib.Path
    chromfile: pathlib.Path
    fasta: pathlib.Path
    key_protein_abundance: str
    key_tpm: str
    liftover: pathlib.Path
    liftover_chain: pathlib.Path
    marker_name: str
    ppi_tissue: str
    tf_motifs: pathlib.Path
    tissue: str
    tissue_name: str
    tpm: pathlib.Path
    rna: pathlib.Path

    # Tissue specific nodes
    tissue_specific_nodes: Dict[str, str]
