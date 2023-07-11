#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] fix params for cores
# - [ ] try and refactor yamls and init


"""Parse local genomic data to nodes and attributes"""

import argparse
import csv
from itertools import repeat
from multiprocessing import Pool
import os
import pickle
import subprocess
from subprocess import PIPE
from subprocess import Popen
from typing import Dict, List, Optional, Tuple

import pybedtools
from pybedtools.featurefuncs import extend_fields

from utils import _listdir_isfile_wrapper
from utils import _tpm_filter_gene_windows
from utils import dir_check_make
from utils import genes_from_gencode
from utils import NODES
from utils import parse_yaml
from utils import time_decorator

ATTRIBUTES = [
    "gc",
    "cnv",
    "cpgislands",
    "indels",
    "line",
    "ltr",
    "microsatellites",
    "phastcons",
    "polyasites",
    "rbpbindingsites",
    "recombination",
    "repg1b",
    "repg2",
    "reps1",
    "reps2",
    "reps3",
    "reps4",
    "rnarepeat",
    "simplerepeats",
    "sine",
    "snp",
    "methylation-10",
    "methylation-11",
    "methylation-12",
    "methylation-13",
    "methylation-14",
    "methylation-15",
    "methylation-16",
    "methylation-17",
    "methylation-18",
    "methylation-19",
    "methylation-1",
    "methylation-20",
    "methylation-21",
    "methylation-22",
    "methylation-23",
    "methylation-24",
    "methylation-25",
    "methylation-26",
    "methylation-27",
    "methylation-28",
    "methylation-29",
    "methylation-2",
    "methylation-30",
    "methylation-31",
    "methylation-32",
    "methylation-33",
    "methylation-34",
    "methylation-35",
    "methylation-36",
    "methylation-37",
    "methylation-3",
    "methylation-4",
    "methylation-5",
    "methylation-6",
    "methylation-7",
    "methylation-8",
    "methylation-9",
    "atac-seq-adipose",
    "atac-seq-bloodandt-cell",
    "atac-seq-bone",
    "atac-seq-brain",
    "atac-seq-digestive",
    "atac-seq-endocrine",
    "atac-seq-endothelial",
    "atac-seq-epithelial",
    "atac-seq-esc",
    "atac-seq-es-deriv",
    "atac-seq-eye",
    "atac-seq-heart",
    "atac-seq-hscandb-cell",
    "atac-seq-ipsc",
    "atac-seq-kidney",
    "atac-seq-liver",
    "atac-seq-lung",
    "atac-seq-lymphoblastoid",
    "atac-seq-mesench",
    "atac-seq-muscle",
    "atac-seq-myosat",
    "atac-seq-neurosph",
    "atac-seq-pancreas",
    "atac-seq-placentaandeem",
    "atac-seq-pns",
    "atac-seq-reproductive",
    "atac-seq-sm",
    "atac-seq-spleen",
    "atac-seq-stromal",
    "atac-seq-thymus",
    "atac-seq-urinary",
    "ctcf-adipose",
    "ctcf-bloodandt-cell",
    "ctcf-bone",
    "ctcf-brain",
    "ctcf-digestive",
    "ctcf-endocrine",
    "ctcf-endothelial",
    "ctcf-epithelial",
    "ctcf-esc",
    "ctcf-es-deriv",
    "ctcf-eye",
    "ctcf-heart",
    "ctcf-hscandb-cell",
    "ctcf-ipsc",
    "ctcf-kidney",
    "ctcf-liver",
    "ctcf-lung",
    "ctcf-lymphoblastoid",
    "ctcf-mesench",
    "ctcf-muscle",
    "ctcf-myosat",
    "ctcf-neurosph",
    "ctcf-pancreas",
    "ctcf-placentaandeem",
    "ctcf-pns",
    "ctcf-reproductive",
    "ctcf-sm",
    "ctcf-spleen",
    "ctcf-stromal",
    "ctcf-thymus",
    "ctcf-urinary",
    "dnase-seq-adipose",
    "dnase-seq-bloodandt-cell",
    "dnase-seq-bone",
    "dnase-seq-brain",
    "dnase-seq-digestive",
    "dnase-seq-endocrine",
    "dnase-seq-endothelial",
    "dnase-seq-epithelial",
    "dnase-seq-esc",
    "dnase-seq-es-deriv",
    "dnase-seq-eye",
    "dnase-seq-heart",
    "dnase-seq-hscandb-cell",
    "dnase-seq-ipsc",
    "dnase-seq-kidney",
    "dnase-seq-liver",
    "dnase-seq-lung",
    "dnase-seq-lymphoblastoid",
    "dnase-seq-mesench",
    "dnase-seq-muscle",
    "dnase-seq-myosat",
    "dnase-seq-neurosph",
    "dnase-seq-pancreas",
    "dnase-seq-placentaandeem",
    "dnase-seq-pns",
    "dnase-seq-reproductive",
    "dnase-seq-sm",
    "dnase-seq-spleen",
    "dnase-seq-stromal",
    "dnase-seq-thymus",
    "dnase-seq-urinary",
    "ep300-adipose",
    "ep300-bloodandt-cell",
    "ep300-bone",
    "ep300-brain",
    "ep300-digestive",
    "ep300-endocrine",
    "ep300-endothelial",
    "ep300-epithelial",
    "ep300-esc",
    "ep300-es-deriv",
    "ep300-eye",
    "ep300-heart",
    "ep300-hscandb-cell",
    "ep300-ipsc",
    "ep300-kidney",
    "ep300-liver",
    "ep300-lung",
    "ep300-lymphoblastoid",
    "ep300-mesench",
    "ep300-muscle",
    "ep300-myosat",
    "ep300-neurosph",
    "ep300-pancreas",
    "ep300-placentaandeem",
    "ep300-pns",
    "ep300-reproductive",
    "ep300-sm",
    "ep300-spleen",
    "ep300-stromal",
    "ep300-thymus",
    "ep300-urinary",
    "h2afz-adipose",
    "h2afz-bloodandt-cell",
    "h2afz-bone",
    "h2afz-brain",
    "h2afz-digestive",
    "h2afz-endocrine",
    "h2afz-endothelial",
    "h2afz-epithelial",
    "h2afz-esc",
    "h2afz-es-deriv",
    "h2afz-eye",
    "h2afz-heart",
    "h2afz-hscandb-cell",
    "h2afz-ipsc",
    "h2afz-kidney",
    "h2afz-liver",
    "h2afz-lung",
    "h2afz-lymphoblastoid",
    "h2afz-mesench",
    "h2afz-muscle",
    "h2afz-myosat",
    "h2afz-neurosph",
    "h2afz-pancreas",
    "h2afz-placentaandeem",
    "h2afz-pns",
    "h2afz-reproductive",
    "h2afz-sm",
    "h2afz-spleen",
    "h2afz-stromal",
    "h2afz-thymus",
    "h2afz-urinary",
    "h3k27ac-adipose",
    "h3k27ac-bloodandt-cell",
    "h3k27ac-bone",
    "h3k27ac-brain",
    "h3k27ac-digestive",
    "h3k27ac-endocrine",
    "h3k27ac-endothelial",
    "h3k27ac-epithelial",
    "h3k27ac-esc",
    "h3k27ac-es-deriv",
    "h3k27ac-eye",
    "h3k27ac-heart",
    "h3k27ac-hscandb-cell",
    "h3k27ac-ipsc",
    "h3k27ac-kidney",
    "h3k27ac-liver",
    "h3k27ac-lung",
    "h3k27ac-lymphoblastoid",
    "h3k27ac-mesench",
    "h3k27ac-muscle",
    "h3k27ac-myosat",
    "h3k27ac-neurosph",
    "h3k27ac-pancreas",
    "h3k27ac-placentaandeem",
    "h3k27ac-pns",
    "h3k27ac-reproductive",
    "h3k27ac-sm",
    "h3k27ac-spleen",
    "h3k27ac-stromal",
    "h3k27ac-thymus",
    "h3k27ac-urinary",
    "h3k27me3-adipose",
    "h3k27me3-bloodandt-cell",
    "h3k27me3-bone",
    "h3k27me3-brain",
    "h3k27me3-digestive",
    "h3k27me3-endocrine",
    "h3k27me3-endothelial",
    "h3k27me3-epithelial",
    "h3k27me3-esc",
    "h3k27me3-es-deriv",
    "h3k27me3-eye",
    "h3k27me3-heart",
    "h3k27me3-hscandb-cell",
    "h3k27me3-ipsc",
    "h3k27me3-kidney",
    "h3k27me3-liver",
    "h3k27me3-lung",
    "h3k27me3-lymphoblastoid",
    "h3k27me3-mesench",
    "h3k27me3-muscle",
    "h3k27me3-myosat",
    "h3k27me3-neurosph",
    "h3k27me3-pancreas",
    "h3k27me3-placentaandeem",
    "h3k27me3-pns",
    "h3k27me3-reproductive",
    "h3k27me3-sm",
    "h3k27me3-spleen",
    "h3k27me3-stromal",
    "h3k27me3-thymus",
    "h3k27me3-urinary",
    "h3k36me3-adipose",
    "h3k36me3-bloodandt-cell",
    "h3k36me3-bone",
    "h3k36me3-brain",
    "h3k36me3-digestive",
    "h3k36me3-endocrine",
    "h3k36me3-endothelial",
    "h3k36me3-epithelial",
    "h3k36me3-esc",
    "h3k36me3-es-deriv",
    "h3k36me3-eye",
    "h3k36me3-heart",
    "h3k36me3-hscandb-cell",
    "h3k36me3-ipsc",
    "h3k36me3-kidney",
    "h3k36me3-liver",
    "h3k36me3-lung",
    "h3k36me3-lymphoblastoid",
    "h3k36me3-mesench",
    "h3k36me3-muscle",
    "h3k36me3-myosat",
    "h3k36me3-neurosph",
    "h3k36me3-pancreas",
    "h3k36me3-placentaandeem",
    "h3k36me3-pns",
    "h3k36me3-reproductive",
    "h3k36me3-sm",
    "h3k36me3-spleen",
    "h3k36me3-stromal",
    "h3k36me3-thymus",
    "h3k36me3-urinary",
    "h3k4me1-adipose",
    "h3k4me1-bloodandt-cell",
    "h3k4me1-bone",
    "h3k4me1-brain",
    "h3k4me1-digestive",
    "h3k4me1-endocrine",
    "h3k4me1-endothelial",
    "h3k4me1-epithelial",
    "h3k4me1-esc",
    "h3k4me1-es-deriv",
    "h3k4me1-eye",
    "h3k4me1-heart",
    "h3k4me1-hscandb-cell",
    "h3k4me1-ipsc",
    "h3k4me1-kidney",
    "h3k4me1-liver",
    "h3k4me1-lung",
    "h3k4me1-lymphoblastoid",
    "h3k4me1-mesench",
    "h3k4me1-muscle",
    "h3k4me1-myosat",
    "h3k4me1-neurosph",
    "h3k4me1-pancreas",
    "h3k4me1-placentaandeem",
    "h3k4me1-pns",
    "h3k4me1-reproductive",
    "h3k4me1-sm",
    "h3k4me1-spleen",
    "h3k4me1-stromal",
    "h3k4me1-thymus",
    "h3k4me1-urinary",
    "h3k4me2-adipose",
    "h3k4me2-bloodandt-cell",
    "h3k4me2-bone",
    "h3k4me2-brain",
    "h3k4me2-digestive",
    "h3k4me2-endocrine",
    "h3k4me2-endothelial",
    "h3k4me2-epithelial",
    "h3k4me2-esc",
    "h3k4me2-es-deriv",
    "h3k4me2-eye",
    "h3k4me2-heart",
    "h3k4me2-hscandb-cell",
    "h3k4me2-ipsc",
    "h3k4me2-kidney",
    "h3k4me2-liver",
    "h3k4me2-lung",
    "h3k4me2-lymphoblastoid",
    "h3k4me2-mesench",
    "h3k4me2-muscle",
    "h3k4me2-myosat",
    "h3k4me2-neurosph",
    "h3k4me2-pancreas",
    "h3k4me2-placentaandeem",
    "h3k4me2-pns",
    "h3k4me2-reproductive",
    "h3k4me2-sm",
    "h3k4me2-spleen",
    "h3k4me2-stromal",
    "h3k4me2-thymus",
    "h3k4me2-urinary",
    "h3k4me3-adipose",
    "h3k4me3-bloodandt-cell",
    "h3k4me3-bone",
    "h3k4me3-brain",
    "h3k4me3-digestive",
    "h3k4me3-endocrine",
    "h3k4me3-endothelial",
    "h3k4me3-epithelial",
    "h3k4me3-esc",
    "h3k4me3-es-deriv",
    "h3k4me3-eye",
    "h3k4me3-heart",
    "h3k4me3-hscandb-cell",
    "h3k4me3-ipsc",
    "h3k4me3-kidney",
    "h3k4me3-liver",
    "h3k4me3-lung",
    "h3k4me3-lymphoblastoid",
    "h3k4me3-mesench",
    "h3k4me3-muscle",
    "h3k4me3-myosat",
    "h3k4me3-neurosph",
    "h3k4me3-pancreas",
    "h3k4me3-placentaandeem",
    "h3k4me3-pns",
    "h3k4me3-reproductive",
    "h3k4me3-sm",
    "h3k4me3-spleen",
    "h3k4me3-stromal",
    "h3k4me3-thymus",
    "h3k4me3-urinary",
    "h3k79me2-adipose",
    "h3k79me2-bloodandt-cell",
    "h3k79me2-bone",
    "h3k79me2-brain",
    "h3k79me2-digestive",
    "h3k79me2-endocrine",
    "h3k79me2-endothelial",
    "h3k79me2-epithelial",
    "h3k79me2-esc",
    "h3k79me2-es-deriv",
    "h3k79me2-eye",
    "h3k79me2-heart",
    "h3k79me2-hscandb-cell",
    "h3k79me2-ipsc",
    "h3k79me2-kidney",
    "h3k79me2-liver",
    "h3k79me2-lung",
    "h3k79me2-lymphoblastoid",
    "h3k79me2-mesench",
    "h3k79me2-muscle",
    "h3k79me2-myosat",
    "h3k79me2-neurosph",
    "h3k79me2-pancreas",
    "h3k79me2-placentaandeem",
    "h3k79me2-pns",
    "h3k79me2-reproductive",
    "h3k79me2-sm",
    "h3k79me2-spleen",
    "h3k79me2-stromal",
    "h3k79me2-thymus",
    "h3k79me2-urinary",
    "h3k9ac-adipose",
    "h3k9ac-bloodandt-cell",
    "h3k9ac-bone",
    "h3k9ac-brain",
    "h3k9ac-digestive",
    "h3k9ac-endocrine",
    "h3k9ac-endothelial",
    "h3k9ac-epithelial",
    "h3k9ac-esc",
    "h3k9ac-es-deriv",
    "h3k9ac-eye",
    "h3k9ac-heart",
    "h3k9ac-hscandb-cell",
    "h3k9ac-ipsc",
    "h3k9ac-kidney",
    "h3k9ac-liver",
    "h3k9ac-lung",
    "h3k9ac-lymphoblastoid",
    "h3k9ac-mesench",
    "h3k9ac-muscle",
    "h3k9ac-myosat",
    "h3k9ac-neurosph",
    "h3k9ac-pancreas",
    "h3k9ac-placentaandeem",
    "h3k9ac-pns",
    "h3k9ac-reproductive",
    "h3k9ac-sm",
    "h3k9ac-spleen",
    "h3k9ac-stromal",
    "h3k9ac-thymus",
    "h3k9ac-urinary",
    "h3k9me3-adipose",
    "h3k9me3-bloodandt-cell",
    "h3k9me3-bone",
    "h3k9me3-brain",
    "h3k9me3-digestive",
    "h3k9me3-endocrine",
    "h3k9me3-endothelial",
    "h3k9me3-epithelial",
    "h3k9me3-esc",
    "h3k9me3-es-deriv",
    "h3k9me3-eye",
    "h3k9me3-heart",
    "h3k9me3-hscandb-cell",
    "h3k9me3-ipsc",
    "h3k9me3-kidney",
    "h3k9me3-liver",
    "h3k9me3-lung",
    "h3k9me3-lymphoblastoid",
    "h3k9me3-mesench",
    "h3k9me3-muscle",
    "h3k9me3-myosat",
    "h3k9me3-neurosph",
    "h3k9me3-pancreas",
    "h3k9me3-placentaandeem",
    "h3k9me3-pns",
    "h3k9me3-reproductive",
    "h3k9me3-sm",
    "h3k9me3-spleen",
    "h3k9me3-stromal",
    "h3k9me3-thymus",
    "h3k9me3-urinary",
    "h4k20me1-adipose",
    "h4k20me1-bloodandt-cell",
    "h4k20me1-bone",
    "h4k20me1-brain",
    "h4k20me1-digestive",
    "h4k20me1-endocrine",
    "h4k20me1-endothelial",
    "h4k20me1-epithelial",
    "h4k20me1-esc",
    "h4k20me1-es-deriv",
    "h4k20me1-eye",
    "h4k20me1-heart",
    "h4k20me1-hscandb-cell",
    "h4k20me1-ipsc",
    "h4k20me1-kidney",
    "h4k20me1-liver",
    "h4k20me1-lung",
    "h4k20me1-lymphoblastoid",
    "h4k20me1-mesench",
    "h4k20me1-muscle",
    "h4k20me1-myosat",
    "h4k20me1-neurosph",
    "h4k20me1-pancreas",
    "h4k20me1-placentaandeem",
    "h4k20me1-pns",
    "h4k20me1-reproductive",
    "h4k20me1-sm",
    "h4k20me1-spleen",
    "h4k20me1-stromal",
    "h4k20me1-thymus",
    "h4k20me1-urinary",
    "polr2a-adipose",
    "polr2a-bloodandt-cell",
    "polr2a-bone",
    "polr2a-brain",
    "polr2a-digestive",
    "polr2a-endocrine",
    "polr2a-endothelial",
    "polr2a-epithelial",
    "polr2a-esc",
    "polr2a-es-deriv",
    "polr2a-eye",
    "polr2a-heart",
    "polr2a-hscandb-cell",
    "polr2a-ipsc",
    "polr2a-kidney",
    "polr2a-liver",
    "polr2a-lung",
    "polr2a-lymphoblastoid",
    "polr2a-mesench",
    "polr2a-muscle",
    "polr2a-myosat",
    "polr2a-neurosph",
    "polr2a-pancreas",
    "polr2a-placentaandeem",
    "polr2a-pns",
    "polr2a-reproductive",
    "polr2a-sm",
    "polr2a-spleen",
    "polr2a-stromal",
    "polr2a-thymus",
    "polr2a-urinary",
    "rad21-adipose",
    "rad21-bloodandt-cell",
    "rad21-bone",
    "rad21-brain",
    "rad21-digestive",
    "rad21-endocrine",
    "rad21-endothelial",
    "rad21-epithelial",
    "rad21-esc",
    "rad21-es-deriv",
    "rad21-eye",
    "rad21-heart",
    "rad21-hscandb-cell",
    "rad21-ipsc",
    "rad21-kidney",
    "rad21-liver",
    "rad21-lung",
    "rad21-lymphoblastoid",
    "rad21-mesench",
    "rad21-muscle",
    "rad21-myosat",
    "rad21-neurosph",
    "rad21-pancreas",
    "rad21-placentaandeem",
    "rad21-pns",
    "rad21-reproductive",
    "rad21-sm",
    "rad21-spleen",
    "rad21-stromal",
    "rad21-thymus",
    "rad21-urinary",
    "smc3-adipose",
    "smc3-bloodandt-cell",
    "smc3-bone",
    "smc3-brain",
    "smc3-digestive",
    "smc3-endocrine",
    "smc3-endothelial",
    "smc3-epithelial",
    "smc3-esc",
    "smc3-es-deriv",
    "smc3-eye",
    "smc3-heart",
    "smc3-hscandb-cell",
    "smc3-ipsc",
    "smc3-kidney",
    "smc3-liver",
    "smc3-lung",
    "smc3-lymphoblastoid",
    "smc3-mesench",
    "smc3-muscle",
    "smc3-myosat",
    "smc3-neurosph",
    "smc3-pancreas",
    "smc3-placentaandeem",
    "smc3-pns",
    "smc3-reproductive",
    "smc3-sm",
    "smc3-spleen",
    "smc3-stromal",
    "smc3-thymus",
    "smc3-urinary",
]


class LocalContextParser:
    """Object that parses local genomic data into graph edges

    Args:
        bedfiles // dictionary containing each local genomic data    type as bedtool
            obj
        windows // bedtool object of windows +/- 250k of protein coding genes
        params // configuration vals from yaml

    Methods
    ----------
    _make_directories:
        prepare necessary directories

    # Helpers
        ATTRIBUTES -- list of node attribute types
        DIRECT -- list of datatypes that only get direct overlaps, no slop
        NODES -- list of nodetypes
        ONEHOT_NODETYPE -- dictionary of node type one-hot vectors
    """

    # list helpers
    DIRECT = ["tads"]
    NODE_FEATS = ["start", "end", "size"] + ATTRIBUTES

    # var helpers - for CPU cores
    NODE_CORES = len(NODES) + 1  # 12
    ATTRIBUTE_CORES = len(ATTRIBUTES)  # 3

    def __init__(
        self,
        bedfiles: List[str],
        params: Dict[str, Dict[str, str]],
    ):
        """Initialize the class"""
        self.tissue = "universalgenome"

        self.bedfiles = bedfiles
        self.resources = params["resources"]
        self.tissue_specific = params["tissue_specific"]
        self.gencode = params["local"]["gencode"]
        self.chromfile = self.resources["chromfile"]
        self.fasta = self.resources["fasta"]

        self.root_dir = params["dirs"]["root_dir"]
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
        self.local_dir = f"{self.tissue_dir}/local"
        self.parse_dir = f"{self.tissue_dir}/parsing"
        self.attribute_dir = f"{self.parse_dir}/attributes"

        genes = f"{self.tissue_dir}/tpm_filtered_genes.bed"
        gene_windows = f"{self.tissue_dir}/tpm_filtered_gene_regions.bed"

        # prepare list of genes passing tpm filter
        if not (os.path.exists(genes) and os.stat(genes).st_size > 0):
            self._prepare_tpm_filtered_genes(
                genes=genes,
                gene_windows=gene_windows,
                base_nodes=f"{self.tissue_dir}/local/basenodes_hg38.txt",
            )

        # prepare references
        self.gencode_ref = pybedtools.BedTool(genes)
        self.gene_windows = pybedtools.BedTool(gene_windows)
        self.genesymbol_to_gencode = genes_from_gencode(
            pybedtools.BedTool(f"{self.tissue_dir}/local/{self.gencode}")
        )

        # make directories
        self._make_directories()

    def _prepare_tpm_filtered_genes(
        self, genes: str, gene_windows: str, base_nodes: str
    ) -> None:
        """Prepare tpm filtered genes and gene windows"""
        filtered_genes = _tpm_filter_gene_windows(
            gencode=f"{self.root_dir}/shared_data/local/{self.gencode}",
            tissue=self.tissue,
            tpm_file=self.resources["tpm"],
            chromfile=self.chromfile,
            slop=False,
        )

        windows = pybedtools.BedTool(base_nodes).slop(g=self.chromfile, b=25000).sort()
        filtered_genes.saveas(genes)
        windows.saveas(gene_windows)

    def _make_directories(self) -> None:
        """Directories for parsing genomic bedfiles into graph edges and nodes"""
        dir_check_make(self.parse_dir)

        for directory in [
            "edges/genes",
            "attributes",
            "intermediate/slopped",
            "intermediate/sorted",
        ]:
            dir_check_make(f"{self.parse_dir}/{directory}")

        for attribute in ATTRIBUTES:
            dir_check_make(f"{self.attribute_dir}/{attribute}")

    @time_decorator(print_args=True)
    def _region_specific_features_dict(
        self, bed: str
    ) -> List[Dict[str, pybedtools.bedtool.BedTool]]:
        """
        Creates a dict of local context datatypes and their bedtools objects.
        Renames features if necessary.
        """

        def rename_feat_chr_start(feature: str) -> str:
            """Add chr, start to feature name
            Cpgislands add prefix to feature names  # enhancers,
            Histones add an additional column
            """
            simple_rename = [
                "cpgislands",
                "crms",
            ]
            if prefix in simple_rename:
                feature = extend_fields(feature, 4)
                feature[3] = f"{feature[0]}_{feature[1]}_{prefix}"
            else:
                feature[3] = f"{feature[0]}_{feature[1]}_{feature[3]}"
            return feature

        # prepare data as pybedtools objects
        bed_dict = {}
        prefix = bed.split("_")[0].lower()
        a = self.gene_windows
        b = pybedtools.BedTool(f"{self.root_dir}/{self.tissue}/local/{bed}").sort()
        ab = b.intersect(a, sorted=True, u=True)

        # take specific windows and format each file
        if prefix in NODES and prefix != "gencode":
            result = ab.each(rename_feat_chr_start).cut([0, 1, 2, 3]).saveas()
            bed_dict[prefix] = pybedtools.BedTool(str(result), from_string=True)
        else:
            bed_dict[prefix] = ab.cut([0, 1, 2, 3])

        return bed_dict

    @time_decorator(print_args=True)
    def _slop_sort(
        self, bedinstance: Dict[str, str], chromfile: str, feat_window: int = 2000
    ) -> Tuple[
        Dict[str, pybedtools.bedtool.BedTool], Dict[str, pybedtools.bedtool.BedTool]
    ]:
        """Slop each line of a bedfile to get all features within a window

        Args:
            bedinstance // a region-filtered genomic bedfile
            chromfile // textfile with sizes of each chromosome in hg38

        Returns:
            bedinstance_sorted -- sorted bed
            bedinstance_slopped -- bed slopped by amount in feat_window
        """
        bedinstance_slopped, bedinstance_sorted = {}, {}
        for key in bedinstance.keys():
            bedinstance_sorted[key] = bedinstance[key].sort()
            if key in ATTRIBUTES + self.DIRECT:
                pass
            else:
                nodes = bedinstance[key].slop(g=chromfile, b=feat_window).sort()
                newstrings = []
                for line_1, line_2 in zip(nodes, bedinstance[key]):
                    newstrings.append(str(line_1).split("\n")[0] + "\t" + str(line_2))
                bedinstance_slopped[key] = pybedtools.BedTool(
                    "".join(newstrings), from_string=True
                ).sort()
        return bedinstance_sorted, bedinstance_slopped

    @time_decorator(print_args=True)
    def _bed_intersect(self, node_type: str, all_files: str) -> None:
        """Function to intersect a slopped bed entry with all other node types.
        Each bed is slopped then intersected twice. First, it is intersected
        with every other node type. Then, the intersected bed is filtered to
        only keep edges within the gene region.

        Args:
            node_type // _description_
            all_files // _description_

        Raises:
            AssertionError: _description_
        """
        print(f"starting combinations {node_type}")

        def _unix_intersect(node_type: str, type: Optional[str] = None) -> None:
            """Intersect and cut relevant columns"""
            if type == "direct":
                folder = "sorted"
                cut_cmd = ""
            else:
                folder = "slopped"
                cut_cmd = " | cut -f5,6,7,8,9,10,11,12"

            final_cmd = f"bedtools intersect \
                -wa \
                -wb \
                -sorted \
                -a {self.parse_dir}/intermediate/{folder}/{node_type}.bed \
                -b {all_files}"

            with open(f"{self.parse_dir}/edges/{node_type}.bed", "w") as outfile:
                subprocess.run(final_cmd + cut_cmd, stdout=outfile, shell=True)
            outfile.close()

        def _filter_duplicate_bed_entries(
            bedfile: pybedtools.bedtool.BedTool,
        ) -> pybedtools.bedtool.BedTool:
            """Filters a bedfile by removing entries that are identical"""
            return bedfile.filter(
                lambda x: [x[0], x[1], x[2], x[3]] != [x[4], x[5], x[6], x[7]]
            ).saveas()

        def _add_distance(feature: str) -> str:
            """Add distance as [8]th field to each overlap interval"""
            feature = extend_fields(feature, 9)
            feature[8] = max(int(feature[1]), int(feature[5])) - min(
                int(feature[2]), int(feature[5])
            )
            return feature

        if node_type in self.DIRECT:
            _unix_intersect(node_type, type="direct")
            _filter_duplicate_bed_entries(
                pybedtools.BedTool(f"{self.parse_dir}/edges/{node_type}.bed")
            ).sort().saveas(f"{self.parse_dir}/edges/{node_type}_dupes_removed")
        else:
            _unix_intersect(node_type)
            _filter_duplicate_bed_entries(
                pybedtools.BedTool(f"{self.parse_dir}/edges/{node_type}.bed")
            ).each(_add_distance).saveas().sort().saveas(
                f"{self.parse_dir}/edges/{node_type}_dupes_removed"
            )

    @time_decorator(print_args=True)
    def _aggregate_attributes(self, node_type: str) -> None:
        """For each node of a node_type get their overlap with gene windows then
        aggregate total nucleotides, gc content, and all other attributes

        Args:
            node_type // node datatype in self.NODES
        """

        def add_size(feature: str) -> str:
            """ """
            feature = extend_fields(feature, 5)
            feature[4] = feature.end - feature.start
            return feature

        def sum_gc(feature: str) -> str:
            """ """
            feature[13] = int(feature[8]) + int(feature[9])
            return feature

        ref_file = pybedtools.BedTool(
            f"{self.parse_dir}/intermediate/sorted/{node_type}.bed"
        )
        ref_file = (
            ref_file.filter(lambda x: "alt" not in x[0]).each(add_size).sort().saveas()
        )

        # for attribute in ATTRIBUTES:
        for attribute in ["gc"]:
            save_file = (
                f"{self.attribute_dir}/{attribute}/{node_type}_{attribute}_percentage"
            )
            print(f"{attribute} for {node_type}")
            if attribute == "gc":
                ref_file.nucleotide_content(fi=self.fasta).each(sum_gc).sort().groupby(
                    g=[1, 2, 3, 4], c=[5, 14], o=["sum"]
                ).saveas(save_file)
            elif attribute == "recombination":
                ref_file.intersect(
                    f"{self.parse_dir}/intermediate/sorted/{attribute}.bed",
                    wao=True,
                    sorted=True,
                ).groupby(g=[1, 2, 3, 4], c=[5, 9], o=["sum", "mean"]).sort().saveas(
                    save_file
                )
            else:
                try:
                    ref_file.intersect(
                        f"{self.parse_dir}/intermediate/sorted/{attribute}.bed",
                        wao=True,
                        sorted=True,
                    ).groupby(g=[1, 2, 3, 4], c=[5, 10], o=["sum"]).sort().saveas(
                        save_file
                    )
                except pybedtools.helpers.BEDToolsError:
                    print(f"broken {attribute} for {node_type}")

    @time_decorator(print_args=True)
    def _generate_edges(self) -> None:
        """Unix concatenate and sort each edge file"""

        def _chk_file_and_run(file: str, cmd: str) -> None:
            """Check that a file does not exist before calling subprocess"""
            if os.path.isfile(file) and os.path.getsize(file) != 0:
                pass
            else:
                subprocess.run(cmd, stdout=None, shell=True)

        cmds = {
            "cat_cmd": [
                f"cat {self.parse_dir}/edges/*_dupes_removed* >",
                f"{self.parse_dir}/edges/all_concat.bed",
            ],
            "sort_cmd": [
                f"LC_ALL=C sort --parallel=32 -S 80% -k10,10 {self.parse_dir}/edges/all_concat.bed |",
                "uniq >" f"{self.parse_dir}/edges/all_concat_sorted.bed",
            ],
        }

        for cmd in cmds:
            _chk_file_and_run(
                cmds[cmd][1],
                cmds[cmd][0] + cmds[cmd][1],
            )

    @time_decorator(print_args=True)
    def _save_node_attributes(self, node: str) -> None:
        """
        Save attributes for all node entries. Used during graph construction for
        gene_nodes that fall outside of the gene window and for some gene_nodes
        from interaction data
        """

        attr_dict, attr_dict_nochr, set_dict = (
            {},
            {},
            {},
        )  # dict[gene] = [chr, start, end, size, gc]
        for attribute in ATTRIBUTES:
            filename = (
                f"{self.parse_dir}/attributes/{attribute}/{node}_{attribute}_percentage"
            )
            with open(filename, "r") as file:
                lines = [tuple(line.rstrip().split("\t")) for line in file]
                set_dict[attribute] = set(lines)
            for line in set_dict[attribute]:
                if attribute == "gc":
                    attr_dict[f"{line[3]}_{self.tissue}"] = {
                        "chr": line[0].replace("chr", ""),
                    }
                    for dictionary in [attr_dict, attr_dict_nochr]:
                        dictionary[f"{line[3]}_{self.tissue}"] = {
                            "start": float(line[1]),
                            "end": float(line[2]),
                            "size": float(line[4]),
                            "gc": float(line[5]),
                        }
                else:
                    try:
                        for dictionary in [attr_dict, attr_dict_nochr]:
                            dictionary[f"{line[3]}_{self.tissue}"][attribute] = float(
                                line[5]
                            )
                    except ValueError:
                        for dictionary in [attr_dict, attr_dict_nochr]:
                            dictionary[f"{line[3]}_{self.tissue}"][attribute] = 0

        with open(f"{self.parse_dir}/attributes/{node}_reference.pkl", "wb") as output:
            pickle.dump(attr_dict, output)

        with open(
            f"{self.parse_dir}/attributes/{node}_reference_nochr.pkl", "wb"
        ) as output:
            pickle.dump(attr_dict_nochr, output)

    @time_decorator(print_args=True)
    def parse_context_data(self) -> None:
        """_summary_

        Args:
            a // _description_
            b // _description_

        Raises:
            AssertionError: _description_

        Returns:
            c -- _description_
        """

        # @time_decorator(print_args=True)
        # def _save_intermediate(
        #     bed_dictionary: Dict[str, pybedtools.bedtool.BedTool], folder: str
        # ) -> None:
        #     """Save region specific bedfiles"""
        #     for key in bed_dictionary:
        #         file = f"{self.parse_dir}/intermediate/{folder}/{key}.bed"
        #         if not os.path.exists(file):
        #             bed_dictionary[key].saveas(file)

        # @time_decorator(print_args=True)
        # def _pre_concatenate_all_files(all_files: str) -> None:
        #     """Lorem Ipsum"""
        #     if not os.path.exists(all_files) or os.stat(all_files).st_size == 0:
        #         cat_cmd = ["cat"] + [
        #             f"{self.parse_dir}/intermediate/sorted/" + x + ".bed"
        #             for x in bedinstance_slopped
        #         ]
        #         sort_cmd = "sort -k1,1 -k2,2n"
        #         concat = Popen(cat_cmd, stdout=PIPE)
        #         with open(all_files, "w") as outfile:
        #             subprocess.run(
        #                 sort_cmd, stdin=concat.stdout, stdout=outfile, shell=True
        #             )
        #         outfile.close()

        # process windows and renaming
        # pool = Pool(processes=self.NODE_CORES)
        # bedinstance = pool.map(
        #     self._region_specific_features_dict, [bed for bed in self.bedfiles]
        # )
        # pool.close()  # re-open and close pool after every multi-process

        # # convert back to dictionary
        # bedinstance = {
        #     key.casefold(): value
        #     for element in bedinstance
        #     for key, value in element.items()
        # }

        # # sort and extend windows according to FEAT_WINDOWS
        # bedinstance_sorted, bedinstance_slopped = self._slop_sort(
        #     bedinstance=bedinstance,
        #     chromfile=self.chromfile,
        #     feat_window=2000,
        # )

        # save intermediate files
        # _save_intermediate(bedinstance_sorted, folder="sorted")
        # _save_intermediate(bedinstance_slopped, folder="slopped")

        # pre-concatenate to save time
        # all_files = f"{self.parse_dir}/intermediate/sorted/all_files_concatenated.bed"
        # _pre_concatenate_all_files(all_files)

        # # perform intersects across all feature types - one process per nodetype
        # pool = Pool(processes=self.NODE_CORES)
        # pool.starmap(self._bed_intersect, zip(NODES, repeat(all_files)))
        # pool.close()

        # # get size and all attributes - one process per nodetype
        pool = Pool(processes=self.ATTRIBUTE_CORES)
        pool.map(self._aggregate_attributes, ["basenodes"] + NODES)
        pool.close()

        # parse edges into individual files
        # self._generate_edges()

        # save node attributes as reference for later - one process per nodetype
        pool = Pool(processes=self.ATTRIBUTE_CORES)
        pool.map(self._save_node_attributes, ["basenodes"] + NODES)
        pool.close()


def main() -> None:
    """Pipeline to parse genomic data into edges"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, help="Path to .yaml file with filenames")

    args = parser.parse_args()
    params = parse_yaml(args.config)

    bedfiles = _listdir_isfile_wrapper(
        dir=f"{params['dirs']['root_dir']}/{params['resources']['tissue']}/local",
    )
    bedfiles = [x for x in bedfiles if "chromatinloops" not in x]

    # instantiate object
    localparseObject = LocalContextParser(
        bedfiles=bedfiles,
        params=params,
    )

    # run parallelized pipeline!
    localparseObject.parse_context_data()

    # cleanup temporary files
    pybedtools.cleanup(remove_all=True)


if __name__ == "__main__":
    main()
