#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Constants for omics graph learning modules."""

from typing import Dict, List

HG38_TOTAL_BASEPAIRS = 3088269227

ATTRIBUTES: List[str] = [
    "gc",
    "atac",
    "cnv",
    "cpg",
    "ctcf",
    "dnase",
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
    "h3k79me2",
    "h3k9ac",
    "h3k9me3",
    "indels",
    "line",
    "ltr",
    "microsatellites",
    "phastcons",
    "polr2a",
    "polyasites",
    "rad21",
    "rbpsiteclusters",
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
    "smc3",
    "snp",
]

NODE_FEAT_IDXS: Dict[str, int] = {
    "size": 0,
    "gc": 1,
    "atac": 2,
    "cnv": 3,
    "cpg": 4,
    "ctcf": 5,
    "dnase": 6,
    "h3k27ac": 7,
    "h3k27me3": 8,
    "h3k36me3": 9,
    "h3k4me1": 10,
    "h3k4me2": 11,
    "h3k4me3": 12,
    "h3k79me2": 13,
    "h3k9ac": 14,
    "h3k9me3": 15,
    "indels": 16,
    "line": 17,
    "ltr": 18,
    "microsatellites": 19,
    "phastcons": 20,
    "polr2a": 21,
    "polyasites": 22,
    "rad21": 23,
    "rbpbindingsites": 24,
    "recombination": 25,
    "repg1b": 26,
    "repg2": 27,
    "reps1": 28,
    "reps2": 29,
    "reps3": 30,
    "reps4": 31,
    "rnarepeat": 32,
    "simplerepeats": 33,
    "sine": 34,
    "smc3": 35,
    "snp": 36,
}

POSSIBLE_NODES: List[str] = [
    "cpgislands",
    "crms",
    "ctcfccre",
    "dyadic",
    "enhancers",
    "gencode",
    "promoters",
    "superenhancers",
    "tads",
    "tfbindingsites",
    "tss",
]

TISSUES: List[str] = [
    "aorta",
    "hippocampus",
    "left_ventricle",
    "liver",
    "lung",
    "mammary",
    "pancreas",
    "skeletal_muscle",
    "skin",
    "small_intestine",
    # "hela",
    # "k562",
    # "npc",
]

REGULATORY_ELEMENTS: Dict[str, Dict[str, str]] = {
    "intersect": {
        "enhancer": "enhancer_epimap_screen_overlap.bed",
        "promoter": "promoter_epimap_screen_overlap.bed",
        "dyadic": "dyadic_epimap_screen_overlap.bed",
    },
    "union": {
        "enhancer": "enhancer_all_union_hg38.bed",
        "promoter": "promoter_all_union_hg38.bed",
        "dyadic": "DYADIC_masterlist_locations._lifted_hg38.bed",
    },
    "epimap": {
        "enhancer": "ENH_masterlist_locations._lifted_hg38.bed",
        "promoter": "PROM_masterlist_locations._lifted_hg38.bed",
        "dyadic": "DYADIC_masterlist_locations._lifted_hg38.bed",
    },
    "encode": {
        "enhancer": "GRCh38-ELS.bed",
        "promoter": "GRCh38-PLS.bed",
    },
}
