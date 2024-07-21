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
    "smc3",
    "snp",
]

NODE_FEAT_IDXS: Dict[str, int] = {
    "start": 0,
    "end": 1,
    "size": 2,
    "gc": 3,
    "atac": 4,
    "cnv": 5,
    "cpg": 6,
    "ctcf": 7,
    "dnase": 8,
    "h3k27ac": 9,
    "h3k27me3": 10,
    "h3k36me3": 11,
    "h3k4me1": 12,
    "h3k4me2": 13,
    "h3k4me3": 14,
    "h3k79me2": 15,
    "h3k9ac": 16,
    "h3k9me3": 17,
    "indels": 18,
    "line": 19,
    "ltr": 20,
    "microsatellites": 21,
    "phastcons": 22,
    "polr2a": 23,
    "polyasites": 24,
    "rad21": 25,
    "rbpbindingsites": 26,
    "recombination": 27,
    "repg1b": 28,
    "repg2": 29,
    "reps1": 30,
    "reps2": 31,
    "reps3": 32,
    "reps4": 33,
    "rnarepeat": 34,
    "simplerepeats": 35,
    "sine": 36,
    "smc3": 37,
    "snp": 38,
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
