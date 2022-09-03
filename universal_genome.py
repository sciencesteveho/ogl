#! /usr/bin/env python
# -*- coding: utf-8 -*-  
#
# // TO-DO //
# - [ ]
#

"""Code to preprocess bedfiles into graph structures for the universal_genome graphs. 
"""

# import os
# import argparse
# import requests
# import subprocess

# import pybedtools

# from typing import Dict

# from pybedtools.featurefuncs import extend_fields

# from utils import dir_check_make, parse_yaml, time_decorator

ATTRIBUTES = ['gc', 'cpg', 'ctcf', 'dnase', 'enh', 'enhbiv', 'enhg', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'line', 'ltr', 'microsatellites', 'phastcons', 'polr2a', 'rnarepeat', 'simplerepeats', 'sine', 'tssa', 'tssaflnk', 'tssbiv', 'txflnk', 'tx', 'txwk', 'znf']
DIRECT = ['chromatinloops', 'tads']
NODES = ['chromatinloops', 'cpgislands', 'enhancers', 'gencode', 'histones', 'mirnatargets', 'polyasites', 'promoters', 'rbpbindingsites', 'tads', 'tfbindingclusters', 'tss']


['enhwk', 'enha', 'txenh', 'txwk', 'tx', 'txex', 'znf', 'bivprom', 'prom', 'tss']

def _split_full_stack_chromhmm(dir: str, samples: int):
    """
    """
    states = {
        'acetylation': 'Acet',
        'weakenhancer': 'EnhWk',
        'enhancer': 'EnhA',
        'transcribedenhancer': 'TxEnh',
        'weaktranscription': 'TxWk',
        'transcription': 'Tx[0-9]',
        'state': [],
        'state': [],
        'state': [],
        'state': [],
    }
    
def _concensus_tads(dir: str, samples: int):
    """
    """

def _concensus_chromatinloops(dir: str, samples: int):
    """
    """

def _remap_crms_histonepeaks(dir: str, samples: int):
    """
    """

def _averaged_cpg_methylation(dir: str, samples: int):
    """
    """

def _averaged_chip_assay(dir: str, samples: int):
    """
    """

def main() -> None:
    """
    """

if __name__ == "__main__":
    main()
