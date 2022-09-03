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


def _run_cmd(self, cmd: str) -> None:
    """Simple wrapper for subprocess as options across this script are constant"""
    subprocess.run(cmd, stdout=None, shell=True)


def _split_full_stack_chromhmm(bed: str):
    """
    """
    segmentations = ['HET1', 'HET2', 'HET3', 'HET4', 'HET5', 'HET6', 'HET7', 'HET8', 'HET9', 'ReprPC1', 'ReprPC2', 'ReprPC3', 'ReprPC4', 'ReprPC5', 'ReprPC6', 'ReprPC7', 'ReprPC8', 'ReprPC9', 'Acet1', 'Acet2', 'Acet3', 'Acet4', 'Acet5', 'Acet6', 'Acet7', 'Acet8', 'EnhWk1', 'EnhWk2', 'EnhWk3', 'EnhWk4', 'EnhWk5', 'EnhWk6', 'EnhWk7', 'EnhWk8', 'EnhA1', 'EnhA2', 'EnhA3', 'EnhA4', 'EnhA5', 'EnhA6', 'EnhA7', 'EnhA8', 'EnhA9', 'EnhA10', 'EnhA11', 'EnhA12', 'EnhA13', 'EnhA14', 'EnhA15', 'EnhA16', 'EnhA17', 'EnhA18', 'EnhA19', 'EnhA20', 'TxEnh1', 'TxEnh2', 'TxEnh3', 'TxEnh4', 'TxEnh5', 'TxEnh6', 'TxEnh7', 'TxEnh8', 'TxWk1', 'TxWk2', 'Tx1', 'Tx2', 'Tx3', 'Tx4', 'Tx5', 'Tx6', 'Tx7', 'Tx8', 'TxEx1', 'TxEx2', 'TxEx3', 'TxEx4', 'znf1', 'znf2', 'BivProm1', 'BivProm2', 'BivProm3', 'BivProm4', 'PromF1', 'PromF2', 'PromF3', 'PromF4', 'PromF5', 'PromF6', 'PromF7', 'TSS1', 'TSS2']

    for segmentation in segmentations:
        cmd = f"grep {segmentation} {self.root_tissue}/unprocessed/{bed} \
            > {self.root_tissue}/local/{segmentation.casefold()}.bed"
        _run_cmd(cmd)
    
def _concensus_tads(dir: str):
    """
    There were single malformed BED lines in KMB7, NHEK, and T470. These lines were removed.
    Lines removed: 1431, 2270, 1291, in referenced order.
    sed -i -e '1431d' KBM7_Rao_2014-raw_TADs.txt
    sed -i -e '2270d' NHEK_Rao_2014-raw_TADs.txt
    sed -i -e '1291d' T470_raw-merged_TADs.txt
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
