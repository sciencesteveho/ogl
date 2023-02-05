#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for graph processing"""

import csv
from datetime import timedelta
import functools
import inspect
import os
import random
import time
from typing import Any, Callable, Dict, List, Tuple, Union
import yaml

import pybedtools


TISSUE_TPM_KEYS = {
    "Adipose - Subcutaneous": 3,
    "Adipose - Visceral (Omentum)": 4,
    "Adrenal Gland": 5,
    "Artery - Aorta": 6,
    "Artery - Coronary": 7,
    "Artery - Tibial": 8,
    "Bladder": 9,
    "Brain - Amygdala": 10,
    "Brain - Anterior cingulate cortex (BA24)": 11,
    "Brain - Caudate (basal ganglia)": 12,
    "Brain - Cerebellar Hemisphere": 13,
    "Brain - Cerebellum": 14,
    "Brain - Cortex": 15,
    "Brain - Frontal Cortex (BA9)": 16,
    "Brain - Hippocampus": 17,
    "Brain - Hypothalamus": 18,
    "Brain - Nucleus accumbens (basal ganglia)": 19,
    "Brain - Putamen (basal ganglia)": 20,
    "Brain - Spinal cord (cervical c-1)": 21,
    "Brain - Substantia nigra": 22,
    "Breast - Mammary Tissue": 23,
    "Cells - Cultured fibroblasts": 24,
    "Cells - EBV-transformed lymphocytes": 25,
    "Cervix - Ectocervix": 26,
    "Cervix - Endocervix": 27,
    "Colon - Sigmoid": 28,
    "Colon - Transverse": 29,
    "Esophagus - Gastroesophageal Junction": 30,
    "Esophagus - Mucosa": 31,
    "Esophagus - Muscularis": 32,
    "Fallopian Tube": 33,
    "Heart - Atrial Appendage": 34,
    "Heart - Left Ventricle": 35,
    "Kidney - Cortex": 36,
    "Kidney - Medulla": 37,
    "Liver": 38,
    "Lung": 39,
    "Minor Salivary Gland": 40,
    "Muscle - Skeletal": 41,
    "Nerve - Tibial": 42,
    "Ovary": 43,
    "Pancreas": 44,
    "Pituitary": 45,
    "Prostate": 46,
    "Skin - Not Sun Exposed (Suprapubic)": 47,
    "Skin - Sun Exposed (Lower leg)": 48,
    "Small Intestine - Terminal Ileum": 49,
    "Spleen": 50,
    "Stomach": 51,
    "Testis": 52,
    "Thyroid": 53,
    "Uterus": 54,
    "Vagina": 55,
    "Whole Blood": 56,
    }


def bool_check_attributes(
    attribute: str,
    attribute_file: str
    ) -> bool:
    """Checks that attribute files exists before making directory for it"""
    if attribute in [
        'ctcf',
        'dnase',
        'H3K27ac',
        'H3K27me3',
        'H3K36me3',
        'H3K4me1',
        'H3K4me3',
        'H3K9me3',
        'polr2a',
        ]:
        return bool(attribute_file)
    else:
        return True


def chunk_genes(
    genes: List[str],
    chunks: int,
    ) -> Dict[int, List[str]]:
    """Constructs graphs in parallel"""
    ### get list of all gencode V26 genes
    for num in range(0, 5):
        random.shuffle(genes)

    split_list = lambda l, chunks: [l[n:n+chunks] for n in range(0, len(l), chunks)]
    split_genes = split_list(genes, chunks)
    return {index:gene_list for index, gene_list in enumerate(split_genes)}


def dir_check_make(dir: str) -> None:
    """Utility to make directories only if they do not already exist"""
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass
    

def _ls(dir: str) -> List[str]:
    """
    Returns a list of files within the directory
    """
    return [
        file for file in os.listdir(dir)
        if os.path.isfile(f'{dir}/{file}')
        ]


def filtered_genes_from_bed(tpm_filtered_genes: str) -> List[str]:
    with open(tpm_filtered_genes, newline='') as file:
        return [line[3] for line in csv.reader(file, delimiter='\t')]


def gene_list_from_graphs(root_dir: str, tissue: str) -> List[str]:
    """Returns a list of genes with constructed graphs, avoiding genes that may
    not have edges in smaller window"""
    directory = f'{root_dir}/{tissue}/parsing/graphs'
    return [
        gene.split("_")[0] for gene
        in os.listdir(directory)
    ]


def genes_from_gff(gff: str) -> List[str]:
    """Get list of gtex genes from GFF file"""
    with open(gff, newline = '') as file:
        return {
            line[3]: line[0] for line in csv.reader(file, delimiter='\t')
            if line[0] not in ['chrX', 'chrY', 'chrM']
            }


def parse_yaml(config_file: str) -> Dict[str, Union[str, list]]:
    """Load yaml for parsing"""
    with open(config_file, 'r') as stream:
        params = yaml.safe_load(stream)
    return params
        

def time_decorator(print_args: bool = False, display_arg: str ="") -> Callable:
    def _time_decorator_func(function: Callable) -> Callable:
        @functools.wraps(function)
        def _execute(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            fxn_args = inspect.signature(function).bind(*args, **kwargs).arguments
            try:
                result = function(*args, **kwargs)
                return result
            except Exception as error:
                result = str(error)
                raise
            finally:
                end_time = time.monotonic()
                if print_args == True:
                    print(f'Finished {function.__name__} {[val for val in fxn_args.values()]} - Time: {timedelta(seconds=end_time - start_time)}')
                else:
                    print(f'Finished {function.__name__} {display_arg} - Time: {timedelta(seconds=end_time - start_time)}')
        return _execute
    return _time_decorator_func


@time_decorator(print_args=True)
def _filter_low_tpm(
    tissue: str,
    file: str,
    return_list: False,
    ) -> List[str]:
    """Remove genes expressing less than 0.10 TPM across 20% of samples"""
    df = pd.read_table(file, index_col=0, header=[2])
    sample_n = len(df.columns)
    df['total'] = df.select_dtypes(np.number).gt(0.10).sum(axis=1)
    df['result'] = df['total'] >= (.30 * sample_n)
    if return_list == False:
        return [
            f'{gene}_{tissue}' for gene
            in list(df.loc[df['result'] == True].index)
        ]
    else:
        return list(df.loc[df['result'] == True].index)


@time_decorator(print_args=True)
def _filtered_gene_windows(
    gencode: str,
    chromfile: str,
    slop: bool,
    tissue: str,
    tpm_file: str,
    window: int,
    ) -> Tuple[pybedtools.BedTool, List[str]]:
    """
    Filter out genes in a GTEx tissue with less than 0.1 tpm across 20% of
    samples in that tissue. Additionally, we exclude analysis of sex
    chromosomes. 

    Returns:
        pybedtools object with +/- <window> windows around that gene
    """
    tpm_filtered_genes = _filter_low_tpm(
        tissue,
        tpm_file,
        return_list=True,
    )
    genes = pybedtools.BedTool(gencode)
    genes_filtered = genes.filter(
        lambda x: x[3] in tpm_filtered_genes and x[0] not in ['chrX', 'chrY', 'chrM']
        )
    
    if slop:
        return genes_filtered.slop(g=chromfile, b=window)\
            .cut([0, 1, 2, 3])\
            .sort(), [x[3] for x in genes_filtered]
    else:
        return genes_filtered.sort(), [x[3] for x in genes_filtered]