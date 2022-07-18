#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for graph processing"""

import csv
import functools
import inspect
import os
import random
import time
import yaml

from datetime import timedelta
from typing import Any, Callable, Dict, Union


def bool_check_attributes(
    attribute: str,
    attribute_file: str
    ) -> bool:
    """Checks that attribute files exists before making directory for it"""
    if attribute in ['gc', 'microsatellites', 'phastcons', 'simplerepeats']:
        return True
    else:
        return bool(attribute_file)


def chunk_genes(
    gff: str,
    chunks: int,
    ) -> None:
    """Constructs graphs in parallel"""
    ### get list of all gencode V26 genes
    with open(gff, newline = '') as file:
        genes = [line[3] for line in csv.reader(file, delimiter='\t')]

    for num in range(0, 5):
        random.shuffle(genes)

    split_list = lambda l, chunks: [l[n:n+chunks] for n in range(0, len(l), chunks)]
    split_genes = split_list(genes, chunks)
    return {index:gene_list for index, gene_list in enumerate(split_genes)}


def dir_check_make(dir: str) -> None:
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


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
    

"""
### Code for saving chunked genes
dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data'
gff = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'
chunks = 1124
chunk_dict = chunk_genes(gff, chunks)
output = open(f'{dir}/gencode_chunks_{chunks}.pkl', 'wb)
try:
    pickle.dump(chunk_dict, output)
finally:
    output.close()
"""