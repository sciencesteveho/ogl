#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for graph processing"""

import functools
import inspect
import os
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
    

# def tpm_from_gct(gct: str, tissue: str) -> pd.DataFrame:
#     """_lorem ipsum"""
#     gtex_tpm = parse(gct, cid=[tissue])
#     return gtex_tpm.data_df