#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Look into synder paper for mass spec values as a potential target

"""Get dataset train/val/test split"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import repeat
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from utils import time_decorator, TISSUE_TPM_KEYS


@time_decorator(print_args=True)
def std_dev_and_mean_gtex() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get means and standard deviation for TPM across all GTEx tissues"""
    

def main() -> None:
    """Pipeline to generate dataset split and target values"""
    gct = ''


if __name__ == '__main__':
    main()