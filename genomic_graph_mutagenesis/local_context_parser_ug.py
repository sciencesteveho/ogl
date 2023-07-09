#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

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
from utils import ATTRIBUTES
from utils import dir_check_make
from utils import genes_from_gencode
from utils import NODES
from utils import parse_yaml
from utils import time_decorator


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
        FEAT_WINDOWS -- dictionary of each nodetype: overlap windows
        NODES -- list of nodetypes
        ONEHOT_NODETYPE -- dictionary of node type one-hot vectors
    """

    DIRECT = ["chromatinloops", "tads"]
    NODE_FEATS = ["start", "end", "size", "gc"] + ATTRIBUTES

    # var helpers - for CPU cores
    NODE_CORES = len(NODES) + 1  # 12
    ATTRIBUTE_CORES = len(ATTRIBUTES)  # 30


def place_holder_function():
    """_summary_ of function"""
    pass


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()
