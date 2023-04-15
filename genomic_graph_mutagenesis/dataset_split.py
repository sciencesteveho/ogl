#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 
#

"""Get train / test / val splits for nodes in graphs and generate targets for
training the network.
"""

import argparse
import csv
from itertools import repeat
from multiprocessing import Pool
import pickle
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pybedtools

from utils import dir_check_make, NODES, parse_yaml, time_decorator