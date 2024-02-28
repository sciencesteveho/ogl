#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import os
import pickle

import utils


def place_holder_function():
    """_summary_ of function"""
    pass


def main() -> None:
    """Main function"""
    # get reference files
    graph, indexes, split, targets = utils._open_graph(
        g_path="regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_scaled.pkl",
        indexes="regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_idxs.pkl",
        split="training_targets_split.pkl",
        targets="targets.pkl",
    )

    # get indexes and not present idxs


if __name__ == "__main__":
    main()
