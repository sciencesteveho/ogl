#! /usr/bin/env python
# -*- coding: utf-8 -*-
#


import os

import cooler
import cooltools
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
import pandas as pd
import seaborn as sns

"""_summary_ of project"""

import os
import pickle

bp_formatter = EngFormatter("b")


def format_ticks(ax, x=True, y=True, rotate=True):
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis="x", rotation=45)


def main() -> None:
    """Main function"""
    cooler_name = "pancreas_40000_4dn.cool"
    clr = cooler.Cooler(cooler_name)

    f, axs = plt.subplots(figsize=(14, 4), ncols=3)
    ax = axs[0]
    im = ax.matshow(
        clr.matrix(balance=False).fetch("chr1"),
        vmax=2500,
        extent=(0, clr.chromsizes["chr1"], clr.chromsizes["chr1"], 0),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="raw counts")
    ax.set_title("chr1", y=1.08)
    ax.set_ylabel("position, Mb")
    format_ticks(ax)
    plt.tight_layout()


if __name__ == "__main__":
    main()
