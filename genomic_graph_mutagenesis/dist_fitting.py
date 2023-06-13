#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions


def main(tpm_pickle: str) -> None:
    """Main function"""
    with open(tpm_pickle, 'rb') as f:
        tpm = pickle.load(f)
    
    tpms = [np.log2(x + 0.25) for subset in tpm for x in subset]
    f = Fitter(tpms, timeout=360)
    f.fit()
    print(f.summary())
    print(f.get_best(method = 'sumsquare_error'))


if __name__ == "__main__":
    main(
        tpm_pickle='per_sample_median_tpm.pkl'
    )