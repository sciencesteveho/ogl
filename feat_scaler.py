#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import joblib
import numpy as np
import pickle

from sklearn.preprocessing import RobustScaler

TISSUES = [
    'hippocampus',
    'liver',
    'lung',
    'mammary', 
    'pancreas',
    'skeletal_muscle',
    'left_ventricle',
    ]

WORKDIR = '/ocean/projects/bio210019p/stevesho/data/preprocess/check_num_nodes'

def all_feats(tissue, dir):
    with open(f'{dir}/graph_stats_{tissue}.pkl', 'rb') as f:
        stats = pickle.load(f)
    for key in stats.keys():
        feats = [x[0] for x in stats[key][2]]
    del stats
    return feats

if __name__ == "__main__":
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat', type=int, default=0,
                        help='col')
    args = parser.parse_args()

    # define scaler
    scaler = RobustScaler()

    # get all feats
    feats = []
    for tissue in TISSUES:
        feats += all_feats(tissue, WORKDIR)

    scaler.fit(np.array(feats).reshape(-1, 1))
    joblib.dump(scaler, f'feat{args.feat}_scaler.pt')

# torch.from_numpy(test)