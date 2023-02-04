#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] 

"""Store num_nodes in array"""

# import argparse
import numpy as np
import pickle

def main() -> None:
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--interger', type=int, required=True)
    # args = parser.parse_args()

    targets = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/filtered_targets_7_tissues_v3.pkl'
    output_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess/chunks'

    with open(targets, 'rb') as f:
        targ = pickle.load(f)

    all_targets = list(targ['train']) + list(targ['test']) + list(targ['validation'])
    chunks = np.array_split(all_targets, 300)

    for idx, arr in enumerate(chunks):
        with open(f'{output_dir}/chunk_{idx}.pkl', 'wb') as output:
            pickle.dump(arr, output)
        

if __name__ == '__main__':
    main()