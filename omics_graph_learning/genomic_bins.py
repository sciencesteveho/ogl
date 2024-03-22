#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
# Plan
# fxn to bin genome into 250kb bins, with a 50kb sliding window
# fxn to get reg elements within each bin
# fxn to get sequence for reg elements
# fxn to match sequences in each bin of elemetns
# fxn to output edges of graph

import sourmash
from sourmash import MinHash


# Function to create a MinHash signature of a given sequence.
def create_signature(seq, ksize=31, n=500):
    mh = MinHash(n=n, ksize=ksize)
    mh.add_sequence(seq)
    return mh


signatures = [sourmash.SourmashSignature(create_signature(seq)) for seq in sequences]

"""Construct initial base graph by linking regulatory elements based on sequence similarity."""

import os
import pickle


def place_holder_function() -> None:
    """_summary_ of function"""
    pass


def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()


# # Compare all signatures using Jaccard similarity and output the results.
# # This will be O(n^2), so for large lists, you might need a more efficient approach or parallelization.
# for i in range(len(signatures)):
#     for j in range(i + 1, len(signatures)):
#         similarity = signatures[i].jaccard(signatures[j])
#         if similarity > 0:  # or some threshold that you consider as 'similar'
#             print(f"Sequences {i} and {j} have a Jaccard similarity of {similarity}"
