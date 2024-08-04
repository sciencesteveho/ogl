#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Class to handle creation of binned positional encodings to replace chromosome
start and end features."""


from typing import cast

import cooler  # type: ignore
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Object to return positional encoding embeddings, inherits from nn.Module.
    Some features are expected to span multiple bins, so we use a pooling
    strategy to combine the embeddings of all bins the feature falls within.
    Defaults to average pooling.

    Attributes:
        bins_df: The bins dataframe.
        embedding_dim: The size of the embedding.

    Methods
    --------
    get_bin_index(chromosome, start, end):
        Get the bin index for a given chromosome, start, and end position.
    forward(chr, node_start, node_end):
        Get the positional encoding for a given chromosome, start, and end
        position.

    Examples:
    --------
    >>> positional_encoding = PositionalEncoding(chromfile="chrom.sizes", binsize=50000)
    >>> positional_encoding("chr1", 100000, 150000)
    """

    def __init__(self, chromfile: str, binsize: int, embedding_dim: int = 5):
        """Instantiate the PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.bins_df = self.get_bins(chromsize_file=chromfile, binsize=binsize)
        self.embedding = nn.Embedding(len(self.bins_df), embedding_dim)

    def get_bin_indices(self, chromosome: str, start: int, end: int) -> np.ndarray:
        """Get all bin indices that overlap with a given chromosome, start, and end position."""
        chrom_bins = self.bins_df[self.bins_df["chrom"] == chromosome]

        # cast to numpy arrays to deal with typing
        start_values = cast(np.ndarray, chrom_bins["start"].values)
        end_values = cast(np.ndarray, chrom_bins["end"].values)

        start_bin_idx = np.searchsorted(start_values, start, side="right") - 1
        end_bin_idx = np.searchsorted(end_values, end, side="left")
        overlapping_bins = chrom_bins.iloc[start_bin_idx : end_bin_idx + 1]
        return np.array(overlapping_bins.index)

    def forward(
        self, chromosome: str, node_start: int, node_end: int, pooling: str = "average"
    ) -> np.ndarray:
        """Return the positional encoding tensor, with pooling if the feature
        spans multiple bins."""
        # get indices for all bins the feature spans
        bin_idxs = self.get_bin_indices(
            chromosome=chromosome, start=node_start, end=node_end
        )

        # single bin => no pooling
        if len(bin_idxs) == 1:
            return (
                self.embedding(torch.tensor(bin_idxs, dtype=torch.long))
                .detach()
                .numpy()
            )

        bin_embeddings = self.embedding(torch.tensor(bin_idxs, dtype=torch.long))

        if pooling == "max":
            pooled_embedding, _ = torch.max(bin_embeddings, dim=0)
        elif pooling == "average":
            pooled_embedding = torch.mean(bin_embeddings, dim=0)
        else:
            raise ValueError("Pooling type not supported. Choose 'max' or 'average'.")

        return pooled_embedding.detach().numpy()

    @staticmethod
    def get_bins(chromsize_file: str, binsize: int) -> pd.DataFrame:
        """Get the bins for a given chromosome size file and bin size.

        Args:
            chromsize_file: The path to the chromosome size file.
            binsize: The size of the bins.

        Returns:
            pd.DataFrame: The bins dataframe.
        """
        chromsizes_df = pd.read_csv(chromsize_file, sep="\t", names=["name", "length"])
        chromsizes = chromsizes_df.set_index("name")["length"]
        return cooler.binnify(chromsizes, binsize)
