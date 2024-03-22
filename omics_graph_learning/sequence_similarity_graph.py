#! /usr/bin_element/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
# Plan
# fxn to match sequences in each bin_element of elements
# fxn to output edges of graph

"""Create a secondary graph structure based off of sequence similarity of
regulatory elements. Graphs will be used to test training, but also as a
potential auxilliary task."""

import argparse
import csv
import datetime
from itertools import combinations
import multiprocessing
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from Bio import Align
import pandas as pd
import pybedtools
from pybedtools import BedTool  # type: ignore


def log_progress(message: str) -> None:
    """Print a log message with timestamp to stdout"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{timestamp}] {message}")


def _chr_lengths_ucsc(chrom_sizes_file: str) -> dict[str, int]:
    """Get chromosome lengths for reference genome from ucsc txt file. Assumes
    text file is tab-delimited with chromosome name in first column and length
    in second column. Addtionally, ignores sex chromosomes."""
    with open(chrom_sizes_file, "r") as file:
        chrom_lengths = {
            line[0]: int(line[1])
            for line in csv.reader(file, delimiter="\t")
            if line[0] not in ["chrX", "chrY"]
        }
    return chrom_lengths


def _get_regulatory_element_size_metrics(elements: pybedtools.BedTool) -> float:
    """Get the mean and standard deviation of the regulatory element sizes."""
    element_sizes = [int(feature[2]) - int(feature[1]) for feature in elements]
    return int(statistics.stdev(element_sizes))


def _bin_genome(
    chrom_lengths: dict[str, int], bin_size: int, step_length: int
) -> pd.DataFrame:
    """Bin genome into bins according to bin_size and step_length, including the
    last bin_element that can be smaller than bin_size. Returns a dataframe."""
    bins = []
    for chrom, length in chrom_lengths.items():
        start = 0
        while start < length - bin_size:
            end = start + bin_size
            bins.append((chrom, start, end))
            start += step_length
        if start < length:
            bins.append((chrom, start, length))
    return pd.DataFrame(bins, columns=["chrom", "start", "end"])


def _bin_elements(
    reg_elements: pybedtools.BedTool,
    row: pd.Series,
) -> pybedtools.BedTool:
    """Get regulatory elements that are within the given bin_element."""
    return reg_elements.intersect(
        pybedtools.BedTool(  # type: ignore
            f"{row['chrom']}\t{row['start']}\t{row['end']}", from_string=True
        ),
        sorted=True,
    )


def _bin_from_df(
    bins: pd.DataFrame, reg_elements: pybedtools.BedTool
) -> Tuple[pd.DataFrame, List[List[pybedtools.BedTool]]]:
    """Add a column to the bins dataframe that contains the regulatory elements
    that are within the bin_element."""
    bins["reg_elements"] = bins.apply(
        lambda bin_element: _bin_elements(reg_elements, bin_element), axis=1
    )
    return bins, bins["reg_elements"].tolist()


def _initialize_pairwise_aligner(
    match_score: int = 2,
    mismatch_score: int = -1,
    open_gap_score: int = -10,
    extend_gap_score: int = -2,
) -> Align.PairwiseAligner:
    """Initialize a pairwise aligner with given match, mismatch, open gap, and
    extend gap scores."""
    aligner = Align.PairwiseAligner()
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    return aligner


def similar_enhancers(
    sequences: List[str],
    aligner: Align.PairwiseAligner,
    min_similarity: float = 0.8,
    max_length_diff: Optional[Union[int, float]] = None,
) -> Set[Tuple[str, str]]:
    """Find similar sequences based on pairwise sequence alignment and only
    keeps alignments passing both a threshold score and as long as the length
    difference between the sequences is within a given threshold. Keeps a cache
    to avoid redundant alignments."""
    # set up vars
    max_length_diff = max_length_diff if max_length_diff is not None else float("inf")
    pairs = set()
    alignment_cache: Dict[tuple[str, str], Any] = {}

    # compare pairwise sequences
    for seq1, seq2 in combinations(sequences, 2):
        if (seq1, seq2) in alignment_cache:
            alignment = alignment_cache[(seq1, seq2)]
        else:
            alignment = aligner.align(seq1, seq2)[0]
            alignment_cache[(seq1, seq2)] = alignment
        score = alignment.score / len(alignment)
        if score >= min_similarity and abs(len(seq1) - len(seq2)) <= max_length_diff:
            pairs.add((seq1, seq2))
    return pairs


def _map_elements_to_nucleotide_content(
    bin_element: pybedtools.BedTool,
    fasta: str,
) -> Dict[str, str]:
    """Map regulatory elements to nucleotide content."""
    return {
        feature[13]: f"{feature[0]}_{feature[1]}_{feature[3]}"
        for feature in bin_element.nucleotide_content(fi=fasta, seq=True)
    }


def _seq_to_elements(
    nuc_to_element: Dict[str, str], edges: Set[Tuple[str, str]]
) -> Set[Tuple[str, str]]:
    """Convert sequence pairs to regulatory element pairs."""
    return {
        (nuc_to_element[edge_0], nuc_to_element[edge_1]) for edge_0, edge_1 in edges
    }


def process_bin_element(
    bin_element: pd.Series,
    fasta: str,
    aligner: Align.PairwiseAligner,
    standard_deviation: int,
) -> Set[Tuple[str, str]]:
    """Process a bin_element of regulatory elements to find similar regulatory edges"""
    sequences = _map_elements_to_nucleotide_content(
        bin_element=bin_element, fasta=fasta
    )
    seqs = list(sequences.values())
    matched = similar_enhancers(
        seqs, aligner, min_similarity=0.8, max_length_diff=standard_deviation
    )
    return _seq_to_elements(nuc_to_element=sequences, edges=matched)


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chrom_sizes",
        type=str,
    )
    parser.add_argument(
        "--reg_elements",
        type=str,
    )
    parser.add_argument(
        "--fasta",
        type=str,
    )
    parser.add_argument(
        "--savedir",
        type=str,
    )
    args = parser.parse_args()
    aligner = _initialize_pairwise_aligner()

    # make genomic bins
    log_progress("Making genomic bins")
    chrom_lengths = _chr_lengths_ucsc(chrom_sizes_file=args.chrom_sizes)
    bins = _bin_genome(chrom_lengths=chrom_lengths, bin_size=250000, step_length=50000)
    bins.to_csv(f"{args.savedir}/bins.csv", index=False)  # temp

    # set up regulatory element catalogue
    log_progress("Binning regulatory elements")
    reg_elements = BedTool(args.reg_elements)
    standard_deviation = _get_regulatory_element_size_metrics(reg_elements)

    # bin regulatory elements
    with multiprocessing.Pool(processes=16) as pool:
        bins["reg_elements"] = pool.starmap(
            _bin_elements,
            [(reg_elements, row) for _, row in bins.iterrows()],
        )

    with multiprocessing.Pool(processes=16) as pool:
        bins["sequences"] = pool.starmap(
            _map_elements_to_nucleotide_content,
            [(bin_element, args.fasta) for bin_element in bins["reg_elements"]],
        )

    bin_element = bins["reg_elements"].tolist()
    bins.drop(columns=["reg_elements"], inplace=True)
    # bins, bin_element = _bin_from_df(bins=bins, reg_elements=reg_elements)
    bins.to_pickle(f"{args.savedir}/bins.pkl")  # temp
    log_progress("Saved binned elements elements. Doing pairwise alignments.")

    # link similar regulatory elements in parallel
    all_edges = set()
    with multiprocessing.Pool(processes=16) as pool:
        results = pool.starmap(
            process_bin_element,
            [
                (bin_element, args.fasta, aligner, standard_deviation)
                for bin_element in bin_element
            ],
        )
        for edges in results:
            all_edges.update(edges)

    # write edges to file
    with open(f"{args.save_dir}/sequence_similarity_graph_edges.txt", "w") as file:
        for edge in all_edges:
            file.write(f"{edge[0]}\t{edge[1]}\n")


if __name__ == "__main__":
    main()
