#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Script to handle conversion, liftover, and mergining of uniformly processed
epimap data."""


import argparse
from contextlib import suppress
from multiprocessing import Pool
import os
import subprocess
from typing import List

# Cutoffs for each mark
CUTOFFS = {
    "DNase-seq": 1.9,
    "H3K27ac": 2.2,
    "H3K27me3": 1.2,
    "H3K36me3": 1.7,
    "H3K4me1": 1.7,
    "H3K4me2": 2.0,
    "H3K4me3": 1.7,
    "H3K79me2": 2.2,
    "H3K9ac": 1.6,
    "H3K9me3": 1.1,
    "ATAC-seq": 2.0,
    "CTCF": 2.0,
    "POLR2A": 2.0,
    "RAD21": 2.0,
    "SMC3": 2.0,
}

REMAP_CRM_CALLER = (
    "/ocean/projects/bio210019p/stevesho/resources/remap2022/peakMerge.py"
)


def _make_directories(directory: str) -> None:
    """Make directories to store sorted files, merged files, called peaks, and
    lifted peaks"""
    for dir in ["sorted", "merged", "peaks", "tmp", "crms", "crms_processing"]:
        with suppress(FileExistsError):
            os.makedirs(os.path.join(directory, dir), exist_ok=True)


def _get_mark_files(path: str, mark: str) -> List[str]:
    """Get all files for a mark"""
    return [file for file in os.listdir(path) if mark in file]


def _sort_mark_file(path: str, file: str) -> str:
    """Sort a single file using subprocess"""
    sorted_file = f"{path}/tmp/{file}.sorted"
    try:
        with open(sorted_file, "w") as outfile:
            subprocess.run(
                [
                    "sort",
                    "--parallel=12",
                    "-S",
                    "80%",
                    "-k1,1",
                    "-k2,2n",
                    f"{path}/{file}",
                ],
                env={"LC_ALL": "C"},  # Set the environment variable for LC_ALL
                stdout=outfile,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        print(f"Error sorting {file} with error {error}")
    return sorted_file


def _sort_marks_parallel(path: str, files: List[str]) -> List[str]:
    """Sort multiple files in parallel using multiprocessing pool"""
    pool = Pool(processes=3)
    sorted_files = pool.starmap(_sort_mark_file, [(path, file) for file in files])
    pool.close()
    pool.join()
    return sorted_files


def _sum_coverage_and_average(path: str, mark: str, files: List[str]) -> str:
    """Subprocess call to sum the coverage and average the values"""
    merged_bedgraph = f"{path}/merged/{mark}.merged.bedgraph"
    try:
        unionbed = subprocess.run(
            ["bedtools", "unionbedg", "-i"] + files, stdout=subprocess.PIPE, check=True
        )
        try:
            with open(merged_bedgraph, "w") as outfile:
                subprocess.run(
                    ["awk", "{sum=$4+$5+$6; print $1, $2, $3, sum / 3}"],
                    input=unionbed.stdout,
                    stdout=outfile,
                    check=True,
                )
        except subprocess.CalledProcessError as error:
            print(f"Error running awk for {mark} with error {error}")
    except subprocess.CalledProcessError as error:
        print(f"Error running bedtools unionbedg for {mark} with error {error}")

    return merged_bedgraph


def _call_peaks(path: str, mark: str, bedgraph: str) -> str:
    """Subprocess call to call peaks"""
    peaks = f"{path}/tmp/{mark}_merged.narrow.peaks.bed"
    try:
        subprocess.run(
            [
                "macs2",
                "bdgpeakcall",
                "-i",
                bedgraph,
                "-o",
                peaks,
                "-c",
                str(CUTOFFS[mark]),
                "-l",
                "73",
                "-g",
                "100",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"Error calling peaks for {mark} with error {error}")
    return peaks


def _remove_first_two_lines(bedfile: str) -> None:
    """Removes the first two lines of the called peaks, because the header
    conflicts with the LiftOver tool."""
    with open(bedfile, "r") as file:
        lines = file.readlines()
    with open(bedfile, "w") as file:
        file.writelines(lines[2:])


def _lift_over_peaks(path: str, mark: str, peaks: str) -> None:
    resource_dir = "/ocean/projects/bio210019p/stevesho/resources"
    try:
        subprocess.run(
            [
                f"{resource_dir}/liftOver",
                peaks,
                f"{resource_dir}/hg19ToHg38.over.chain",
                f"{path}/peaks/{mark}_merged.narrow.peaks.hg38.bed",
                f"{path}/tmp/{mark}_merged.narrow.peaks.hg38.unmapped.bed",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"Error lifting over peaks for {mark} with error {error}")


def call_crms(working_dir: str, peakmerge_script: str, bedfile_dir: str) -> None:
    """Steps
    1. Symlink files
    2. Call!
    """
    for file in os.listdir(bedfile_dir):
        os.symlink(
            src=os.path.join(bedfile_dir, file),
            dst=os.path.join(f"{working_dir}/crms_processing", file),
        )
    try:
        subprocess.run(
            [
                "python",
                peakmerge_script,
                "/ocean/projects/bio210019p/stevesho/resources/hg38.chrom.sizes",
                f"{working_dir}/crms_processing",
                "narrowPeak",
                f"{working_dir}/crms",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"Error calling CRMs with error {error}")


def process_mark(mark: str, path: str) -> None:
    """Process a single mark"""
    # get files for mark
    files = _get_mark_files(path=path, mark=mark)

    # sort in parallel
    sorted_files = _sort_marks_parallel(path=path, files=files)

    # combine bedgraphs, average, and call peaks
    merged_bedgraph = _sum_coverage_and_average(
        path=path, mark=mark, files=sorted_files
    )

    # call peaks with MACS2
    peaks = _call_peaks(path=path, mark=mark, bedgraph=merged_bedgraph)

    # remove header for liftover
    _remove_first_two_lines(bedfile=peaks)

    # liftover to hg38
    _lift_over_peaks(path=path, mark=mark, peaks=peaks)


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(description="Merge epimap bedgraphs")
    parser.add_argument("path", type=str, help="Path to bedgraph files")
    args = parser.parse_args()

    # Make directories
    _make_directories(directory=args.path)

    # Process each mark
    for mark in CUTOFFS.keys():
        process_mark(mark=mark, path=args.path)

    # Call CRMS after processing all marks
    call_crms(
        working_dir=args.path,
        peakmerge_script=REMAP_CRM_CALLER,
        bedfile_dir=os.path.join(args.path, "peaks"),
    )


if __name__ == "__main__":
    main()
