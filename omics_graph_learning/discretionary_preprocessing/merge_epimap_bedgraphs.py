#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Script to handle conversion, liftover, and mergining of uniformly processed
epimap data."""


import argparse
from contextlib import suppress
from multiprocessing import Pool
import os
import subprocess
import time
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
    for dir in ["merged", "peaks", "tmp", "crms", "crms_processing"]:
        with suppress(FileExistsError):
            os.makedirs(os.path.join(directory, dir), exist_ok=True)


def _get_mark_files(path: str, mark: str) -> List[str]:
    """Get all files for a mark"""
    return [file for file in os.listdir(path) if mark in file]


def _bigwig_to_bedgraph(
    path: str,
    file: str,
    resource_dir: str,
) -> str:
    """Convert bigwig to bedgraph using subprocess"""
    bedgraph = f"{path}/tmp/{file}.bedgraph"
    try:
        with open(bedgraph, "w") as outfile:
            subprocess.run(
                [f"{resource_dir}/bigWigToBedGraph", f"{path}/{file}", bedgraph],
                stdout=outfile,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        print(f"Error converting {file} to bedgraph with error {error}")
    return bedgraph


def _sort_mark_file(file: str) -> str:
    """Sort a single file using subprocess"""
    sorted_file = f"{file}.sorted"
    try:
        with open(sorted_file, "w") as outfile:
            subprocess.run(
                [
                    "sort",
                    "--parallel=12",
                    "-S",
                    "60%",
                    "-k1,1",
                    "-k2,2n",
                    file,
                ],
                stdout=outfile,
                check=True,
            )
    except Exception as error:
        print(f"Error sorting {file}: {error}")
    return sorted_file


def _bigwig_to_bedgraph_sequential(path: str, files: List[str]) -> List[str]:
    resource_dir = "/ocean/projects/bio210019p/stevesho/resources"
    return [
        _bigwig_to_bedgraph(path=path, file=file, resource_dir=resource_dir)
        for file in files
    ]


def _sort_marks_sequential(path: str, files: List[str]) -> List[str]:
    return [_sort_mark_file(file=file) for file in files]


def _sum_coverage_and_average(path: str, mark: str, files: List[str]) -> str:
    """Subprocess call to sum the coverage and average the values"""
    merged_bedgraph = f"{path}/merged/{mark}.merged.bedgraph"
    merged_bedgraph_summed = f"{path}/merged/{mark}.merged.summed.bedgraph"
    if len(files) == 3:
        try:
            with open(merged_bedgraph, "w") as outfile:
                subprocess.run(
                    ["bedtools", "unionbedg", "-i"] + files,
                    stdout=outfile,
                    check=True,
                )
        except subprocess.CalledProcessError as error:
            print(f"Error running bedtools unionbedg for {mark} with error {error}")
    else:
        merged_bedgraph = files[0]
    try:
        with open(merged_bedgraph_summed, "w") as outfile:
            subprocess.run(
                ["awk", "{sum=$4+$5+$6; print $1, $2, $3, sum / 3}", merged_bedgraph],
                stdout=outfile,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        print(f"Error running awk for {mark} with error {error}")
    return merged_bedgraph_summed


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
        with suppress(FileExistsError):
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


def _print_with_timer(message: str, start_time: float) -> None:
    """Print message with elapsed time from the start_time."""
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"{message} | Time Elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    )


def process_mark(mark: str, path: str) -> None:
    """Process a single mark"""
    start_time = time.time()

    # get files for mark
    files = _get_mark_files(path=path, mark=mark)

    # convert bigwig to bedgraph
    bedgraphs = _bigwig_to_bedgraph_sequential(path=path, files=files)

    # sorted_files = _sort_marks_parallel(path=path, files=bedgraphs)
    sorted_files = _sort_marks_sequential(path=path, files=bedgraphs)
    # if not os.path.exists(sorted_files[0]):
    #     print(f"Error sorting {mark}: {sorted_files}")

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
    print(f"Peaks for {mark} called!")
    _print_with_timer(f"Completed processing for {mark}", start_time)


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
