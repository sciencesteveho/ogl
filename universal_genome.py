#! /usr/bin/env python
# -*- coding: utf-8 -*-  
#
# // TO-DO //
# - [ ]
#

"""Code to preprocess bedfiles into graph structures for the universal_genome graphs. 
"""

# import os
# import argparse
# import requests
# import subprocess

# import pybedtools

# from typing import Dict

# from pybedtools.featurefuncs import extend_fields

# from utils import dir_check_make, parse_yaml, time_decorator


def concensus_overlap_by_samples(dir: str, samples: int):
    '''
    '''
    

class UniversalGenomePreprocessor:
    """Data preprocessor for dealing with differences in bed files.

    Args:
        Params // Filenames and options parsed from initial yaml

    Methods
    ----------
    _run_cmd:
        runs shell command via subprocess call

    # Helpers
        HISTONES -- list of histone features
    """
    # HISTONES = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3',]

    def __init__(self, params: Dict[str, Dict[str, str]]) -> None:
        """Initialize the class"""
        self.resources = params['resources']
        self.dirs = params['dirs']
        self.tissue_specific = params['tissue_specific']
        self.shared = params['shared']
        self.options = params['options']

        self.tissue = self.resources['tissue']
        self.data_dir = self.dirs['data_dir']
        self.root_dir = self.dirs['root_dir']
        self.root_tissue = f"{self.root_dir}/{self.tissue}"

        # make directories, link files, and download shared files if necessary
        self._make_directories()
        self._symlink_rawdata()
        # self._download_shared_files()

    def _make_directories(self) -> None:
        """Make directories for processing"""
        dir_check_make(f'{self.root_dir}/shared_data')

        for directory in ['local', 'interaction', 'unprocessed', 'histones']:
            dir_check_make(f'{self.root_tissue}/{directory}')

        for directory in ['local', 'interaction']:
            dir_check_make(f'{self.root_dir}/shared_data/{directory}')

    def _run_cmd(self, cmd: str) -> None:
        """Simple wrapper for subprocess as options across this script are constant"""
        subprocess.run(cmd, stdout=None, shell=True)

    def _symlink_rawdata(self) -> None:
        """Make symlinks for tissue specific files in unprocessed folder"""
        for file in self.tissue_specific.values():
            try:
                if (bool(file) and os.path.exists(f'{self.data_dir}/{file}')) and (not os.path.exists(f'{self.root_tissue}/unprocessed/{file}')):
                    src = f'{self.data_dir}/{file}'
                    dst = f'{self.root_tissue}/unprocessed/{file}'
                    os.symlink(src, dst)
            except FileExistsError:
                pass

    def _download_shared_files(self) -> None:
        """Download shared local features if not already present"""
        def download(url, filename):
            with open(filename, "wb") as file:
                response = requests.get(url)
                file.write(response.content)

        if os.listdir(f'{self.root_dir}/shared_data/local_feats') == self.shared:
            pass
        else:
            for file in self.shared.values():
                download(f'https://raw.github.com/sciencesteveho/genome_graph_perturbation/raw/master/shared_files/local_feats/{file}', f'{self.root_dir}/shared_data/local_feats/{file}')

    @time_decorator(print_args=True)
    def _add_TAD_id(self, bed: str) -> None:
        """Add identification number to each TAD"""
        cmd = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"tad_\"NR}}' {self.root_tissue}/unprocessed/{bed} \
            > {self.root_tissue}/local/tads_{self.tissue}.txt"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def prepare_data_files(self) -> None:
        """Pipeline to prepare all bedfiles"""

        ### Make symlinks for shared data files
        for file in self.shared.values():
                src = f'{self.root_dir}/shared_data/local/{file}'
                dst = f'{self.root_tissue}/local/{file}'
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass

        ### Make symlinks and rename files that do not need preprocessing
        nochange = ['dnase', 'ctcf', 'polr2a', 'chromhmm']
        for datatype in nochange:
            if self.tissue_specific[datatype]:
                src = f'{self.data_dir}/{self.tissue_specific[datatype]}'
                dst = f'{self.root_tissue}/local/{datatype}_{self.tissue}.bed'
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
        
        self._add_TAD_id(self.tissue_specific['tads'])

        self._split_chromatinloops(self.tissue_specific['chromatinloops'])

        self._fenrir_enhancers(
            self.tissue_specific['enhancers_e_e'],
            self.tissue_specific['enhancers_e_g'],
            )

        self._tf_binding_clusters(self.tissue_specific['tf_binding'])

        self._merge_cpg(self.tissue_specific['cpg'])

        self._combine_histones()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to .yaml file with filenames",
    )

    args = parser.parse_args()
    params = parse_yaml(args.config)

    preprocessObject = GenomeDataPreprocessor(params)
    preprocessObject.prepare_data_files()

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
