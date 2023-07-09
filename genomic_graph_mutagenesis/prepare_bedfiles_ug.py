#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO


"""_summary_ of project"""

import os
import argparse
import requests
import subprocess
from typing import Dict

import pandas as pd
import pybedtools

from utils import dir_check_make, parse_yaml, time_decorator


class UniversalGenomeDataPreprocessor:
    """Data preprocessor for dealing with differences in bed files.

    Args:
        Params // Filenames and options parsed from initial yaml

    Methods
    ----------
    _run_cmd:
        runs shell command via subprocess call
    _make_directories:
        make required directories for processing
    _symlink_rawdata:
        symlink the raw data to a folder within the directory
    _download_shared_files:
        download files used by multiple tissues
    _add_tad_id:
        add IDs to tad file
    _tf_binding_clusters:
        clusters of tf binding sites
    _merge_cpg:
        merge book-ended cpg features
    prepare_data_files:
        main pipeline function
    """

    def __init__(self, params: Dict[str, Dict[str, str]]) -> None:
        """Initialize the class"""
        self.dirs = params["dirs"]
        self.interaction = params["interaction"]
        self.options = params["options"]
        self.resources = params["resources"]
        self.shared = params["local"]
        self.tissue_specific = params["tissue_specific"]

        self.tissue = 'universalgenome'
        self.root_dir = self.dirs["root_dir"]
        self.shared_data_dir = f"{self.root_dir}/shared_data"
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
        self.data_dir = f"{self.root_dir}/raw_files/{self.tissue}"

        # make directories, link files, and download shared files if necessary
        self._make_directories()
        self._symlink_rawdata()
        # self._download_shared_files()

    def _make_directories(self) -> None:
        """Make directories for processing"""
        dir_check_make(f"{self.root_dir}/shared_data")

        for directory in ["local", "interaction", "unprocessed", "histones"]:
            dir_check_make(f"{self.tissue_dir}/{directory}")

        for directory in ["local", "interaction"]:
            dir_check_make(f"{self.root_dir}/shared_data/{directory}")

    def _run_cmd(self, cmd: str) -> None:
        """Simple wrapper for subprocess as options across this script are
        constant"""
        subprocess.run(cmd, stdout=None, shell=True)

    def _symlink_rawdata(self) -> None:
        """Make symlinks for tissue specific files in unprocessed folder"""

        def check_and_symlink(dst, src, boolean=False):
            try:
                if boolean == True:
                    if (bool(file) and os.path.exists(src)) and (
                        not os.path.exists(dst)
                    ):
                        os.symlink(src, dst)
                else:
                    if not os.path.exists(dst):
                        os.symlink(src, dst)
            except FileExistsError:
                pass

        interact_files = {
            "mirnatargets": f"{self.shared_data_dir}/interaction/{self.interaction['mirnatargets']}",
            "ppis": f"{self.shared_data_dir}/interaction/{self.interaction['ppis']}",
            "tf_marker": f"{self.shared_data_dir}/interaction/{self.interaction['tf_marker']}",
        }

        for file in interact_files:
            check_and_symlink(
                dst=f"{self.tissue_dir}/interaction/" + self.interaction[file],
                src=interact_files[file],
                boolean=False,
            )

    @time_decorator(print_args=True)
    def _add_TAD_id(self, bed: str) -> None:
        """Add identification number to each TAD"""
        cmd = f"sed 's/ /\t/g' {self.data_dir}/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"tad_\"NR}}' \
            > {self.tissue_dir}/local/tads_{self.tissue}.txt"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _superenhancers(self, bed: str) -> None:
        """Simple parser to remove superenhancer bed unneeded info"""
        cmd = f" tail -n +2 {self.data_dir}/{bed} \
            | awk -vOFS='\t' '{{print $3, $4, $5, $2}}' \
            | sort -k1,1 -k2,2n \
            | bedtools merge -i - \
            | awk -vOFS='\t' '{{print $1, $2, $3, \"superenhancer\"}}' \
            > {self.tissue_dir}/local/superenhancers_{self.tissue}.bed"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _split_merge_fractional_methylation(self) -> None:
        """Splits fractional methylation into merged bedfiles for each sample.
        We keep CpGs if fractional methylation is greater than 0.70 and adjacent
        methylated CpGs are merged.
        """
        # dir_check_make(f"{self.dirs['methylation_dir']}/processing")

        # for chr in range(1, 23):
        #     df = pd.read_csv(
        #         f"{self.dirs['methylation_dir']}/chr{chr}.fm",
        #         header=None,
        #         delimiter="\t",
        #         index_col=0,
        #     )
        #     for column in df.columns:
        #         newdf = pd.DataFrame(columns=["chrom", "start", "end", "value"])
        #         newdf["end"] = list(df.index)
        #         newdf["start"] = list(df.index - 1)
        #         newdf["value"] = list(df[column])
        #         newdf["chrom"] = f"chr{chr}"

        #         pybedtools.BedTool.from_dataframe(newdf).filter(
        #             lambda x: float(x[3]) > 0.7
        #         ).merge().saveas(
        #             f"{self.dirs['methylation_dir']}/processing/chr{chr}_{column}.bed"
        #         )

        # for column in range(1, 38):
        #     cmd = f"cat {self.dirs['methylation_dir']}/processing/*_{column}.bed* \
        #         | sort -k1,1 -k2,2n \
        #         > {self.tissue_dir}/local/methylation_{column}.bed"
        #     self._run_cmd(cmd)
            
    @time_decorator(print_args=True)
    def prepare_data_files(self) -> None:
        """Pipeline to prepare all bedfiles"""

        ### Make symlinks for shared data files
        for file in self.shared.values():
            src = f"{self.root_dir}/shared_data/local/{file}"
            dst = f"{self.tissue_dir}/local/{file}"
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
            
        ### Make symlinks and rename files that do not need preprocessing
        for datatype in self.tissue_specific:
            if datatype not in ['super_enhancer', 'tads']:
                if self.tissue_specific[datatype]:
                    if datatype == 'tf_binding' or '_' not in datatype:
                        src = f"{self.data_dir}/{self.tissue_specific[datatype]}"
                    else:
                        src = f"{self.dirs['bigwig_dir']}/{self.tissue_specific[datatype]}"
                    dst = f"{self.tissue_dir}/local/{datatype}_{self.tissue}.bed"
                    try:
                        os.symlink(src, dst)
                    except FileExistsError:
                        pass
                
        self._add_TAD_id(self.tissue_specific["tads"])
        
        self._superenhancers(self.tissue_specific["super_enhancer"])
        
        self._split_merge_fractional_methylation()


def main() -> None:
    """Main function"""
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

    preprocessObject = UniversalGenomeDataPreprocessor(params)
    preprocessObject.prepare_data_files()

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
