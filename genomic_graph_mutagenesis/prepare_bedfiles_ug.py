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

        self.tissue = self.resources["tissue"]
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

        for file in self.tissue_specific.values():
            check_and_symlink(
                dst=f"{self.tissue_dir}/unprocessed/{file}",
                src=f"{self.data_dir}/{file}",
                boolean=True,
            )

        # for file in ["enhancers_e_e", "enhancers_e_g"]:
        #     check_and_symlink(
        #         dst=f"{self.tissue_dir}/interaction/{self.tissue_specific[file]}",
        #         src=f"{self.data_dir}/{self.tissue_specific[file]}",
        #         boolean=False,
        #     )

        interact_files = {
            "circuits": f"{self.dirs['circuit_dir']}/{self.interaction['circuits']}",
            "mirdip": f"{self.shared_data_dir}/interaction/mirdip_tissue/{self.interaction['mirdip']}",
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



def main() -> None:
    """Main function"""
    pass


if __name__ == "__main__":
    main()