#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] fix _download_shared_files, which is currently broken
#   - [ ] resolve hosting due to bandwidth limits
#   - [ ] fix URLs as they point to incorrect locations

"""Code to preprocess bedfiles before parsing into graph structures"""


import contextlib
import os
import subprocess
from typing import Dict, List

import requests

import utils

NODETYPES_LOCAL = ["cpgislands", "ctcfccre", "tss"]


class GenomeDataPreprocessor:
    """Data preprocessor for dealing with differences in bed files.

    Attributes:
        experiment_name (str): The name of the experiment.
        interaction_types (List[str]): The types of interactions.
        nodes (List[str]): The nodes.
        working_directory (str): The working directory.
        params (Dict[str, Dict[str, str]]): The parameters.

    Methods
    ----------
    _run_cmd:
        Runs shell command via subprocess call.
    _make_directories:
        Make required directories for processing.
    _symlink_rawdata:
        Symlink the raw data to a folder within the directory.
    _download_shared_files:
        Download files used by multiple tissues.
    _add_TAD_id:
        Add identification number to each TAD.
    _superenhancers:
        Simple parser to remove superenhancer bed unneeded info.
    _tf_binding_sites:
        Parse tissue-specific transcription factor binding sites.
    _merge_cpg:
        Merge individual CPGs with optional liftover.
    prepare_data_files:
        Main pipeline function.
    """

    def __init__(
        self,
        experiment_name: str,
        interaction_types: List[str],
        nodes: List[str],
        working_directory: str,
        params: Dict[str, Dict[str, str]],
    ) -> None:
        """Initialize the class"""
        self.experiment_name = experiment_name
        self.interaction_types = interaction_types
        self.nodes = nodes
        self.working_directory = working_directory

        self.dirs = params["dirs"]
        self.interaction = params["interaction"]
        self.methylation = params["methylation"]
        self.resources = params["resources"]
        self.shared = params["local"]
        self.features = params["features"]
        self.tissue_specific_nodes = params["tissue_specific_nodes"]

        self.tissue = self.resources["tissue"]
        self.root_dir = self.dirs["root_dir"]
        self.shared_data_dir = f"{self.root_dir}/shared_data"
        self.tissue_dir = (
            f"{self.working_directory}/{self.experiment_name}/{self.tissue}"
        )
        self.data_dir = f"{self.root_dir}/raw_files/{self.tissue}"

        # make directories, link files, and download shared files if necessary
        self._make_directories()
        self._symlink_rawdata()
        # self._download_shared_files()

    def _make_directories(self) -> None:
        """Make directories for processing"""
        utils.dir_check_make(self.tissue_dir)

        for directory in ["local", "interaction", "unprocessed"]:
            utils.dir_check_make(f"{self.tissue_dir}/{directory}")

    def _run_cmd(self, cmd: str) -> None:
        """Simple wrapper for subprocess as options across this script are
        constant"""
        subprocess.run(cmd, stdout=None, shell=True)

    def _symlink_rawdata(self) -> None:
        """Make symlinks for tissue specific files in unprocessed folder"""
        for file in self.tissue_specific_nodes.values():
            utils.check_and_symlink(
                dst=f"{self.tissue_dir}/unprocessed/{file}",
                src=f"{self.data_dir}/{file}",
                boolean=True,
            )

        interact_files = {
            "circuits": f"{self.dirs['circuit_dir']}/{self.interaction['circuits']}",
            "ppis": f"{self.shared_data_dir}/interaction/{self.interaction['ppis']}",
            "tf_marker": f"{self.shared_data_dir}/interaction/{self.interaction['tf_marker']}",
            "tf_binding": f"{self.shared_data_dir}/interaction/{self.interaction['tf_binding']}",
        }

        with contextlib.suppress(TypeError):
            for datatype in self.interaction_types:
                if datatype == "mirna":
                    utils.check_and_symlink(
                        src=f"{self.shared_data_dir}/interaction/mirdip_tissue/{self.interaction['mirdip']}",
                        dst=f"{self.tissue_dir}/interaction/"
                        + self.interaction["mirdip"],
                        boolean=True,
                    )
                    utils.check_and_symlink(
                        src=f"{self.shared_data_dir}/interaction/{self.interaction['mirnatargets']}",
                        dst=f"{self.tissue_dir}/interaction/"
                        + self.interaction["mirnatargets"],
                        boolean=True,
                    )
                else:
                    utils.check_and_symlink(
                        src=interact_files[datatype],
                        dst=f"{self.tissue_dir}/interaction/{self.interaction[datatype]}",
                        boolean=False,
                    )

    def _download_shared_files(self) -> None:
        """Download shared local features if not already present"""

        def download(url, filename):
            with open(filename, "wb") as file:
                response = requests.get(url)
                file.write(response.content)

        if os.listdir(f"{self.root_dir}/local_feats") != self.shared:
            for file in self.shared.values():
                download(
                    f"https://raw.github.com/sciencesteveho/genome_graph_perturbation/raw/master/shared_files/local_feats/{file}",
                    f"{self.root_dir}/shared_data/local_feats/{file}",
                )

    @utils.time_decorator(print_args=True)
    def _add_TAD_id(self, bed: str) -> None:
        """Add identification number to each TAD"""
        cmd = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"tad_\"NR}}' \
            {self.tissue_dir}/unprocessed/{bed} \
            > {self.tissue_dir}/local/tads_{self.tissue}.txt"

        self._run_cmd(cmd)

    @utils.time_decorator(print_args=True)
    def _superenhancers(self, bed: str) -> None:
        """Simple parser to remove superenhancer bed unneeded info"""
        cmd = f" tail -n +2 {self.tissue_dir}/unprocessed/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"superenhancer\"}}' \
            > {self.tissue_dir}/local/superenhancers_{self.tissue}.bed"

        self._run_cmd(cmd)

    @utils.time_decorator(print_args=True)
    def _tf_binding_sites(self, bed: str) -> None:
        """
        Parse tissue-specific transcription factor binding sites Vierstra et
        al., Nature, 2020, or from Funk et al., Cell Reports, 2020. Funk et al.,
        footprints are 20-seed HINT TFs with score > 200 and use the locations
        of the motifs, not the footprints, as HINT footprints are motif
        agnostic. Motifs are merged to form tf-binding clusters (within 46bp,
        from Chen et al., Scientific Reports, 2015). Vierstra et al., footprints
        are FPR thresholded with p value < 0.001. Footprints within 46bp are
        merged to form clusters, and then intersected with collapsed motifs. If
        90% of the collapsed motif falls within the cluster, the cluster is
        annotated with the binding motif.

        ** Removed clustering
        """
        # if study == "Funk":
        #     cmd = f"awk -v FS='\t' -v OFS='\t' '$5 >= 200' {self.tissue_dir}/unprocessed/{bed} \
        #         | cut -f1,2,3,4 \
        #         | sed 's/-/\t/g' \
        #         | cut -f1,2,3,6 \
        #         | sed 's/\..*$//g' \
        #         | sort -k1,1 -k2,2n \
        #         | bedtools merge -i - -d 46 -c 4 -o distinct \
        #         > {self.tissue_dir}/local/tfbindingclusters_{self.tissue}.bed"
        # else:
        # cmd = f"cut -f1,2,3 {self.tissue_dir}/unprocessed/{bed} \
        #     | bedtools merge -i - -d 46 \
        #     | bedtools intersect -wa -wb -a - -b {self.resources['tf_motifs']} -F 0.9 \
        #     | bedtools groupby -i - -g 1,2,3 -c 7 -o distinct \
        #     > {self.tissue_dir}/local/tfbindingclusters_{self.tissue}.bed"

        # cmd = f"cut -f1,2,3 {self.tissue_dir}/unprocessed/{bed} \
        #     | bedtools intersect -wa -wb -a - -b {self.resources['tf_motifs']} -F 0.9 \
        #     | bedtools groupby -i - -g 1,2,3 -c 7 -o distinct \
        #     > {self.tissue_dir}/local/tfbindingsites_{self.tissue}.bed"

        # cmd = f"cut -f1,2,3 {self.tissue_dir}/unprocessed/{bed} \
        #     | bedtools intersect -wa -wb -a - -b {self.resources['tf_motifs']} \
        #     | bedtools groupby -i - -g 1,2,3 -c 7 -o distinct \
        #     > {self.tissue_dir}/local/tfbindingsites_{self.tissue}.bed"

        cmd = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"footprint\"}}' {self.tissue_dir}/unprocessed/{bed} \
            > {self.tissue_dir}/local/tfbindingsites_{self.tissue}.bed"
        rename = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, $1\"_\"$2\"_footprint\"}}' {self.tissue_dir}/unprocessed/{bed} \
            > {self.tissue_dir}/unprocessed/tfbindingsites_ref.bed"
        for cmds in [cmd, rename]:
            self._run_cmd(cmds)

    @utils.time_decorator(print_args=True)
    def _merge_cpg(self, bed: str) -> None:
        """Merge individual CPGs with optional liftover"""
        if self.methylation["cpg_liftover"] == True:
            liftover_sort = f"{self.resources['liftover']} \
                {self.tissue_dir}/unprocessed/{bed} \
                {self.resources['liftover_chain']} \
                {self.tissue_dir}/unprocessed/{bed}_lifted \
                {self.tissue_dir}/unprocessed/{bed}_unlifted \
                && bedtools sort -i {self.tissue_dir}/unprocessed/{bed}_lifted \
                > {self.tissue_dir}/unprocessed/{bed}_lifted_sorted \
                && mv {self.tissue_dir}/unprocessed/{bed}_lifted_sorted {self.tissue_dir}/unprocessed/{bed}_lifted"
            self._run_cmd(liftover_sort)

        if self.methylation["cpg_filetype"] == "ENCODE":
            file = f"{self.tissue_dir}/unprocessed/{bed}_gt75"
            gt_gc = f"awk -v FS='\t' -v OFS='\t' '$11 >= 70' {self.tissue_dir}/unprocessed/{bed} \
                > {file}"
            self._run_cmd(gt_gc)
        else:
            file = f"{self.tissue_dir}/unprocessed/{bed}_lifted"

        bedtools_cmd = f"bedtools merge \
            -i {file} \
            -d 1 \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"cpg_methyl\"}}' \
            > {self.tissue_dir}/local/cpg_{self.tissue}_parsed.bed"

        self._run_cmd(bedtools_cmd)

    @utils.time_decorator(print_args=True)
    def prepare_data_files(self) -> None:
        """Pipeline to prepare all bedfiles"""

        ### Make symlinks for shared data files
        for file in self.shared.values():
            src = f"{self.shared_data_dir}/local/{file}"
            dst = f"{self.tissue_dir}/local/{file}"
            if (
                file in NODETYPES_LOCAL
                and file in self.nodes
                or file not in NODETYPES_LOCAL
            ):
                utils.check_and_symlink(
                    src=src,
                    dst=dst,
                )
        ### Make symlinks for histone marks
        for datatype in self.features:
            utils.check_and_symlink(
                src=f"{self.data_dir}/{self.features[datatype]}",
                dst=f"{self.tissue_dir}/local/{datatype}_{self.tissue}.bed",
            )

        ### Make symlink for cpg
        src = f"{self.data_dir}/{self.methylation['cpg']}"
        dst = f"{self.tissue_dir}/unprocessed/{self.methylation['cpg']}"
        utils.check_and_symlink(
            src=src,
            dst=dst,
        )

        if self.nodes is not None:
            if "crms" in self.nodes:
                utils.check_and_symlink(
                    src=f"{self.data_dir}/{self.tissue_specific_nodes['crms']}",
                    dst=f"{self.tissue_dir}/local/crms_{self.tissue}.bed",
                )
            if "tads" in self.nodes:
                self._add_TAD_id(self.tissue_specific_nodes["tads"])
            if "superenhancers" in self.nodes:
                self._superenhancers(self.tissue_specific_nodes["super_enhancer"])
            if "tfbindingsites" in self.nodes:
                self._tf_binding_sites(self.tissue_specific_nodes["tf_binding"])

        self._merge_cpg(self.methylation["cpg"])
