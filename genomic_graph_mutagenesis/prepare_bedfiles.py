#! /usr/bin/env python
# -*- coding: utf-8 -*-  
#
# // TO-DO //
# - [ ] fix _download_shared_files, which is currently broken
#   - [ ] resolve hosting due to bandwidth limits
#   - [ ] fix URLs as they point to incorrect locations

"""Code to preprocess bedfiles before parsing into graph structures"""

import os
import argparse
import requests
import subprocess

import pybedtools

from typing import Dict
from pybedtools.featurefuncs import extend_fields

from utils import dir_check_make, parse_yaml, time_decorator


class GenomeDataPreprocessor:
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
    _split_chromatinloops:
        split double columned file to single column
    _add_tad_id:
        add IDs to tad file
    _fenrir_enhancers:
        enhancers from fenrir bayesian network
    _chromhmm_to_attribute:
        selected chromhmm genome segmentations to attributes
    _tf_binding_clusters:
        clusters of tf binding sites
    _merge_cpg:
        merge book-ended cpg features
    _combine_and_split_histones:
        collapse histones into a single file with bp count
    prepare_data_files:
        main pipeline function

    # Helpers
        HISTONES -- list of histone features
    """

    HISTONES = [
        "H3K27ac",
        "H3K27me3",
        "H3K36me3",
        "H3K4me1",
        "H3K4me3",
        "H3K9me3",
    ]

    def __init__(self, params: Dict[str, Dict[str, str]]) -> None:
        """Initialize the class"""
        self.dirs = params['dirs']
        self.interaction = params['interaction']
        self.options = params['options']
        self.resources = params['resources']
        self.shared = params['shared']
        self.tissue_specific = params['tissue_specific']
        
        self.tissue = self.resources['tissue']
        self.root_dir = self.dirs['root_dir']
        self.tissue_dir = f"{self.root_dir}/{self.tissue}"
        self.data_dir = f'{self.root_dir}/raw_files/{self.tissue}'
        
        # make directories, link files, and download shared files if necessary
        self._make_directories()
        self._symlink_rawdata()
        # self._download_shared_files()

    def _make_directories(self) -> None:
        """Make directories for processing"""
        dir_check_make(f'{self.root_dir}/shared_data')

        for directory in ['local', 'interaction', 'unprocessed', 'histones']:
            dir_check_make(f'{self.tissue_dir}/{directory}')

        for directory in ['local', 'interaction']:
            dir_check_make(f'{self.root_dir}/shared_data/{directory}')

    def _run_cmd(self, cmd: str) -> None:
        """Simple wrapper for subprocess as options across this script are
        constant"""
        subprocess.run(cmd, stdout=None, shell=True)

    def _symlink_rawdata(self) -> None:
        """Make symlinks for tissue specific files in unprocessed folder"""

        def check_and_symlink(dst, src, boolean=False):
            try:
                if boolean == True:
                    if (bool(file) and os.path.exists(src)) and (not os.path.exists(dst)):
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

        for file in ["enhancers_e_e", "enhancers_e_g"]:
            check_and_symlink(
                dst=f"{self.tissue_dir}/interaction/{self.tissue_specific[file]}",
                src=f"{self.data_dir}/{self.tissue_specific[file]}",
                boolean=False,
            )

        interact_files = {
            "circuits": f"{self.dirs['circuit_dir']}/{self.interaction['circuits']}",
            "mirdip": f"{self.dirs['shared_dir']}/interaction/mirdip_tissue/{self.interaction['mirdip']}",
            "mirnatargets": f"{self.dirs['shared_dir']}/interaction/{self.interaction['mirnatargets']}",
            "ppis": f"{self.dirs['shared_dir']}/interaction/{self.interaction['ppis']}",
            "tf_marker": f"{self.dirs['shared_dir']}/interaction/{self.interaction['tf_marker']}",
        }

        for file in interact_files:
            check_and_symlink(
                dst=f"{self.tissue_dir}/interaction/" + self.interaction[file],
                src=interact_files[file],
                boolean=False,
            )

    def _download_shared_files(self) -> None:
        """Download shared local features if not already present"""

        def download(url, filename):
            with open(filename, "wb") as file:
                response = requests.get(url)
                file.write(response.content)

        if os.listdir(f"{self.root_dir}/shared_data/local_feats") == self.shared:
            pass
        else:
            for file in self.shared.values():
                download(
                    f"https://raw.github.com/sciencesteveho/genome_graph_perturbation/raw/master/shared_files/local_feats/{file}",
                    f"{self.root_dir}/shared_data/local_feats/{file}",
                )

    @time_decorator(print_args=True)
    def _add_TAD_id(self, bed: str) -> None:
        """Add identification number to each TAD"""
        cmd = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"tad_\"NR}}' {self.tissue_dir}/unprocessed/{bed} \
            > {self.tissue_dir}/local/tads_{self.tissue}.txt"

        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _split_chromatinloops(self, bed: str) -> None:
        """Split chromatinloop file in separate entries"""
        full_loop = f"sort -k 1,1 -k2,2n {self.tissue_dir}/unprocessed/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $6, \"loop_\"NR}}' \
            > {self.tissue_dir}/local/chromatinloops_{self.tissue}.txt"

        self._run_cmd(full_loop)

    @time_decorator(print_args=True)
    def _fenrir_enhancers(self, e_e_bed: str, e_g_bed: str) -> None:
        """
        Get a list of the individual enhancer sites from FENRIR (Chen et al.,
        Cell Systems, 2021) by combining enhancer-gene and enhancer-enhancer
        networks and sorting
        """
        split_1 = f"tail -n +2 {self.tissue_dir}/unprocessed/{e_e_bed} \
            | cut -f1 \
            | sed -e 's/:/\t/g' -e s'/-/\t/g' \
            > {self.tissue_dir}/unprocessed/{e_e_bed}_1"
        split_2 = f"tail -n +2 {self.tissue_dir}/unprocessed/{e_e_bed} \
            | cut -f2 \
            | sed -e 's/:/\t/g' -e s'/-/\t/g' \
            > {self.tissue_dir}/unprocessed/{e_e_bed}_2"
        enhancer_cat = f"tail -n +2 {self.tissue_dir}/unprocessed/{e_g_bed} \
            | cut -f1 \
            | sed -e 's/:/\t/g' -e s'/-/\t/g' \
            | cat - {self.tissue_dir}/unprocessed/{e_e_bed}_1 {self.tissue_dir}/unprocessed/{e_e_bed}_2 \
            | sort -k1,1 -k2,2n \
            | uniq \
            > {self.tissue_dir}/unprocessed/enhancers.bed"

        liftover_sort = f"{self.resources['liftover']} \
            {self.tissue_dir}/unprocessed/enhancers.bed \
            {self.resources['liftover_chain']} \
            {self.tissue_dir}/local/enhancers_lifted_{self.tissue}.bed \
            {self.tissue_dir}/unprocessed/enhancers_unlifted "

        for cmd in [split_1, split_2, enhancer_cat, liftover_sort]:
            self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _superenhancers(self, bed: str) -> None:
        """Simple parser to remove superenhancer bed unneeded info"""
        cmd = f" tail -n +2 {self.tissue_dir}/unprocessed/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"superenhancer\"}}' \
            > {self.tissue_dir}/local/superenhancers_{self.tissue}.bed"
        
        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _tf_binding_clusters(self, bed: str) -> None:
        """
        Parse tissue-specific transcription factor binding sites from Funk et
        al., Cell Reports, 2020. We use 20-seed HINT TFs with score > 200 and
        use the locations of the motifs, not the footprints, as HINT footprints
        are motif agnostic. Motifs are merged to form tf-binding clusters
        (within 46bp, from Chen et al., Scientific Reports, 2015)
        """
        cmd = f"awk -v FS='\t' -v OFS='\t' '$5 >= 200' {self.tissue_dir}/unprocessed/{bed} \
            | cut -f1,2,3,4 \
            | sed 's/-/\t/g' \
            | cut -f1,2,3,6 \
            | sed 's/\..*$//g' \
            | sort -k1,1 -k2,2n \
            | bedtools merge -i - -d 46 -c 4 -o distinct \
            > {self.tissue_dir}/local/tfbindingclusters_{self.tissue}.bed"
        
        self._run_cmd(cmd)

    @time_decorator(print_args=True)
    def _merge_cpg(self, bed: str) -> None:
        """Merge individual CPGs with optional liftover"""
        if self.options['cpg_liftover'] == True:
            liftover_sort = f"{self.resources['liftover']} \
                {self.tissue_dir}/unprocessed/{bed} \
                {self.resources['liftover_chain']} \
                {self.tissue_dir}/unprocessed/{bed}_lifted \
                {self.tissue_dir}/unprocessed/{bed}_unlifted \
                && bedtools sort -i {self.tissue_dir}/unprocessed/{bed}_lifted \
                > {self.tissue_dir}/unprocessed/{bed}_lifted_sorted \
                && mv {self.tissue_dir}/unprocessed/{bed}_lifted_sorted {self.tissue_dir}/unprocessed/{bed}_lifted"
            self._run_cmd(liftover_sort)

        if self.options['cpg_filetype'] == 'ENCODE':
            file = f"{self.tissue_dir}/unprocessed/{bed}_gt75"
            gt_gc = f"awk -v FS='\t' -v OFS='\t' '$11 >= 75' {self.tissue_dir}/unprocessed/{bed} \
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

    @time_decorator(print_args=True)
    def _combine_and_split_histones(self) -> None:
        """Overlap and merge histone chip-seq bedfiles. Histone marks are
        combined if they overlap and their measurement is score / base pairs

        1 - Histone marks are collapsed, base pairs are kept constant. Clusters
        are made if greater than .5 reciprocal overlap
        2 - Clusters are kept only if more than 1 type of histone mark is
        represented
        3 - Adacent marks are combined, and number of base pairs are
        kept for each histone mark across the combined feature
        """

        histone_idx = {
            'H3K27ac': 4,
            'H3K27me3': 5,
            'H3K36me3': 6,
            'H3K4me1': 7,
            'H3K4me3': 8,
            'H3K9me3': 9,
        }  

        def _add_histone_feat(feature: str, histone: str) -> str:
            feature[3] = histone + '_' + feature[3]
            return feature

        def count_histone_bp(feature: str) -> str:
            feature = extend_fields(feature, 10)
            for histone in histone_idx:
                if histone in feature[3]:
                    feature[histone_idx[histone]] = feature.length
                else:
                    feature[histone_idx[histone]] = 0
            return feature

        all_histones = []
        for histone in self.HISTONES:
            file = self.tissue_specific[histone]
            if file:
                a = pybedtools.BedTool(f'{self.tissue_dir}/unprocessed/{file}')
                b = a.each(_add_histone_feat, histone)
                b.cut([0,1,2,3]).sort().saveas(f'{self.tissue_dir}/histones/{file}')
                all_histones.append(f'{self.tissue_dir}/histones/{file}')

        bedops_everything = f"bedops --everything {' '.join(all_histones)} \
            > {self.tissue_dir}/histones/histones_union.bed"
        bedops_partition = f"bedops --partition {' '.join(all_histones)} \
            > {self.tissue_dir}/histones/histones_partition.bed"
        bedmap = f"bedmap \
            --echo \
            --echo-map-id \
            --delim '\t' \
            --fraction-both 0.5 \
            {self.tissue_dir}/histones/histones_partition.bed \
            {self.tissue_dir}/histones/histones_union.bed \
            | grep ';' - \
            > {self.tissue_dir}/histones/histones_collapsed.bed"  # grep to select the divider, indicating >1 histone
        for command in [bedops_everything, bedops_partition, bedmap]:
            self._run_cmd(command)

        ### chr start end H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me3 H3K9me3
        a = pybedtools.BedTool(f"{self.tissue_dir}/histones/histones_collapsed.bed")
        b = (
            a.each(count_histone_bp)
            .sort()
            .saveas(f"{self.tissue_dir}/histones/histones_collapsed_bp.bed")
        )

        #bedtools merge -i histones_collapsed_bp.bed -c 5,6,7,8,9,10 -o sum
        bedtools_merge = f"bedtools merge \
            -i {self.tissue_dir}/histones/histones_collapsed_bp.bed \
            > {self.tissue_dir}/local/histones_merged_{self.tissue}.bed"
        self._run_cmd(bedtools_merge)

    @time_decorator(print_args=True)
    def prepare_data_files(self) -> None:
        """Pipeline to prepare all bedfiles"""

        ### Make symlinks for shared data files
        for file in self.shared.values():
                src = f'{self.root_dir}/shared_data/local/{file}'
                dst = f'{self.tissue_dir}/local/{file}'
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass

        ### Make symlinks and rename files that do not need preprocessing
        nochange = [
            "ctcf",
            "dnase",
            "H3K27ac",
            "H3K27me3",
            "H3K36me3",
            "H3K4me1",
            "H3K4me3",
            "H3K9me3",
            "polr2a",
        ]
        for datatype in nochange:
            if self.tissue_specific[datatype]:
                src = f'{self.data_dir}/{self.tissue_specific[datatype]}'
                dst = f'{self.tissue_dir}/local/{datatype}_{self.tissue}.bed'
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
        
        self._add_TAD_id(self.tissue_specific['tads'])

        self._split_chromatinloops(self.tissue_specific['chromatinloops'])

        self._superenhancers(self.tissue_specific['super_enhancer'])

        self._fenrir_enhancers(
            self.tissue_specific['enhancers_e_e'],
            self.tissue_specific['enhancers_e_g'],
        )

        self._tf_binding_clusters(self.tissue_specific['tf_binding'])

        self._merge_cpg(self.tissue_specific['cpg'])

        self._combine_and_split_histones()


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
