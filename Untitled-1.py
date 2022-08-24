#! /usr/bin/env python
# -*- coding: utf-8 -*-  
#
# // TO-DO //
# - [ ] fix enhancer_atlas parsing name to something with .bed 
# - [ ] fix _download_shared_files, which is currently broken
#   - [ ] resolve hosting due to bandwidth limits
#   - [ ] fix URLs as they point to incorrect locations
#

"""Code to preprocess bedfiles before parsing into graph structures. 
URL for tissue-specific regulatory builds: http://ftp.ensembl.org/pub/current_regulation/homo_sapiens/RegulatoryFeatureActivity/"""

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
    _format_enhancer_atlas:
        format enhancer atlas as tab delimited file
    _prepare_ensembl:
        remove redundant features and combine with enhancer atlas
    _merge_cpg:
        merge book-ended cpg features
    _combine_histones:
        collapse histones into a single file with bp count
    prepare_data_files:
        main pipeline function

    # Helpers
        HISTONES -- list of histone features
    """
    HISTONES = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3',]

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


    @time_decorator
    def _add_TAD_id(self, bed: str) -> None:
        """Add identification number to each TAD"""
        cmd = f"awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"tad_\"NR}}' {self.root_tissue}/unprocessed/{bed} \
            > {self.root_tissue}/local/tads_{self.tissue}.txt"

        self._run_cmd(cmd)


    @time_decorator
    def _format_enhancer_atlas(self, bed: str) -> None:
        """Format enhancer atlas to bed"""
        tab_delim = f"sed -e 's/:/\t/g' -e 's/_EN/\tEN/g' -e 's/\$/\t/g' -e 's/-/\t/g' {self.root_tissue}/unprocessed/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"enhancer_ATLAS_\"$1\"_\"$2}}' \
            > {self.root_tissue}/interaction/{bed}.interaction"

        tabbed = f"sed -e 's/:/\t/g' -e 's/_EN/\tEN/g' -e 's/\$/\t/g' {self.root_tissue}/unprocessed/{bed} \
            | sort -k1,1 -k2,2n \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, \"enhancer_ATLAS\"}}' \
            | sed -e 's/-/\t/' \
            > {self.root_tissue}/interaction/{bed}.tabbed"

        for cmd in [tab_delim, tabbed]:
            self._run_cmd(cmd)


    @time_decorator
    def _split_chromatinloops(self, bed: str) -> None:
        """Split chromatinloop file in separate entries"""
        split_1 = f"sort -k 1,1 -k2,2n {self.root_tissue}/unprocessed/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"loop_\"NR}}' \
            > {self.root_tissue}/unprocessed/{bed}_1"
        split_2 = f"sort -k 1,1 -k2,2n {self.root_tissue}/unprocessed/{bed} \
            | awk -v FS='\t' -v OFS='\t' '{{print $4, $5, $6, \"loop_\"NR}}' \
            > {self.root_tissue}/unprocessed/{bed}_2"
        loop_cat = f'cat {self.root_tissue}/unprocessed/{bed}_1 {self.root_tissue}/unprocessed/{bed}_2 \
            | sort -k 1,1 -k2,2n \
            > {self.root_tissue}/local/chromatinloops_{self.tissue}.txt'

        for cmd in [split_1, split_2, loop_cat]:
            self._run_cmd(cmd)


    @time_decorator
    def _fenrir_enhancers(self, e_e_bed: str, e_g_bed: str) -> None:
        """
        Get a list of the individual enhancer sites from FENRIR (Chen et al., Cell Systems, 2021) by combining enhancer-gene and enhancer-enhancer networks and sorting
        """
        split_1 = f"tail -n +2 {self.root_tissue}/unprocessed/{e_e_bed} \
            | cut -f1 \
            > {self.root_tissue}/unprocessed/{e_e_bed}_1"
        split_2 = f"tail -n +2 {self.root_tissue}/unprocessed/{e_e_bed} \
            | cut -f2 \
            > {self.root_tissue}/unprocessed/{e_e_bed}_2"
        enhancer_cat = f"tail -n +2 {self.root_tissue}/unprocessed/{e_g_bed} \
            | cut -f1 \
            | cat - {self.root_tissue}/unprocessed/{e_e_bed}_1 {self.root_tissue}/unprocessed/{e_e_bed}_2 \
            | sort -u \
            > {self.root_tissue}/unprocessed/enhancers.bed"

        liftover_sort = f"./shared_data/liftOver \
            {self.root_tissue}/unprocessed/enhancers.bed \
            {self.root_dir}/shared_data/hg19ToHg38.over.chain.gz \
            {self.root_tissue}/local/enhancers_lifted_{self.tissue}.bed \
            {self.root_tissue}/unprocessed/enhancers_unlifted "

        for cmd in [split_1, split_2, enhancer_cat, liftover_sort]:
            self._run_cmd(cmd)

    @time_decorator
    def _tf_binding_sites(self, bed: str) -> None:
        """
        Parse tissue-specific transcription factor binding sites from Funk et al., Cell Reports, 2020.
        We use 20-seed HINT TFs with score > 200 and use the locations of the motifs, not the footprints,
        as HINT footprints are motif agnostic. Motifs are merged with bedtools if they overlap bases.
        """
        cmd = f"awk -v FS='\t' -v OFS='\t' '{{sub(/:/, \"\t\", $1); sub(/-/, \"\t\", $1)}}1' {bed} \
            | awk -v FS='\t' -v OFS='\t' '$11 > 200' \
            | cut -f1,2,3,7 \
            | sort -k1,1 -k2,2n \
            | sed -e 's/-/\t/g' -e 's/_/\t/g' \
            | cut -f1,2,3,6 \
            | sed 's/\..*$//g' \
            | bedtools merge -i - -c 4 -o distinct \
            > {self.root_tissue}/local/tf_binding_sites_{self.tissue}.bed"
        
        self._run_cmd(cmd)

    @time_decorator
    def _merge_cpg(self, bed: str) -> None:
        """Merge individual CPGs with optional liftover"""
        if self.options['cpg_liftover'] == True:
            liftover_sort = f"./shared_data/liftOver \
                {self.root_tissue}/unprocessed/{bed} \
                {self.root_dir}/shared_data/hg19ToHg38.over.chain.gz \
                {self.root_tissue}/unprocessed/{bed}_lifted \
                {self.root_tissue}/unprocessed/{bed}_unlifted \
                && bedtools sort -i {self.root_tissue}/unprocessed/{bed}_lifted \
                > {self.root_tissue}/unprocessed/{bed}_lifted_sorted \
                && mv {self.root_tissue}/unprocessed/{bed}_lifted_sorted {self.root_tissue}/unprocessed/{bed}_lifted"
            self._run_cmd(liftover_sort)

        if self.options['cpg_filetype'] == 'ENCODE':
            file = f"{self.root_tissue}/unprocessed/{bed}_gt75"
            gt_gc = f"awk -v FS='\t' -v OFS='\t' '$11 >= 75' {self.root_tissue}/unprocessed/{bed} \
                > {file}"
            self._run_cmd(gt_gc)
        else:
            file = f"{self.root_tissue}/unprocessed/{bed}_lifted"
            
        bedtools_cmd = f"bedtools merge -i {file} -d 1 \
            | awk -v FS='\t' -v OFS='\t' '{{print $1, $2, $3, \"cpg_methyl\"}}' \
            > {self.root_tissue}/local/cpg_{self.tissue}_parsed.bed"

        self._run_cmd(bedtools_cmd)

    @time_decorator
    def _combine_histones(self) -> None:
        """Overlap and merge histone chip-seq bedfiles. Histone marks are combined if they overlap and their measurement is score / base pairs
        
        1 - Histone marks are collapsed, base pairs are kept constant
        2 - Adjacent marks are combined, and number of base pairs are kept for each histone mark across the combined feature
        """

        histone_idx = {
            'H3K27ac': 4,
            'H3K27me3': 5,
            'H3K36me3': 6,
            'H3K4me1': 7,
            'H3K4me3': 8,
            'H3K9ac': 9,
            'H3K9me3': 10,
        }

        def _add_histone_feat(feature: str, histone: str) -> str:
            feature[3] = histone + '_' + feature[3]
            return feature

        def count_histone_bp(feature: str) -> str:
            feature = extend_fields(feature, 12)
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
                a = pybedtools.BedTool(f'{self.root_tissue}/unprocessed/{file}')
                b = a.each(_add_histone_feat, histone)
                b.cut([0,1,2,3]).sort().saveas(f'{self.root_tissue}/histones/{file}')
                all_histones.append(f'{self.root_tissue}/histones/{file}')

        bedops_everything = f"bedops --everything {' '.join(all_histones)} \
            > {self.root_tissue}/histones/histones_union.bed"
        bedops_partition = f"bedops --partition {' '.join(all_histones)} \
            > {self.root_tissue}/histones/histones_partition.bed"
        bedmap = f"bedmap --echo --echo-map-id --delim '\t' {self.root_tissue}/histones/histones_partition.bed {self.root_tissue}/histones/histones_union.bed \
            > {self.root_tissue}/histones/histones_collapsed.bed"
        bedtools_merge = f"bedtools merge -i {self.root_tissue}/histones/histones_collapsed_bp.bed -c 5,6,7,8,9,10,11 -o sum \
            > {self.root_tissue}/local/histones_merged_{self.tissue}.bed"

        for command in [bedops_everything, bedops_partition, bedmap]:
            self._run_cmd(command)

        a = pybedtools.BedTool(f'{self.root_tissue}/histones/histones_collapsed.bed')
        b = a.each(count_histone_bp).sort().saveas(f'{self.root_tissue}/histones/histones_collapsed_bp.bed')

        ### chr start end H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me3 H3K9ac H3K9me3
        self._run_cmd(bedtools_merge)

    @time_decorator
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
        # self._split_chromatinloops(self.tissue_specific['chromatinloops'])

        # self._add_TAD_id(self.tissue_specific['tads'])

        # self._format_enhancer_atlas(self.tissue_specific['enhancers'])

        # self._merge_cpg(self.tissue_specific['cpg'])

        # self._combine_histones()


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

    preprocessObject._format_enhancer_atlas(
        preprocessObject.tissue_specific['ehancers_e_e'],
        preprocessObject.tissue_specific['enhaners_e_g'],
    )

    preprocessObject._fenrir_enhancers(preprocessObject.tissue_specific['tf_binding'])

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
