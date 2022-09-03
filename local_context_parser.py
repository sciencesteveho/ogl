#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# !!! - [ ] fix attr_save, too slow right now
# !!! - [ ] fix slop - at some point, chr is being put down instead of distance
# !!! - [ ] add enhancers to aggregate attributes and add it to the attribute ref code
# - [ ] fix, add number of cores as a number in params
# - [ ] finish class docstring
# - [ ] re-work the attribute index code to get node indexes from the initial bed dict and add the node type for each node
# - [ ] add indexes as node attribute 'feat' and change current 'feat' to 'feat_type'
# - [ ] Properly integrate the code for splitting the files by gene. Currently located at the bottom of parser 
# - [ ] Add the code for graph construction
# - [??] Add node type as an attribute. Type as each of the input bed files (is it a loop, type of chrom mark, etc)
#

"""Parse local genomic data to nodes and attributes"""

import argparse
import os
import pickle
import subprocess

import pybedtools

from itertools import repeat
from multiprocessing import Pool
from pybedtools.featurefuncs import extend_fields
from typing import Dict, List, Optional, Tuple
from subprocess import Popen, PIPE

from utils import bool_check_attributes, dir_check_make, parse_yaml, time_decorator
from target_labels_train_split import _filter_low_tpm


@time_decorator(print_args=True)
def _filtered_gene_windows(
    gencode: str,
    chromfile: str,
    tissue: str,
    tpm_file: str,
    ):
    """
    Filter out genes in a GTEx tissue with less than 0.1 tpm across 20% of samples in that tissue.
    Additionally, we exclude analysis of sex chromosomes
    Returns pybedtools object with +/- 250kb windows around that gene
    """
    tpm_filtered_genes = _filter_low_tpm(
        tissue,
        tpm_file,
        return_list=True,
    )
    genes = pybedtools.BedTool(gencode)
    genes_filtered = genes.filter(
        lambda x: x[3] in tpm_filtered_genes and x[0] not in ['chrX', 'chrY']
        )

    return genes_filtered.slop(g=chromfile, b=250000)\
        .cut([0, 1, 2, 3])\
        .sort(), tpm_filtered_genes


@time_decorator(print_args=True)
def _gene_window(dir: str) -> List[str]:
    """
    Returns a list of bedfiles within the directory.
    """
    return [
        file for file in os.listdir(dir)
        if os.path.isfile(f'{dir}/{file}')
        ]


class LocalContextFeatures:
    """Object that parses local genomic data into graph edges

    Args:
        bedfiles // dictionary containing each local genomic datatype as bedtool obj
        windows // bedtool object of windows +/- 250k of protein coding genes
        params // configuration vals from yaml 

    Methods
    ----------
    _make_directories:
        prepare necessary directories
    _window_specific_features_dict:
        retrieve bed info for specific windows
    _slop_sort:
        apply slop to each bed and sort it
    _save_feature_indexes:
        save indexes for each node name
    _bed_intersect:
        intersect each bed with every datatype
    _aggregate_attributes:
        get attributes for each node
    _genesort_attributes:
        save attributes for empty genes
    _generate_edges:
        convert bed lines into edges for each gene
    parse_context_data:
        main pipeline function

    # Helpers
        ATTRIBUTES -- list of node attribute types
        DIRECT -- list of datatypes that only get direct overlaps, no slop
        FEAT_WINDOWS -- dictionary of each nodetype: overlap windows
        NODES -- list of nodetypes
        ONEHOT_NODETYPE -- dictionary of node type one-hot vectors 

    The following features have node representations:
        Tissue-specific
            chromatinloops
            enhancers (tissue-specific)
            histone binding clusters (collapsed)
            transcription factor binding clusters
            tads

        Genome-static
            cpgislands
            gencode (genes)
            miRNA targets
            poly(a) binding sites
            promoters 
            rna binding protein binding sites
            transcription start sites

    The following are represented as attributes:
        Tissue-specific
            CpG methylation

            ChIP-seq peaks
                ctcf ChIP-seq peaks
                DNase ChIP-seq peaks
                H3K27ac ChIP-seq peaks
                H3K27me3 ChIP-seq peaks
                H3K36me3 ChIP-seq peaks
                H3K4me1 ChIP-seq peaks
                H3K4me3 ChIP-seq peaks
                H3K9me3 ChIP-seq peaks
                polr2a ChIP-seq peaks

            ChromHMM segmentations
                enhancers
                bivalent enhancers
                genic enhancers
                active TSS
                flanking active TSS
                bivalent TSS
                flanking transcription
                active transcription
                weak transcription
                zinc-finger proteins

        Genome-static
            gc content
            microsatellites
            conservation (phastcons)
            LINEs
            long terminal repeats
            simple repeats
            SINEs
    """

    # list helpers
    ATTRIBUTES = ['gc', 'cpg', 'ctcf', 'dnase', 'enh', 'enhbiv', 'enhg', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3', 'line', 'ltr', 'microsatellites', 'phastcons', 'polr2a', 'rnarepeat', 'simplerepeats', 'sine', 'tssa', 'tssaflnk', 'tssbiv', 'txflnk', 'tx', 'txwk', 'znf']
    DIRECT = ['chromatinloops', 'tads']
    NODES = ['chromatinloops', 'cpgislands', 'enhancers', 'gencode', 'histones', 'mirnatargets', 'polyasites', 'promoters', 'rbpbindingsites', 'tads', 'tfbindingclusters', 'tss']

    # dict helpers
    ONEHOT_NODETYPE = {
        'chromatinloops': [1,0,0,0,0,0,0,0,0,0,0,0],
        'cpgislands': [0,1,0,0,0,0,0,0,0,0,0,0],
        'enhancers': [0,0,1,0,0,0,0,0,0,0,0,0],
        'gencode': [0,0,0,1,0,0,0,0,0,0,0,0],
        'histones': [0,0,0,0,1,0,0,0,0,0,0,0],
        'mirnatargets': [0,0,0,0,0,1,0,0,0,0,0,0],
        'polyasites': [0,0,0,0,0,0,1,0,0,0,0,0],
        'promoters': [0,0,0,0,0,0,0,1,0,0,0,0],
        'rbpbindingsites': [0,0,0,0,0,0,0,0,1,0,0,0],
        'tads': [0,0,0,0,0,0,0,0,0,1,0,0],
        'tfbindingclusters': [0,0,0,0,0,0,0,0,0,0,1,0],
        'tss': [0,0,0,0,0,0,0,0,0,0,0,1],
    }

    # cpgislands - 2kb, baced on precedence from CpGcluster
    # enhancers - can vary widely, so dependent on 3d chromatin structure and from FENRIR networks
    # direct binding, such as mirna, polyasites, rbps are set to 500bp
    FEAT_WINDOWS = {
        'cpgislands': 2000,
        'enhancers': 2000,
        'gencode': 2500,
        'histones': 2000,
        'mirnatargets': 500,
        'polyasites': 500,
        'promoters': 1000,
        'rbpbindingsites': 500,
        'tfbindingclusters': 2000,
        'tss': 2000,
    }

    def __init__(
        self,
        bedfiles: List[str],
        params: Dict[str, Dict[str, str]]):
        """Initialize the class"""
        self.bedfiles = bedfiles

        self.tissue = params['resources']['tissue']
        self.tissue_name = params['resources']['tissue_name']
        self.tissue_specific = params['tissue_specific']
        self.chromfile = params['resources']['chromfile']
        self.fasta = params['resources']['fasta']
        self.shared_data = params['shared']

        self.root_dir = params['dirs']['root_dir']
        self.parse_dir = f'{self.root_dir}/{self.tissue}/parsing'
        self.local_dir = f'{self.root_dir}/{self.tissue}/local'
        self.attribute_dir = f"{self.parse_dir}/attributes"

        self.parsed_features = {
            'gc': '_',
            'cpg': self.tissue_specific['cpg'],
            'ctcf': self.tissue_specific['ctcf'],
            'dnase': self.tissue_specific['dnase'],
            'enh': f'{self.local_dir}/enh.bed',
            'enhbiv': f'{self.local_dir}/enhiv.bed',
            'enhg': f'{self.local_dir}/enhg.bed',
            'H3K27ac': self.tissue_specific['H3K27ac'],
            'H3K27me3': self.tissue_specific['H3K27me3'],
            'H3K36me3': self.tissue_specific['H3K36me3'],
            'H3K4me1': self.tissue_specific['H3K4me1'],
            'H3K4me3': self.tissue_specific['H3K4me3'],
            'H3K9me3': self.tissue_specific['H3K9me3'],
            'microsatellites': self.shared_data['microsatellites'],
            'phastcons': self.shared_data['phastcons'],
            'polr2a': self.tissue_specific['polr2a'],
            'rnarepeat': self.shared_data['rnarepeat'],
            'simplerepeats': self.shared_data['simplerepeats'],
            'line': self.shared_data['line'],
            'ltr': self.shared_data['ltr'],
            'sine': self.shared_data['sine'],
            'tssa': f'{self.local_dir}/tssa.bed',
            'tssaflnk': f'{self.local_dir}/tssaflnk.bed',
            'tssbiv': f'{self.local_dir}/tssbiv.bed',
            'txflnk': f'{self.local_dir}/txflnk.bed',
            'tx': f'{self.local_dir}/tx.bed',
            'txwk': f'{self.local_dir}/txwk.bed',
            'znf': f'{self.local_dir}/znf.bed',
        }

        # make directories
        self._make_directories()

    def _make_directories(self) -> None:
        """Directories for parsing genomic bedfiles into graph edges and nodes"""
        dir_check_make(self.parse_dir)

        for directory in ['edges/genes', 'attributes', 'intermediate/slopped', 'intermediate/sorted']:
            dir_check_make(f'{self.parse_dir}/{directory}')

        for attribute in self.ATTRIBUTES:
            if bool_check_attributes(attribute, self.parsed_features[attribute]):
                dir_check_make(f'{self.attribute_dir}/{attribute}/genes')

    @time_decorator(print_args=True)
    def _region_specific_features_dict(self, bed: str) -> List[Dict[str, pybedtools.bedtool.BedTool]]:
        """
        _lorem
        """
        def rename_feat_chr_start(feature: str) -> str:
            """Add chr, start to feature name
            Cpgislands add prefix to feature names
            Histones add an additional column
            """
            rename_strings = ['cpgislands', 'histones', 'enhancers']
            if prefix in rename_strings:
                feature[3] = f'{prefix}_{feature[0]}_{feature[1]}'
            else:
                feature[3] = f'{feature[3]}_{feature[0]}_{feature[1]}'
            return feature

        # prepare data as pybedtools objects
        bed_dict = {}
        prefix = bed.split("_")[0]
        a = pybedtools.BedTool(f'{self.root_dir}/{self.tissue}/gene_regions_tpm_filtered.bed')
        b = pybedtools.BedTool(f'{self.root_dir}/{self.tissue}/local/{bed}').sort()
        ab = a.intersect(b, wb=True, sorted=True)
        col_idx = ab.field_count()  # get number of columns

        # take specific windows and format each file
        regioned = ab.cut(list(range(4, col_idx))) 
        if prefix in self.NODES and prefix != 'gencode':
            result = regioned.each(rename_feat_chr_start)\
                .saveas()\
                .cut([0, 1, 2, 3])
            bed_dict[prefix] = pybedtools.BedTool(str(result), from_string=True)
        else:
            bed_dict[prefix] = regioned.cut([0, 1, 2 ,3])

        return bed_dict

    @time_decorator(print_args=True)
    def _slop_sort(
        self,
        bedinstance: Dict[str, str],
        chromfile: str
        ) -> Tuple[Dict[str, pybedtools.bedtool.BedTool], Dict[str, pybedtools.bedtool.BedTool]]:
        """Slop each line of a bedfile to get all features within a window

        Args:
            bedinstance // a region-filtered genomic bedfile
            chromfile // textfile with sizes of each chromosome in hg38
        
        Returns:
            bedinstance_sorted -- sorted bed
            bedinstance_slopped -- bed slopped by amount in feat_window
        """
        bedinstance_slopped, bedinstance_sorted = {}, {}
        for key in bedinstance.keys():
            bedinstance_sorted[key] = bedinstance[key].sort()
            if key in self.ATTRIBUTES + self.DIRECT:
                pass
            else:
                nodes = bedinstance[key].slop(g=chromfile, b=self.FEAT_WINDOWS[key])\
                    .sort()
                newstrings = []
                for line_1, line_2 in zip(nodes, bedinstance[key]):
                    newstrings.append(str(line_1).split('\n')[0] + '\t' + str(line_2))
                bedinstance_slopped[key] = pybedtools.BedTool(''.join(newstrings), from_string=True)\
                    .sort()
        return bedinstance_sorted, bedinstance_slopped

    @time_decorator(print_args=True)
    def _save_feature_indexes(self, bedinstance_sorted: Dict[str, pybedtools.bedtool.BedTool]) -> None:
        """Gets a list of the possible node names, dedupes them, annotates 
        each with their data type and saves the dict for using later.
        """
        feats = [
            (line[3], key) for key in bedinstance_sorted.keys()
            if key in self.NODES
            for line in bedinstance_sorted[key]
        ]
        feats_deduped = list(set([tup for tup in feats]))
        feats_deduped.sort()
        feat_idxs = {
            val[0]: (idx, val[1])
            for idx, val in enumerate(feats_deduped)
            }

        ### save the dictionary for later use
        output = open(f'{self.root_dir}/{self.tissue}/{self.tissue}_feat_idxs.pkl', "wb")
        try:
            pickle.dump(feat_idxs, output)
        finally:
            output.close()

    @time_decorator(print_args=True)
    def _bed_intersect(
        self,
        node_type: str,
        all_files: str
        ) -> None:
        """Function to intersect a slopped bed entry with all other node types.
        Each bed is slopped then intersected twice. First, it is intersected with every other node type. Then, the intersected bed is filtered to only keep edges within the gene region.

        Args:
            node_type // _description_
            all_files // _description_

        Raises:
            AssertionError: _description_
        """
        print(f'starting combinations {node_type}')

        def _unix_intersect(node_type: str, type: Optional[str]=None) -> None:
            """Intersect and cut relevant columns"""
            if type == 'direct':
                folder = 'sorted'
                cut_cmd = ''
            else:
                folder = 'slopped'
                cut_cmd =" | cut -f5,6,7,8,9,10,11,12"

            final_cmd = f'bedtools intersect \
                -wa \
                -wb \
                -sorted \
                -a {self.parse_dir}/intermediate/{folder}/{node_type}.bed \
                -b {all_files}'

            with open(f'{self.parse_dir}/edges/{node_type}.bed', "w") as outfile:
                subprocess.run(
                    final_cmd + cut_cmd,
                    stdout=outfile,
                    shell=True
                    )
            outfile.close()

        def _filter_duplicate_bed_entries(bedfile: pybedtools.bedtool.BedTool) -> pybedtools.bedtool.BedTool:
            """Filters a bedfile by removing entries that are identical"""
            return bedfile.filter(lambda x: [x[0], x[1], x[2], x[3]] != [x[4], x[5], x[6], x[7]])\
                .saveas()

        def _add_distance(feature: str) -> str:
            """Add distance as [8]th field to each overlap interval"""
            feature = extend_fields(feature, 9)
            feature[8] = max(
                int(feature[1]), int(feature[5])) - min(int(feature[2]), int(feature[5]))
            return feature

        if node_type in self.DIRECT:
            _unix_intersect(node_type, type='direct')
            a = pybedtools.BedTool(f'{self.parse_dir}/edges/{node_type}.bed')
            b = _filter_duplicate_bed_entries(a)\
                .sort()\
                .saveas(f'{self.parse_dir}/edges/{node_type}_dupes_removed')
            cut_cmd = 'cut -f1,2,3,4,5,6,7,8,9,12'
        else:
            _unix_intersect(node_type)
            a = pybedtools.BedTool(f'{self.parse_dir}/edges/{node_type}.bed')
            b = _filter_duplicate_bed_entries(a)\
                .each(_add_distance)\
                .sort()\
                .saveas(f'{self.parse_dir}/edges/{node_type}_dupes_removed')
            cut_cmd = 'cut -f1,2,3,4,5,6,7,8,9,13'

        print(f'finished intersect for {node_type}. proceeding with windows')
        
        window_cmd = f'bedtools intersect \
            -wa \
            -wb \
            -sorted \
            -a {self.parse_dir}/edges/{node_type}_dupes_removed \
            -b {self.root_dir}/{self.tissue}/gene_regions_tpm_filtered.bed | '

        with open(f'{self.parse_dir}/edges/{node_type}_genewindow.txt', "w") as outfile:
            subprocess.run(
                window_cmd + cut_cmd,
                stdout=outfile,
                shell=True
                )
        outfile.close()

    @time_decorator(print_args=True)
    def _aggregate_attributes(self, node_type: str) -> None:
        """For each node of a node_type get their overlap with gene windows
        then aggregate total nucleotides, gc content, and all other attributes

        Args:
            node_type // node datatype in self.NODES
        """
        # start_time = time.monotonic()
        def add_size(feature: str) -> str:
            """
            """
            feature = extend_fields(feature, 5)
            feature[4] = feature.end - feature.start
            return feature

        def sum_gc(feature: str) -> str:
            """
            """
            feature[14] = int(feature[9]) + int(feature[10])
            return feature

        ### Add size as 5th column for each entry 
        sorted = pybedtools.BedTool(f'{self.parse_dir}/intermediate/sorted/{node_type}.bed')\
            .each(add_size)\
            .sort()
        
        ### total basepair number for phastcons // columns should be ordered as below
        ### chr str end feat    size    gene
        b = sorted.intersect(f'{self.root_dir}/{self.tissue}/gene_regions_tpm_filtered.bed', \
            wa=True, \
            wb=True, \
            sorted=True)\
            .cut([0,1,2,3,8,4])\
            .sort()

        for attribute in self.ATTRIBUTES:
            if bool_check_attributes(attribute, self.parsed_features[attribute]):
                print(f'{attribute} for {node_type}')
                if attribute == 'gc':
                    b.nucleotide_content(fi=self.fasta)\
                    .each(sum_gc)\
                    .cut([0,1,2,3,4,5,14])\
                    .saveas(f'{self.attribute_dir}/{attribute}/{node_type}_{attribute}')
                else:
                    b.intersect(f'{self.parse_dir}/intermediate/sorted/{attribute}.bed', wao=True, sorted=True)\
                    .cut([0,1,2,3,4,5,10])\
                    .saveas(f'{self.attribute_dir}/{attribute}/{node_type}_{attribute}')

                with open(f'{self.attribute_dir}/{attribute}/{node_type}_{attribute}_percentage', "w") as outfile:
                    subprocess.run(
                        f'datamash -s -g 1,2,3,4,5 sum 6,7 < {self.attribute_dir}/{attribute}/{node_type}_{attribute}',
                        stdout=outfile,
                        shell=True
                        )
                outfile.close()

    @time_decorator(print_args=True)
    def _genesort_attributes(self, attribute: str) -> None:
        """Lorem"""
        cat = f"cat {self.attribute_dir}/{attribute}/*_percentage* \
            | sort -k5,5 --parallel=16 -S 50% \
            > {self.attribute_dir}/{attribute}/all_{attribute}.txt"

        awk = f"awk -F'\t' '{{print>\"{self.attribute_dir}/{attribute}/genes/\"$5}}' \
            {self.attribute_dir}/{attribute}/all_{attribute}.txt"

        for cmd in [cat, awk]:
            subprocess.run(cmd, stdout=None, shell=True)

    @time_decorator(print_args=True)
    def _generate_edges(self) -> None:
        """Unix concatenate and sort each edge file"""
        def _chk_file_and_run(file: str, cmd: str) -> None:
            """Check that a file does not exist before calling subprocess"""
            if os.path.isfile(file) and os.path.getsize(file) != 0:
                pass
            else:
                subprocess.run(cmd, stdout=None, shell=True)

        cmds = {
            'cat_cmd': [f"cat {self.parse_dir}/edges/*genewindow* >", \
                f"{self.parse_dir}/edges/all_concat.bed"],
            'sort_cmd': [f"LC_ALL=C sort --parallel=72 -S 80% -k10,10 {self.parse_dir}/edges/all_concat.bed >", \
                f"{self.parse_dir}/edges/all_concat_sorted.bed"],
        }

        for cmd in cmds:
            _chk_file_and_run(
                cmds[cmd][1],
                cmds[cmd][0] + cmds[cmd][1],
            )

        sorted_beds = f"{self.parse_dir}/edges/all_concat_sorted.bed"
        awk_cmd = f"awk -F'\t' '{{print>\"{self.parse_dir}/edges/genes/\"$5}}' {self.parse_dir}/edges/all_concat_sorted.bed" 

        if os.path.isfile(sorted_beds) and os.path.getsize(sorted_beds) !=0:
            subprocess.run(awk_cmd, stdout=None, shell=True)

    @time_decorator(print_args=True)
    def _save_node_attributes(self, node: str) -> None:
        """
        Save attributes for all node entries. Used during graph construction
        for gene_nodes that fall outside of the gene window and for some gene_nodes
        from interaction data
        """
        attr_dict, set_dict = {}, {}  # dict[gene] = [chr, start, end, size, gc]
        for attribute in self.ATTRIBUTES:
            if bool_check_attributes(attribute, self.parsed_features[attribute]):
                filename = f'{self.parse_dir}/attributes/{attribute}/{node}_{attribute}_percentage'
                with open(filename, 'r') as file:
                    lines = []
                    for line in file:
                        stripped_line = line.rstrip().split('\t')
                        del stripped_line[4]
                        lines.append(tuple(stripped_line))
                    set_dict[attribute] = set(lines)

        for attribute in set_dict.keys():
            for line in set_dict[attribute]:
                if attribute == 'gc':
                    attr_dict[line[3]] = {
                        'type': self.ONEHOT_NODETYPE[node],
                        'chr': line[0].replace('chr', ''),
                        'start': line[1],
                        'end': line[2],
                        'size': line[4],
                        'gc': line[5],
                        'polyadenylation': 0,
                    }
                else:
                    attr_dict[line[3]][attribute] = line[5]
        
        output = open(f'{self.parse_dir}/attributes/{node}_reference.pkl', "wb")
        try:
            pickle.dump(attr_dict, output)
        finally:
            output.close()

    @time_decorator(print_args=True)
    def parse_context_data(self) -> None:
        """_summary_

        Args:
            a // _description_
            b // _description_

        Raises:
            AssertionError: _description_
        
        Returns:
            c -- _description_
        """
        @time_decorator(print_args=True)
        def _save_intermediate(
            bed_dictionary: Dict[str, pybedtools.bedtool.BedTool],
            folder: str
            ) -> None:
            """Save region specific bedfiles"""
            for key in bed_dictionary:
                file = f'{self.parse_dir}/intermediate/{folder}/{key}.bed'
                if not os.path.exists(file):
                    bed_dictionary[key].saveas(file)

        def _intersect_combinations(
            bedinstance_slopped: Dict[str, pybedtools.bedtool.BedTool]
            ) -> Dict[str, pybedtools.bedtool.BedTool]:
            """Lorem Ipsum"""
            nodes = [key for key in bedinstance_slopped]
            return {key:nodes for key in self.NODES}

        @time_decorator(print_args=True)
        def _pre_concatenate_all_files(all_files: str) -> None:
            """Lorem Ipsum"""
            if not os.path.exists(all_files) or os.stat(all_files).st_size == 0:
                cat_cmd = ['cat'] + [f'{self.parse_dir}/intermediate/sorted/' + x + '.bed' for x in bedinstance_slopped]  
                sort_cmd = 'sort -k1,1 -k2,2n'
                concat = Popen(cat_cmd, stdout=PIPE)
                with open(all_files, "w") as outfile:
                    subprocess.run(
                        sort_cmd,
                        stdin=concat.stdout,
                        stdout=outfile,
                        shell=True
                        )
                outfile.close()

        ### process windows and renaming 
        pool = Pool(processes=32)
        bedinstance = pool.map(self._region_specific_features_dict,\
            [bed for bed in self.bedfiles])
        pool.close()  # re-open and close pool after every multi-process

        ### convert back to dictionary
        bedinstance = {key.casefold():value for element in bedinstance for key, value in element.items()}

        ### sort and extend windows according to FEAT_WINDOWS
        bedinstance_sorted, bedinstance_slopped = self._slop_sort(bedinstance=bedinstance, chromfile=self.chromfile)

        ### save a list of the nodes and their indexes
        self._save_feature_indexes(bedinstance_sorted)

        ### save intermediate files
        _save_intermediate(bedinstance_sorted, folder='sorted')
        _save_intermediate(bedinstance_slopped, folder='slopped')

        ### get keys for intersect features and attribute features
        combinations = _intersect_combinations(bedinstance_slopped)

        ### pre-concatenate to save time
        all_files = f'{self.parse_dir}/intermediate/sorted/all_files_concatenated.bed'
        _pre_concatenate_all_files(all_files)

        ### perform intersects across all feature types
        pool = Pool(processes=32)
        pool.starmap(self._bed_intersect, zip(combinations.keys(), repeat(all_files)))
        pool.close()

        ### get size and all attributes
        pool = Pool(processes=32)
        pool.map(self._aggregate_attributes, combinations)
        pool.close()

        ### parse attributes into individual files
        pool = Pool(processes=28)
        pool.map(self._genesort_attributes, self.ATTRIBUTES)
        pool.close()

        ### parse edges into individual files
        self._generate_edges()

        ### save node attributes as reference for later
        pool = Pool(processes=12)
        pool.map(self._save_node_attributes, self.NODES)
        pool.close()


def main() -> None:
    """Pipeline to parse genomic data into edges"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to .yaml file with filenames"
    )

    args = parser.parse_args()
    params = parse_yaml(args.config)

    window, tpm_filtered_genes = _filtered_gene_windows(
        f"shared_data/local/{params['shared']['gencode']}",
        params['resources']['chromfile'],
        params['resources']['tissue'],
        params['resources']['tpm'],
    )

    ### save window file
    window.saveas(f"{params['dirs']['root_dir']}/{params['resources']['tissue']}/gene_regions_tpm_filtered.bed")

    ### get features within 500kb of protein coding regions
    bedfiles = _gene_window(
        dir=f"{params['dirs']['root_dir']}/{params['resources']['tissue']}/local",
    )

    localparseObject = LocalContextFeatures(
        bedfiles=bedfiles,
        params=params,
    )

    ### run parallelized pipeline! 
    localparseObject.parse_context_data()

    ### cleanup
    pybedtools.cleanup(remove_all=True)


if __name__ == '__main__':
    main()