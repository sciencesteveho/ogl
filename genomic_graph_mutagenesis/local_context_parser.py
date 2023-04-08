#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] fix, add number of cores as a number in params
# - [ ] finish class docstring
#
# // PLAN //
# Take the gene window - within 500kb of a TPM filtered gene
# Make slop + take edges b/w local contexts 
# Get edges b/w local contexts and BASE NODES
# Add all edges, but only keep edges that can be traversed back to a base node
# Aggregate and save attributes

"""Parse local genomic data to nodes and attributes"""

import argparse
from itertools import repeat
from multiprocessing import Pool
import os
import pickle
import subprocess
from subprocess import Popen, PIPE
from typing import Dict, List, Optional, Tuple

import pybedtools
from pybedtools.featurefuncs import extend_fields

from utils import (
    _tpm_filter_gene_windows,
    dir_check_make,
    parse_yaml,
    time_decorator,
)


def _listdir_isfile_wrapper(dir: str) -> List[str]:
    """
    Returns a list of bedfiles within the directory.
    """
    return [
        file for file in os.listdir(dir)
        if os.path.isfile(f'{dir}/{file}')
        ]


class LocalContextParser:
    """Object that parses local genomic data into graph edges

    Args:
        bedfiles // dictionary containing each local genomic data    type as bedtool
            obj
        windows // bedtool object of windows +/- 250k of protein coding genes
        params // configuration vals from yaml 

    Methods
    ----------
    _make_directories:
        prepare necessary directories

    # Helpers
        ATTRIBUTES -- list of node attribute types
        DIRECT -- list of datatypes that only get direct overlaps, no slop
        FEAT_WINDOWS -- dictionary of each nodetype: overlap windows
        NODES -- list of nodetypes
        ONEHOT_NODETYPE -- dictionary of node type one-hot vectors 
    """

    ATTRIBUTES = [
        "cnv",
        "cpg",
        "ctcf",
        "dnase",
        "h3k27ac",
        "h3k27me3",
        "h3k36me3",
        "h3k4me1",
        "h3k4me3",
        "h3k9me3",
        "indels",
        "line",
        "ltr",
        "microsatellites",
        "mirnatargets",
        "phastcons",
        "polr2a",
        "polya",
        "rbpbindingsites",
        "recombination",
        "repg1b",
        "repg2",
        "reps1",
        "reps2",
        "reps3",
        "reps4",
        "rnarepeat",
        "simplerepeats",
        "sine",
        "snp",
    ]  # no gc; hardcoded in as initial

    NODES = [
        "chromatinloops",
        "cpgislands",
        "ctcfccre",
        "enhancers",
        "histones",
        "polyasites",
        "promoters",
        "superenhancers",
        "tads",
        "tfbindingclusters",
        "tss",
    ]  # no gencode; hardcoded in as initial file

    DIRECT = ['chromatinloops', 'tads']

    NODE_FEATS = ["start", "end", "size", "gc"] + ATTRIBUTES

    # var helpers - for CPU cores
    NODE_CORES=len(NODES)  # 12
    ATTRIBUTE_CORES=len(ATTRIBUTES)  # 28

    # dict helpers
    ONEHOT_NODETYPE = {
    }

    # Local context set at 2kb. Enhancers - can vary widely, so dependent on 3d
    # chromatin structure and from FENRIR network. Direct binding, polyasites
    # set to 500bp
    FEAT_WINDOWS = {
        'cpgislands': 2000,
        'ctcfccre': 2000,
        'enhancers': 2000,
        'gencode': 5000,
        'histones': 2000,
        'polyasites': 500,
        'promoters': 2000,
        'superenhancers': 2000,
        'tfbindingclusters': 2000,
        'tss': 2000,
    }

    # ONEHOT_EDGETYPE = {
    #     'local': [1,0,0,0,0],
    #     'enhancer-enhancer': [0,1,0,0,0],
    #     'enhancer-gene': [0,0,1,0,0],
    #     'circuits': [0,0,0,1,0],
    #     'ppi': [0,0,0,0,1],
    # }

    def __init__(self, bedfiles: List[str], params: Dict[str, Dict[str, str]]):
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

        # make directories
        self._make_directories()

    def _make_directories(self) -> None:
        """Directories for parsing genomic bedfiles into graph edges and nodes"""
        dir_check_make(self.parse_dir)

        for directory in [
            "edges/genes",
            "attributes",
            "intermediate/slopped",
            "intermediate/sorted",
        ]:
            dir_check_make(f"{self.parse_dir}/{directory}")

        for attribute in self.ATTRIBUTES:
            dir_check_make(f"{self.attribute_dir}/{attribute}")

    @time_decorator(print_args=True)
    def _region_specific_features_dict(
        self, bed: str
    ) -> List[Dict[str, pybedtools.bedtool.BedTool]]:
        """
        Creates a dict of local context datatypes and their bedtools objects.
        Renames features if necessary.
        """

        def rename_feat_chr_start(feature: str) -> str:
            """Add chr, start to feature name
            Cpgislands add prefix to feature names  # enhancers,
            Histones add an additional column
            """
            rename_strings = [
                "cpgislands",
                "enhancers",
                "histones",
                "polyasites",
                "tfbindingclusters",
            ]
            if prefix in rename_strings:
                feature = extend_fields(feature, 4)
                feature[3] = f"{prefix}_{feature[0]}_{feature[1]}"
            else:
                feature[3] = f"{feature[3]}_{feature[0]}_{feature[1]}"
            return feature

        # prepare data as pybedtools objects
        bed_dict = {}
        prefix = bed.split("_")[0]
        a = pybedtools.BedTool(
            f"{self.root_dir}/{self.tissue}/gene_regions_tpm_filtered.bed"
        )
        b = pybedtools.BedTool(f"{self.root_dir}/{self.tissue}/local/{bed}").sort()
        ab = b.intersect(a, sorted=True, u=True)
        if prefix == "enhancers":  # save enhancers early for attr ref
            b.each(rename_feat_chr_start).filter(lambda x: "alt" not in x[0]).saveas(
                f"{self.local_dir}/enhancers_lifted_{self.tissue}.bed_noalt"
            )

        # take specific windows and format each file
        if prefix in self.NODES and prefix != "gencode":
            result = ab.each(rename_feat_chr_start).cut([0, 1, 2, 3]).saveas()
            bed_dict[prefix] = pybedtools.BedTool(str(result), from_string=True)
        else:
            bed_dict[prefix] = ab.cut([0, 1, 2, 3])

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
        Each bed is slopped then intersected twice. First, it is intersected
        with every other node type. Then, the intersected bed is filtered to
        only keep edges within the gene region.

        Args:
            node_type // _description_
            all_files // _description_

        Raises:
            AssertionError: _description_
        """
        print(f'starting combinations {node_type}')

        def _unix_intersect(
            node_type: str,
            type: Optional[str]=None
            ) -> None:
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
                .each(_add_distance).saveas()\
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
        """For each node of a node_type get their overlap with gene windows then
        aggregate total nucleotides, gc content, and all other attributes

        Args:
            node_type // node datatype in self.NODES
        """
        if node_type == 'gencode':  # if gencode, create attr ref for all genes, as they might show up in interactions
            ref_file = f"{self.local_dir}/{self.shared_data['gencode']}"
            pybedtools.BedTool(ref_file).cut([0,1,2,3]).saveas(ref_file + '_cut')
            ref_file += '_cut'
        elif node_type == 'enhancers':  # ignore ALT chr
            ref_file = f"{self.local_dir}/enhancers_lifted_{self.tissue}.bed_noalt"
        else:
            ref_file = f'{self.parse_dir}/intermediate/sorted/{node_type}.bed'

        def add_size(feature: str) -> str:
            """
            """
            feature = extend_fields(feature, 5)
            feature[4] = feature.end - feature.start
            return feature

        def sum_gc(feature: str) -> str:
            """
            """
            feature[13] = int(feature[8]) + int(feature[9])
            return feature
        
        polyadenylation = self._polyadenylation_targets(
            f"{self.interaction_dir}" f"/{self.interaction_files['polyadenylation']}"
        )

    # @time_decorator(print_args=True)
    # def _polyadenylation_targets(
    #     self,
    #     interaction_file: str
    #     ) -> List[str]:
    #     """Genes which are listed as alternative polyadenylation targets"""
    #     with open(interaction_file, newline = '') as file:
    #         file_reader = csv.reader(file, delimiter='\t')
    #         next(file_reader)
    #         return [
    #             self.genesymbol_to_gencode[line[6]]
    #             for line in file_reader
    #             if line[6] in self.genesymbol_to_gencode.keys()
    #             ]

        for attribute in self.ATTRIBUTES:
            save_file = f'{self.attribute_dir}/{attribute}/{node_type}_{attribute}_percentage'
            print(f'{attribute} for {node_type}')
            if attribute == 'gc':
                pybedtools.BedTool(ref_file)\
                .each(add_size)\
                .nucleotide_content(fi=self.fasta)\
                .each(sum_gc)\
                .sort()\
                .groupby(g=[1,2,3,4], c=[5,14], o=['sum'])\
                .saveas(save_file)
            else:
                pybedtools.BedTool(ref_file)\
                .each(add_size)\
                .intersect(f'{self.parse_dir}/intermediate/sorted/{attribute}.bed', wao=True, sorted=True)\
                .groupby(g=[1,2,3,4], c=[5,10], o=['sum'])\
                .sort()\
                .saveas(save_file)

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
            'sort_cmd': [f"LC_ALL=C sort --parallel=48 -S 80% -k10,10 {self.parse_dir}/edges/all_concat.bed >", \
                f"{self.parse_dir}/edges/all_concat_sorted.bed"],
        }

        for cmd in cmds:
            _chk_file_and_run(
                cmds[cmd][1],
                cmds[cmd][0] + cmds[cmd][1],
            )

        sorted_beds = f"{self.parse_dir}/edges/all_concat_sorted.bed"
        awk_cmd = f"awk -F'\t' '{{print>\"{self.parse_dir}/edges/genes/\"$10}}' {self.parse_dir}/edges/all_concat_sorted.bed" 

        if os.path.isfile(sorted_beds) and os.path.getsize(sorted_beds) !=0:
            subprocess.run(awk_cmd, stdout=None, shell=True)

    @time_decorator(print_args=True)
    def _save_node_attributes(self, node: str) -> None:
        """
        Save attributes for all node entries. Used during graph construction for
        gene_nodes that fall outside of the gene window and for some gene_nodes
        from interaction data
        """
        attr_dict, set_dict = {}, {}  # dict[gene] = [chr, start, end, size, gc]
        for attribute in self.ATTRIBUTES:
            filename = f'{self.parse_dir}/attributes/{attribute}/{node}_{attribute}_percentage'
            with open(filename, 'r') as file:
                lines = [tuple(line.rstrip().split('\t')) for line in file]
                set_dict[attribute] = set(lines)
            empty_attr = 'placeholder'
        else:
            empty_attr = attribute

            if attribute == empty_attr:
                for line in set_dict['gc']:
                    attr_dict[line[3]][attribute] = 0
            else:
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

        # process windows and renaming 
        pool = Pool(processes=32)
        bedinstance = pool.map(self._region_specific_features_dict,\
            [bed for bed in self.bedfiles])
        pool.close()  # re-open and close pool after every multi-process

        # convert back to dictionary
        bedinstance = {key.casefold():value for element in bedinstance for key, value in element.items()}

        # sort and extend windows according to FEAT_WINDOWS
        bedinstance_sorted, bedinstance_slopped = self._slop_sort(bedinstance=bedinstance, chromfile=self.chromfile)

        # save a list of the nodes and their indexes
        self._save_feature_indexes(bedinstance_sorted)

        # save intermediate files
        _save_intermediate(bedinstance_sorted, folder='sorted')
        _save_intermediate(bedinstance_slopped, folder='slopped')

        # pre-concatenate to save time
        all_files = f'{self.parse_dir}/intermediate/sorted/all_files_concatenated.bed'
        _pre_concatenate_all_files(all_files)

        # perform intersects across all feature types - one process per nodetype
        pool = Pool(processes=self.NODE_CORES)
        pool.starmap(self._bed_intersect, zip(self.NODES, repeat(all_files)))
        pool.close()

        # get size and all attributes - one process per nodetype
        pool = Pool(processes=self.NODE_CORES)
        pool.map(self._aggregate_attributes, self.NODES)
        pool.close()

        # parse edges into individual files
        self._generate_edges()

        # save node attributes as reference for later - one process per nodetype
        pool = Pool(processes=self.NODE_CORES)
        pool.map(self._save_node_attributes, self.NODES)
        pool.close()


def main() -> None:
    """Pipeline to parse genomic data into edges"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, help="Path to .yaml file with filenames")

    args = parser.parse_args()
    params = parse_yaml(args.config)

    genes = f"{params['dirs']['root_dir']}/{params['resources']['tissue']}/gene_regions_tpm_filtered.bed"
    if not (
        os.path.exists(genes) and os.stat(genes).st_size > 0
    ):
        window = _tpm_filter_gene_windows(
            gencode=f"shared_data/local/{params['shared']['gencode']}",
            tissue=params["resources"]["tissue"],
            tpm_file=params["resources"]["tpm"],
            chromfile=params["resources"]["chromfile"],
            window=params["resources"]["window"],
            slop=True,
        )
        window.saveas(genes)

    bedfiles = _listdir_isfile_wrapper(
         dir=f"{params['dirs']['root_dir']}/{params['resources']['tissue']}/local",
    )

    # instantiate object
    localparseObject = LocalContextParser(
        bedfiles=bedfiles,
        genes=genes,
        params=params,
    )

    # run parallelized pipeline!
    localparseObject.parse_context_data()

    # cleanup temporary files
    pybedtools.cleanup(remove_all=True)


if __name__ == '__main__':
    main()

