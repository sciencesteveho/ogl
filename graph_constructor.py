#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Fix filepaths. They are super ugly! 
# - [ ] one-hot encode node_feattype
# - [ ] scale feats... 

"""Create graphs from parsed genomic data"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import pybedtools
import pandas as pd
import tensorflow as tf

from itertools import repeat
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from utils import bool_check_attributes, dir_check_make, parse_yaml, time_decorator


class GraphConstructor:
    """Object to construct tensor based graphs from parsed befiles

    Args:
        params // configuration vals from yaml 

    Methods
    ----------
    _gene_symbol_to_gencode_ref:
        Lorem
    _co_expressed_pairs:
        Lorem
    _shared_eqtls:
        Lorem
    _gene_enhancer_atlas_links:
        Lorem
    _tissuenet_ppis:
        Lorem
    _marbach_regulatory_circuits:
        Lorem
    _polyadenylation_targets:
        Lorem
    _interaction_data_preprocess:
        Lorem
    _prepare_reference_attributes:
        Lorem
    _prepare_graph_tensors:
        Parses graph data into tensors representing edge and note feats and indexes
    generate graphs:
        Lorem

    # Helpers
        NODE_FEATS --
        ATTRIBUTES --
        NODES --
        ONEHOT_EDGETYPE --
    """

    NODE_FEATS = ['start', 'end', 'size', 'gc', 'cpg', 'ctcf', 'dnase', 'microsatellites', 'phastcons', 'polr2a', 'simplerepeats', 'polyadenylation', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
    ATTRIBUTES = ['cpg', 'ctcf', 'dnase', 'microsatellites', 'phastcons', 'polr2a', 'simplerepeats']  # no gc; hardcoded in as initial file
    NODES = ['chromatinloops', 'chromhmm', 'cpgislands', 'histones', 'regulatorybuild', 'repeatmasker', 'tads']  # no gencode; hardcoded in as initial file 

    ONEHOT_EDGETYPE = {
        'local': [1,0,0,0,0,0],
        'co-exp': [0,1,0,0,0,0],
        'eqtl': [0,0,1,0,0,0],
        'ppi': [0,0,0,1,0,0],
        'circuits': [0,0,0,0,1,0],
        'enhancer_atlas': [0,0,0,0,0,1],
    }

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
        ) -> None:
        """Initialize the class"""

        self.gencode = params['shared']['gencode']
        self.locop_tissue = params['resources']['locop_tissue']
        self.interaction_files = params['interaction']
        self.shared_data = params['shared']
        self.tissue = params['resources']['tissue']
        self.tissue_name = params['resources']['tissue_name']
        self.tissue_specific = params['tissue_specific']

        self.root_dir = params['dirs']['root_dir']
        self.shared_dir = params['dirs']['shared_dir']
        self.interaction_dir = f'{self.shared_dir}/interaction'
        self.parse_dir = f"{self.root_dir}/{self.tissue}/parsing"
        self.graph_dir = f"{self.parse_dir}/graphs"

        self.parsed_features = {
            'cpg': self.tissue_specific['cpg'],
            'ctcf': self.tissue_specific['ctcf'],
            'dnase': self.tissue_specific['dnase'],
            'microsatellites': self.shared_data['microsatellites'],
            'phastcons': self.shared_data['phastcons'],
            'polr2a': self.tissue_specific['polr2a'],
            'simplerepeats': self.shared_data['simplerepeats'],
        }

        dir_check_make(self.graph_dir)
        self.gencode_to_genesymbol, self.ensembl_to_gencode, self.gencode_no_transcript = self._gene_symbol_to_gencode_ref(
            id_file=f"{self.interaction_dir}/{self.interaction_files['id_lookup']}",
            gencode_file=f"{self.interaction_dir}/{self.interaction_files['gencode']}",
            )

    @time_decorator(print_args=True)
    def _gene_symbol_to_gencode_ref(
        self,
        id_file: str,
        gencode_file: str
        ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """_lorem ipsum"""
        gencode_to_genesymbol, ensembl_to_gencode = {}, {}
        gencode_genes = [line[3] for line in pybedtools.BedTool(gencode_file)]

        with open(id_file, newline='') as file:
            for line in csv.reader(file, delimiter='\t'):
                gencode_to_genesymbol[line[1]] = line[0]
                ensembl_to_gencode[line[0].split(".")[0]] = line[0]

        return gencode_to_genesymbol, ensembl_to_gencode, {
            node.split(".")[0]:node
            for node in gencode_genes
            if 'PAR_Y' not in node
            }

    @time_decorator(print_args=True)
    def _co_expressed_pairs(
        self,
        interaction_file: str
        ) -> List[Tuple[str, str, float, str]]:
        """Co-expressed node pairs from Ribeiro et al., 2020"""
        with open(interaction_file, newline = '') as file:
            return [
                (self.gencode_no_transcript[line[0]], self.gencode_no_transcript[line[2]], float(line[5]), 'co-exp')
                for line in csv.reader(file, delimiter='\t')
                if line[4] == self.locop_tissue
                ]

    @time_decorator(print_args=True)
    def _shared_eqtls(
        self,
        interaction_file: str
        ) -> List[Tuple[str, str, float, str]]:
        """Shared eQTLs from Ribeiro et al., 2020"""
        with open(interaction_file, newline = '') as file:
            return [
                (self.gencode_no_transcript[line[2]], self.gencode_no_transcript[line[4]], float(line[5]), 'eqtl')
                for line in csv.reader(file, delimiter='\t')
                if line[6] == self.locop_tissue
                ]

    @time_decorator(print_args=True)
    def _tissuenet_ppis(
        self,
        interaction_file: str
        ) -> List[Tuple[str, str, int, str]]:
        """
        Protein-protein interactions from Tissuenet V.2
        There are no scores, so weights are placeholder values of -1
        """
        with open(interaction_file, newline = '') as file:
            return [
                (self.ensembl_to_gencode[line[0]], self.ensembl_to_gencode[line[1]], -1, 'ppi')
                for line in csv.reader(file, delimiter='\t')
                if line[0] in self.ensembl_to_gencode.keys() and line[1] in self.ensembl_to_gencode.keys()
                ]

    @time_decorator(print_args=True)
    def _marbach_regulatory_circuits(
        self,
        interaction_file: str
        ) -> List[Tuple[str, str, float, str]]:
        """Regulatory circuits from Marbach et al., Nature Methods, 2016"""
        with open(interaction_file, newline = '') as file:
            return [
                (self.gencode_to_genesymbol[line[0]], self.gencode_to_genesymbol[line[1]], line[2], 'circuits')
                for line in csv.reader(file, delimiter='\t')
                if line[0] in self.gencode_to_genesymbol.keys() and line[1] in self.gencode_to_genesymbol.keys()
                ]

    @time_decorator(print_args=True)
    def _gene_enhancer_atlas_links(
        self,
        interaction_file: str
        ) -> List[Tuple[str, str, float, str]]:
        """Enhancer node links from enhancer atlas"""
        enhancers = []
        with open(interaction_file, newline = '') as file:
            for line in csv.reader(file, delimiter='\t'):
                if line[3] in self.gencode_to_genesymbol.keys():
                    enhancers.append(
                        (f"enhancer_ATLAS_{line[0]}_{line[1]}",
                        self.gencode_to_genesymbol[line[3]],
                        line[7],
                        'enhancer_atlas'
                        ))
                elif line[2] in self.ensembl_to_gencode.keys():
                    enhancers.append(
                        (f"enhancer_ATLAS_{line[0]}_{line[1]}",
                        self.ensembl_to_gencode[line[2]],
                        line[7],
                        'enhancer_atlas'
                        ))
                else:
                    pass
        return enhancers

    @time_decorator(print_args=True)
    def _polyadenylation_targets(
        self,
        interaction_file: str
        ) -> List[str]:
        """Genes which are listed as alternative polyadenylation targets"""
        with open(interaction_file, newline = '') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            return [
                self.gencode_to_genesymbol[line[6]]
                for line in file_reader
                if line[6] in self.gencode_to_genesymbol.keys()
                ]

    @time_decorator(print_args=True)
    def _interaction_preprocess(self) -> Tuple[List[Any], List[str]]:
        """Retrieve all interaction edges
        
        Returns:
            A list of all edges
            A list of alternative polyadenylation targets
        """
        cop_edges = self._co_expressed_pairs(
            f"{self.interaction_dir}"
            f"/{self.interaction_files['cops']}"
            )
        shared_qtl_edges = self._shared_eqtls(
            f"{self.interaction_dir}"
            f"/{self.interaction_files['shared_eqtls']}"
            )
        ppi_edges = self._tissuenet_ppis(
            f"{self.interaction_dir}"
            f"/Humanproteinatlas-Protein/{self.interaction_files['ppis']}"
            )
        circuit_edges = self._marbach_regulatory_circuits(
            f"{self.interaction_dir}"
            "/FANTOM5_individual_networks"
            "/394_individual_networks"
            f"/{self.interaction_files['circuits']}"
            )
        enhancer_edges = self._gene_enhancer_atlas_links(
            f"{self.root_dir}"
            f"/{self.tissue}"
            "/interaction"
            f"/{self.tissue_specific['enhancers']}.interaction"
            )
        polyadenylation = self._polyadenylation_targets(
            f"{self.interaction_dir}"
            "/PDUI_polyA_sites"
            f"/{self.interaction_files['polyadenylation']}"
            )
        return cop_edges + shared_qtl_edges + enhancer_edges + ppi_edges + circuit_edges, polyadenylation

    @time_decorator(print_args=False)
    def _prepare_reference_attributes(
        self,
        gencode_ref: str, 
        polyadenylation: List[str],
        ) -> Dict[str, Dict[str, Any]]:
        """Add polyadenylation to gencode ref dict used to fill """
        ref = pickle.load(open(f'{gencode_ref}', 'rb'))
        for gene in ref:
            if gene in polyadenylation:
                ref[gene]['polyadenylation'] = 1
            else:
                pass

        for node in self.NODES:
            ref_for_concat = pickle.load(
                open(f'{self.parse_dir}/attributes/{node}_reference.pkl', 'rb')
            )
            ref.update(ref_for_concat)
        return ref

    @time_decorator(print_args=False)
    def _prepare_graph_tensors(
        self,
        gene: str,
        interaction_edges: List[Any],
        reference_attrs: Dict[str, Dict[str, Any]],
        polyadenylation: Any,
        ) -> Any:
        """_lorem ipsum"""

        def _uniq_nodes_from_df(df: pd.DataFrame) -> np.ndarray:
            all_nodes = pd.Series(df[['node_1', 'node_2']].values.ravel())
            return sorted(all_nodes.unique())

        def _add_interactions(
            df: pd.DataFrame,
            node_list: List[str],
            interaction_edges: List[Any],
            ) -> pd.DataFrame:
            """Add interactions edges to local edges dataframe"""
            for tup in interaction_edges:
                if tup[0] in node_list or tup[1] in node_list:
                    df.loc[len(df.index)] = [
                        np.nan,  # start_1
                        np.nan,  # end_1
                        tup[0],  # node_1
                        np.nan,  # start_2
                        np.nan,  # end_2
                        tup[1],  # node_2
                        tup[2],  # weight
                        np.nan,  # gene
                        tup[3],  # edge_type
                    ]
            return df

        def _reindex_nodes(df):
            """_lorem"""
            uniq_nodes = _uniq_nodes_from_df(df)
            node_idxs = {node: id for id, node in enumerate(uniq_nodes)}
            for nodes in ['node_1', 'node_2']:
                df[nodes] = df[nodes].apply(lambda name: node_idxs[name])
            return df, len(node_idxs), node_idxs

        def _node_attributes(reference_attrs, node_idxs):
            """_lorem ipsum"""
            attribute_df = pd.read_csv(
                f'{self.parse_dir}/attributes/gc/genes/{gene}',
                sep='\t',
                header=None,
                usecols=[1,2,3,5,6],
                names=['start', 'end', 'node', 'size', 'gc']
            )
            
            ### add empty type, reorder so type is first followed by node
            attribute_df = attribute_df[['node', 'start', 'end', 'size', 'gc']]

            for attr in self.ATTRIBUTES:
                if bool_check_attributes(attr, self.parsed_features[attr]):
                    attr_df = pd.read_csv(
                        f'{self.parse_dir}/attributes/{attr}/genes/{gene}',
                        sep='\t',
                        header=None,
                        usecols=[6],
                        names=[f'{attr}']
                    )
                    attribute_df[f'{attr}'] = attr_df[f'{attr}']
                    del attr_df
                
            for feat in ['polyadenylation', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3',]:
                attribute_df[feat] = 0

            for _, row in attribute_df.iterrows():
                if row['node'] in polyadenylation:
                    row['polyadenylation'] = 1
                if 'histone' in row['node']:
                    row['H3K27ac'] = row['node'].split(',')[1], 
                    row['H3K27me3'] = row['node'].split(',')[2], 
                    row['H3K36me3'] = row['node'].split(',')[3], 
                    row['H3K4me1'] = row['node'].split(',')[4], 
                    row['H3K4me3'] = row['node'].split(',')[5], 
                    row['H3K9ac'] = row['node'].split(',')[6], 
                    row['H3K9me3'] = row['node'].split(',')[7],

            nodes_to_add = list(
                set(
                    [gene for gene in node_idxs
                    if gene not in list(attribute_df['node'])
                ]))
            
            df_nodes_to_add = pd.DataFrame(
                columns=[
                    'node',
                    'start',
                    'end',
                    'size',
                    'gc',
                    'cpg',
                    'ctcf',
                    'dnase', 
                    'microsatellites', 
                    'phastcons',
                    'polr2a',
                    'simplerepeats',
                    'polyadenylation',
                    'H3K27ac',
                    'H3K27me3',
                    'H3K36me3',
                    'H3K4me1',
                    'H3K4me3',
                    'H3K9ac',
                    'H3K9me3',
                ])

            for node in nodes_to_add:
                if 'histone' in node:
                    df_nodes_to_add.loc[len(df_nodes_to_add.index)] = [
                        node,
                        reference_attrs[node]['start'], 
                        reference_attrs[node]['end'], 
                        reference_attrs[node]['size'], 
                        reference_attrs[node]['gc'], 
                        reference_attrs[node]['cpg'], 
                        reference_attrs[node]['ctcf'], 
                        reference_attrs[node]['dnase'], 
                        reference_attrs[node]['microsatellites'], 
                        reference_attrs[node]['phastcons'], 
                        reference_attrs[node]['polr2a'], 
                        reference_attrs[node]['simplerepeats'],
                        reference_attrs[node]['polyadenylation'],
                        node.split(',')[1], 
                        node.split(',')[2], 
                        node.split(',')[3], 
                        node.split(',')[4], 
                        node.split(',')[5], 
                        node.split(',')[6], 
                        node.split(',')[7],
                    ]
                else:
                    df_nodes_to_add.loc[len(df_nodes_to_add.index)] = [
                        node,
                        reference_attrs[node]['start'], 
                        reference_attrs[node]['end'], 
                        reference_attrs[node]['size'], 
                        reference_attrs[node]['gc'], 
                        reference_attrs[node]['cpg'], 
                        reference_attrs[node]['ctcf'], 
                        reference_attrs[node]['dnase'], 
                        reference_attrs[node]['microsatellites'], 
                        reference_attrs[node]['phastcons'], 
                        reference_attrs[node]['polr2a'], 
                        reference_attrs[node]['simplerepeats'],
                        reference_attrs[node]['polyadenylation'],
                        0, 
                        0, 
                        0, 
                        0, 
                        0, 
                        0, 
                        0,
                    ]

            all_attr_df = pd.concat([attribute_df, df_nodes_to_add], ignore_index=True, axis=0)
            try:
                all_attr_df['node'] = all_attr_df['node'].apply(lambda node: node_idxs[node])
            except KeyError:
                print(f'Keyerror! Offending gene is {gene}')
                sys.exit(1)
            all_attr_df = all_attr_df.sort_values('node', ignore_index=True)
            return tf.convert_to_tensor(all_attr_df[self.NODE_FEATS].astype('float32'))

        def _get_edge_index(df):
            """_lorem"""
            edges_1 = tf.convert_to_tensor(df['node_1'])
            edges_2 = tf.convert_to_tensor(df['node_2'])
            return tf.convert_to_tensor([edges_1, edges_2])

        def _get_edge_features(df, weight=False):
            """
            First 6 values are one-hot encoding for the edge_datatype
            7th value is weight, which is optional
            """
            df['type'] = df['type'].apply(lambda edge_type: self.ONEHOT_EDGETYPE[edge_type])
            if weight == True:
                df['weight'] = df['weight'].apply(lambda x: [x])
                df['type'] =  np.array(df['type'] + df['weight'])
                return tf.convert_to_tensor([np.array(x).astype('float32') for x in df['type']])
            else:
                return tf.convert_to_tensor([np.array(x) for x in df['type']])

        ### only prepare tensors if file does not exist
        if (os.path.exists(f'{self.graph_dir}/{gene}') and os.stat(f'{self.graph_dir}/{gene}').st_size != 0):
            pass
        else:
            ### open parsed edge file
            edges = pd.read_csv(
                f'{self.parse_dir}/edges/genes/{gene}',
                sep='\t',
                header=None,
                usecols=[1,2,3,5,6,7,8,9],
                names=[
                    'start_1', 
                    'end_1', 
                    'node_1', 
                    'start_2', 
                    'end_2', 
                    'node_2', 
                    'weight', 
                    'gene'
                ],
            )

            edges['weight'] = 0  # temporary fix, remove if good way to normalize weights
            edges['type'] = 'local'  # set local edgetype

            ### get shared nodes between local and interaction edges
            uniq_local_nodes = _uniq_nodes_from_df(edges)
            uniq_interact_nodes = list(
                set([tup[0] for tup in interaction_edges] + [tup[1] for tup in interaction_edges])
            )

            common_nodes = list(
                set(uniq_local_nodes) & set(uniq_interact_nodes)
            )

            all_edges = _add_interactions(
                df=edges,
                node_list=common_nodes,
                interaction_edges=interaction_edges,
            )

            _, num_nodes, node_idxs = _reindex_nodes(all_edges)

            output = open(f'{self.graph_dir}/{gene}', 'wb')
            try:
                pickle.dump({
                'edge_index': _get_edge_index(all_edges),
                'edge_feat': _get_edge_features(all_edges, weight=False),
                'node_feat': _node_attributes(reference_attrs=reference_attrs, node_idxs=node_idxs),
                'num_nodes': num_nodes,
                },
                output
                )
            finally:
                output.close()


    @time_decorator(print_args=True)
    def generate_graphs(self) -> None:
        """_summary_

        Args:
            a // _description_
            b // _description_

        Raises:
            AssertionError: _description_
        
        Returns:
            c -- _description_
        """
        ### base reference
        gencode_ref = f'{self.root_dir}/{self.tissue}/parsing/attributes/gencode_reference.pkl'

        ### retrieve interaction-based edges
        interaction_edges, polyadenylation = self._interaction_preprocess()

        ### prepare nested dict for node features
        reference_attrs = self._prepare_reference_attributes(
            gencode_ref=gencode_ref,
            polyadenylation=polyadenylation,
        )

        ### get list of all gencode V26 genes
        genes = [
            key for key
            in pickle.load(open(f'{gencode_ref}', 'rb')).keys()
        ]

        ### parse graph into tensors and save
        pool = Pool(processes=32)
        pool.starmap(
            self._prepare_graph_tensors,
            zip(genes,
            repeat(interaction_edges),
            repeat(reference_attrs),
            repeat(polyadenylation))
        )
        pool.close()


def main() -> None:
    """Pipeline to generate individual graphs"""
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

    ### instantiate object
    graphconstructingObject = GraphConstructor(params=params)

    ### run pipeline!
    graphconstructingObject.generate_graphs()


if __name__ == '__main__':
    main()