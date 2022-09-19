#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] PRIORITY ** Fix memory leak! 
# - [ ] Fix filepaths. They are super ugly! 
# - [ ] one-hot encode node_feat type?
# - [ ] scale feats... 

"""Create graphs from parsed genomic data"""

import argparse
import csv
import os
import pickle

import numpy as np
import pandas as pd
import pybedtools
import tensorflow as tf

from itertools import repeat
from mygene import MyGeneInfo
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from utils import dir_check_make, parse_yaml, time_decorator


class GraphConstructor:
    """Object to construct tensor based graphs from parsed bedfiles
    
    The following types of interaction data are represented:
        Enhancer-enhancer networks from FENRIR
        Enhancer-gene networks from FENRIR
        Gene(TF)-gene circuits from Marbach et al.,
        Curated protein-protein interactions from the Integrated Interactions Database V 2021-05
        Gold-standard (C1) gene-gene interactions from GIANT/HumanBase
        Alternative polyadenylation targets from APAatlas

    Args:
        params // configuration vals from yaml 

    Methods
    ----------
    _gene_symbol_to_gencode_ref:
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
        ATTRIBUTES --
        HISTONE_IDXS --
        NODES --
        NODE_FEATS --
        ONEHOT_EDGETYPE --
    """

    ATTRIBUTES = ['cpg', 'ctcf', 'dnase', 'enh', 'enhbiv', 'enhg', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'het', 'line', 'ltr', 'microsatellites', 'phastcons', 'polr2a', 'reprpc', 'rnarepeat', 'simplerepeats', 'sine', 'tssa', 'tssaflnk', 'tssbiv', 'txflnk', 'tx', 'txwk', 'znf']  # no gc; hardcoded in as initial file
    NODES = ['chromatinloops', 'cpgislands', 'enhancers', 'histones', 'mirnatargets', 'polyasites', 'promoters', 'rbpbindingsites', 'tads', 'tfbindingclusters', 'tss']  # no gencode; hardcoded in as initial file 
    NODE_FEATS = ['start', 'end', 'size', 'gc'] + ATTRIBUTES

    ONEHOT_EDGETYPE = {
        'local': [1,0,0,0,0,0],
        'enhancer-enhancer': [0,1,0,0,0,0],
        'enhancer-gene': [0,0,1,0,0,0],
        'circuit': [0,0,0,1,0,0],
        'giant': [0,0,0,0,1,0],
        'ppi': [0,0,0,0,0,1],
    }

    def __init__(
        self,
        params: Dict[str, Dict[str, str]]):
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

        self.parse_dir = f"{self.root_dir}/{self.tissue}/parsing"
        self.graph_dir = f"{self.parse_dir}/graphs"
        self.interaction_dir = f"{self.root_dir}/{self.tissue}/interaction"
        self.shared_interaction_dir = f'{self.shared_dir}/interaction'

        dir_check_make(self.graph_dir)
        self.genesymbol_to_gencode = self._genes_from_gencode(
            gencode_file=f"{self.shared_interaction_dir}/{self.interaction_files['gencode']}"
            )

    def _genes_from_gencode(self, gencode_file: str) -> Dict[str, str]:
        """returns a dict of gencode v26 genes, their ids and associated gene symbols"""
        a = pybedtools.BedTool(gencode_file)
        return {
            line[9].split(';')[3].split('\"')[1]:line[3]
            for line in a
            }

    def _format_enhancer(
        self,
        input: str,
        index: int,
        ) -> str:
        return f"{input.replace(':', '-').split('-')[index]}"

    @time_decorator(print_args=True)
    def _fenrir_enhancer_enhancer(
        self,
        interaction_file: str,
        ) -> List[Tuple[str, str, float, str]]:
        """lorem"""
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            return [
                (f"enhancers_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
                f"enhancers_{self._format_enhancer(line[1], 0)}_{self._format_enhancer(line[1], 1)}",
                -1,
                'enhancer-enhancer',)
                for line in file_reader
            ]

    @time_decorator(print_args=True)
    def _fenrir_enhancer_gene(
        self,
        interaction_file: str,
        ) -> List[Tuple[str, str, float, str]]:
        """lorem"""
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            return [
                (f"enhancers_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
                self.genesymbol_to_gencode[line[2]],
                -1,
                'enhancer-gene')
                for line in file_reader
                if line[2] in self.genesymbol_to_gencode.keys()
            ]

    @time_decorator(print_args=True)
    def _giant_network(
        self,
        interaction_file: str,
        ) -> List[Tuple[str, str, float, str]]:
        """Lorem"""
        mg = MyGeneInfo()

        def _read_giant(graph):
            edges, edge1, edge2 = [], [], []
            with open(graph, newline='') as file:
                lines = csv.reader(file, delimiter='\t')
                for line in lines:
                    if line[2] == '1':
                        edges.append(line)
                        edge1.append(line[0])
                        edge2.append(line[1])
            return edges, set(edge1+edge2)

        def _entrez_to_symbol_ref(edge_list):
            edge_lookup = {}
            for edge in edge_list:
                meta = mg.query(edge, fields=['symbol'], species='human', verbose=False)
                try:
                    result = meta['hits'][0]
                    if 'symbol' not in result:
                        edge_lookup[edge] = 'NA'
                    else:
                        edge_lookup[edge] = result['symbol']
                except IndexError:
                    edge_lookup[edge] = 'NA'
            return edge_lookup

        def _convert_giant(edges, symbol_ref, ensembl_ref):
            def _convert_genes(
                edges: List[Tuple[any]],
                ref: Dict[str, str],
                edge_type: str,
                ) -> List[Tuple[any]]:
                return [
                    (ref[edge[0]],
                    ref[edge[1]],
                    -1,
                    edge_type,)
                    for edge in edges
                    if edge[0] in ref.keys()
                    and edge[1] in ref.keys()
                ]
            giant_symbols = self._convert_genes(edges, symbol_ref)
            giant_filtered = [edge for edge in giant_symbols if edge[0] != 'NA' and edge[1] != 'NA']
            return self._convert_genes(
                giant_filtered,
                ensembl_ref,
                'giant',)
        
        edges, edge_list = _read_giant(interaction_file)
        symbol_ref = _entrez_to_symbol_ref(edge_list)
        return _convert_giant(
            edges,
            symbol_ref,
            self.genesymbol_to_gencode
            )

    @time_decorator(print_args=True)
    def _iid_ppi(
        self,
        interaction_file: str,
        tissue: str,
        ) -> List[Tuple[str, str, float, str]]:
        """Protein-protein interactions from the Integrated Interactions Database v 2021-05.
        Interactions"""
        df = pd.read_csv(interaction_file, delimiter='\t')
        df = df[['symbol1', 'symbol2', 'evidence_type', tissue]]
        t_spec_filtered = df[(df[tissue] > 0) & (df['evidence_type'].str.contains('exp'))]
        return list(
            zip(self.genesymbol_to_gencode[t_spec_filtered['symbol1']],
            self.genesymbol_to_gencode[t_spec_filtered['symbol2']], 
            repeat(-1),
            repeat('ppi'))
            )

    @time_decorator(print_args=True)
    def _marbach_regulatory_circuits(
        self,
        interaction_file: str
        ) -> List[Tuple[str, str, float, str]]:
        """Regulatory circuits from Marbach et al., Nature Methods, 2016"""
        with open(interaction_file, newline = '') as file:
            return [
                (self.genesymbol_to_gencode[line[0]], self.genesymbol_to_gencode[line[1]], line[2], 'circuits')
                for line in csv.reader(file, delimiter='\t')
                if line[0] in self.genesymbol_to_gencode.keys() and line[1] in self.genesymbol_to_gencode.keys()
                ]

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
                self.genesymbol_to_gencode[line[6]]
                for line in file_reader
                if line[6] in self.genesymbol_to_gencode.keys()
                ]

    @time_decorator(print_args=True)
    def _interaction_preprocess(self) -> Tuple[List[Any], List[str]]:
        """Retrieve all interaction edges
        
        Returns:
            A list of all edges
            A list of alternative polyadenylation targets
        """
        e_e_edges = self._fenrir_enhancer_enhancer(
            f"{self.interaction_dir}"
            f"/{self.tissue_specific['enhancers_e_e']}"
            )
        e_g_edges = self._fenrir_enhancer_gene(
            f"{self.interaction_dir}"
            f"/{self.tissue_specific['enhancers_e_g']}"
            )
        ppi_edges = self._iid_ppi(
            f"{self.interaction_dir}"
            f"/{self.interaction_files['ppis']}"
            )
        giant_edges = self._giant_network(
            f"{self.interaction_dir}"
            f"{self.interaction_files['giant']}"
            )
        circuit_edges = self._marbach_regulatory_circuits(
            f"{self.interaction_dir}"
            f"/{self.interaction_files['circuits']}"
            )
        polyadenylation = self._polyadenylation_targets(
            f"{self.interaction_dir}"
            f"/{self.interaction_files['polyadenylation']}"
            )
        return e_e_edges + e_g_edges + ppi_edges + giant_edges + circuit_edges, polyadenylation

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
        print(f'starting _prepare_graph_tensors on {gene}')
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
            df.sort_values('node_1', ignore_index=True)
            return df, len(node_idxs), node_idxs

        def _node_attributes(reference_attrs, node_idxs):
            """_lorem ipsum"""
            attribute_df = pd.DataFrame.from_dict({node:reference_attrs[node] for node in node_idxs}, orient='index', columns=['start', 'end', 'size', 'gc', 'cpg', 'ctcf', 'dnase', 'microsatellites', 'phastcons', 'polr2a', 'simplerepeats', 'polyadenylation', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3'])

            ### set index to be a column
            attribute_df.reset_index(inplace=True)
            attribute_df = attribute_df.rename(columns={'index': 'node'})
            
            ### add polyadenylation 
            attribute_df['polyadenylation'] = attribute_df['node'].apply(lambda x: 1 if x in polyadenylation else 0)

            ### add histones
            for histone in self.HISTONE_IDXS:
                attribute_df[histone] = attribute_df['node'].apply(
                    lambda x: x.split(',')[self.HISTONE_IDXS[histone]] if 'histone' in x else 0
                ) 

            attribute_df = attribute_df.fillna(0)
            attribute_df['node'] = attribute_df['node'].apply(lambda node: node_idxs[node])
            attribute_df = attribute_df.sort_values('node', ignore_index=True)
            return tf.convert_to_tensor(attribute_df[self.NODE_FEATS].astype('float32'))

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
            print(f'{gene} already done. Moving to next gene')
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

            with open(f'{self.graph_dir}/{gene}_{self.tissue}', 'wb') as output:
                pickle.dump({
                'edge_index': _get_edge_index(all_edges),
                'edge_feat': _get_edge_features(all_edges, weight=False),
                'node_feat': _node_attributes(reference_attrs=reference_attrs, node_idxs=node_idxs),
                'num_nodes': num_nodes,
                },
                output
                )
            print(f'Finished _prepare_graph_tensors on {gene}')

            del edges
            del all_edges


    @time_decorator(print_args=True)
    def generate_graphs(self) -> None:
        """Constructs graphs in parallel"""
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

        genes_to_construct = [
            gene for gene in genes
            if not (os.path.exists(f'{self.graph_dir}/{gene}_{self.tissue}')
            and os.stat(f'{self.graph_dir}/{gene}_{self.tissue}').st_size > 0)
        ]

        ### parse graph into tensors and save
        pool = Pool(processes=16)
        pool.starmap(
            self._prepare_graph_tensors,
            zip(genes_to_construct,
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