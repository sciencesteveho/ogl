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
import subprocess

import numpy as np
import pandas as pd
import pybedtools

from itertools import repeat
from mygene import MyGeneInfo
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

from utils import dir_check_make, filtered_genes, parse_yaml, time_decorator


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
        'circuits': [0,0,0,1,0,0],
        'giant': [0,0,0,0,1,0],
        'ppi': [0,0,0,0,0,1],
    }

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
        genes: List[any]):
        """Initialize the class"""

        self.genes = genes

        self.gencode = params['shared']['gencode']
        self.interaction_files = params['interaction']
        self.shared_data = params['shared']
        self.tissue = params['resources']['tissue']
        self.tissue_name = params['resources']['tissue_name']
        self.ppi_tissue = params['resources']['ppi_tissue']
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
        self.e_indexes = self._enhancer_index(
            e_index=f"{self.shared_interaction_dir}/enhancer_indexes.txt",
            e_index_unlifted=f"{self.shared_interaction_dir}/enhancer_indexes_unlifted.txt"
        )

    def _genes_from_gencode(self, gencode_file: str) -> Dict[str, str]:
        """returns a dict of gencode v26 genes, their ids and associated gene symbols"""
        a = pybedtools.BedTool(gencode_file)
        return {
            line[9].split(';')[3].split('\"')[1]:line[3]
            for line in a 
            if line[0] not in ['chrX', 'chrY', 'chrM']
            }

    def _enhancer_index(
        self,
        e_index: str, 
        e_index_unlifted: str
        ) -> Dict[str, str]:
        """returns a dict to map enhancers from hg19 to hg38"""
        def text_to_dict(txt, idx1, idx2):
            with open(txt) as file:
                file_reader = csv.reader(file, delimiter='\t')
                return {
                    line[idx1]:line[idx2]
                    for line in file_reader
                }
        e_dict = text_to_dict(e_index, 1, 0)
        e_dict_unlifted = text_to_dict(e_index_unlifted, 0, 1)
        return {
            enhancer:e_dict[e_dict_unlifted[enhancer]]
            for enhancer in e_dict_unlifted
            if e_dict_unlifted[enhancer] in e_dict.keys()
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
        """Convert each enhancer-enhancer link to hg38 and return a formatted tuple"""
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            e_e_liftover = [
                (self.e_indexes[line[0]], self.e_indexes[line[1]])
                for line in file_reader
                if line[0] in self.e_indexes.keys()
                and line[1] in self.e_indexes.keys()
            ]
        return [
            (f"enhancers_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
            f"enhancers_{self._format_enhancer(line[1], 0)}_{self._format_enhancer(line[1], 1)}",
            -1,
            'enhancer-enhancer',)
            for line in e_e_liftover
        ]

    @time_decorator(print_args=True)
    def _fenrir_enhancer_gene(
        self,
        interaction_file: str,
        ) -> List[Tuple[str, str, float, str]]:
        """Convert each enhancer-gene link to hg38 and ensemble ID, return a formatted tuple"""
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            e_g_liftover = [
                (self.e_indexes[line[0]], self.genesymbol_to_gencode[line[2]])
                for line in file_reader
                if line[0] in self.e_indexes.keys()
                and line[2] in self.genesymbol_to_gencode.keys()
            ]
        return [
            (f"enhancers_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
            line[1],
            -1,
            'enhancer-gene')
            for line in e_g_liftover
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
            giant_symbols = _convert_genes(edges, symbol_ref, edge_type='giant')
            giant_filtered = [edge for edge in giant_symbols if edge[0] != 'NA' and edge[1] != 'NA']
            return _convert_genes(
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
        edges = list(
                zip(*map(t_spec_filtered.get, ['symbol1', 'symbol2']),
                repeat(-1),
                repeat('ppi')
                ))
        return [(
            self.genesymbol_to_gencode[edge[0]],
            self.genesymbol_to_gencode[edge[1]],
            edge[2],
            edge[3],)
            for edge in edges
            if edge[0] in self.genesymbol_to_gencode.keys() and edge[1] in self.genesymbol_to_gencode.keys()
            ]

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
    def _interaction_preprocess(self) -> List[str]:
        """Retrieve all interaction edges
        
        Returns:
            A list of all edges
            A list of alternative polyadenylation targets
        """
        all_interaction_file = f'{self.interaction_dir}/interaction_edges.txt' 
        if not (os.path.exists(all_interaction_file) and os.stat(all_interaction_file).st_size > 0):
            e_e_edges = self._fenrir_enhancer_enhancer(
                f"{self.interaction_dir}"
                f"/{self.tissue_specific['enhancers_e_e']}"
                )
            e_g_edges = self._fenrir_enhancer_gene(
                f"{self.interaction_dir}"
                f"/{self.tissue_specific['enhancers_e_g']}"
                )
            ppi_edges = self._iid_ppi(
                interaction_file=f"{self.interaction_dir}/{self.interaction_files['ppis']}",
                tissue=self.ppi_tissue
                )
            giant_edges = self._giant_network(
                f"{self.interaction_dir}"
                f"/{self.interaction_files['giant']}"
                )
            circuit_edges = self._marbach_regulatory_circuits(
                f"{self.interaction_dir}"
                f"/{self.interaction_files['circuits']}"
                )
            interaction_edges = e_e_edges + e_g_edges + ppi_edges + giant_edges + circuit_edges
            with open(all_interaction_file, 'w+') as output:
                writer = csv.writer(output, delimiter='\t')
                writer.writerows(interaction_edges)
        else:
            pass

        polyadenylation = self._polyadenylation_targets(
            f"{self.interaction_dir}"
            f"/{self.interaction_files['polyadenylation']}"
            )

        return polyadenylation
        
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
        reference_attrs: Dict[str, Dict[str, Any]],
        interaction_edges: List[Any],
        ) -> Any:
        """_lorem ipsum"""
        print(f'starting _prepare_graph_tensors on {gene}')

        def _reindex_nodes(edges):
            """_lorem"""
            uniq_nodes = sorted(
                set([edge[0] for edge in edges]+[edge[1] for edge in edges])
                )
            node_idxs = {node: id for id, node in enumerate(uniq_nodes)}
            edges_reindexed = list(
                map(lambda edge: [node_idxs[edge[0]], node_idxs[edge[1]], edge[2]], edges)
                )
            return sorted(edges_reindexed), node_idxs, len(uniq_nodes)

        gene_edges = f'{self.parse_dir}/edges/genes/{gene}'
        ### fast uniq_nodes 
        uniq_local_sort = f"awk '{{print $4 \"\\n\" $8}}' {gene_edges} \
            | sort -u"
        proc = subprocess.Popen(uniq_local_sort, shell=True, stdout=subprocess.PIPE)
        uniq_local = proc.communicate()[0]

        with open(f'{self.interaction_dir}/uniq_interaction_nodes.txt') as f:
            interaction_nodes = [line.rstrip('\n') for line in f.readlines()]

        nodes_to_add = set(str(uniq_local).split('\\n')).intersection(interaction_nodes)

        edges_to_add = [
            [line[0], line[1], line[3]] for line in
            filter(
                lambda interaction: interaction[0] in nodes_to_add or interaction[1] in nodes_to_add,
                interaction_edges
            )
        ]

        with open(gene_edges, newline='') as file:
            local_edges = [
                [line[3], line[7], 'local']
                for line in csv.reader(file, delimiter='\t')]

        edges_reindexed, node_idxs, num_nodes = _reindex_nodes(local_edges + edges_to_add)

        graph_only_refs = {node_idxs[node]:reference_attrs[node] for node in node_idxs}

        with open(f'{self.graph_dir}/{gene}_{self.tissue}', 'wb') as output:
            pickle.dump({
            'edge_index': np.array([[edge[0] for edge in edges_reindexed], [edge[1] for edge in edges_reindexed]]),
            'edge_feat': np.array([self.ONEHOT_EDGETYPE[edge[2]] for edge in edges_reindexed]),
            'node_feat': np.array([[float(x) for x in list(graph_only_refs[key].values())[2:]]
            for key in graph_only_refs.keys()]),
            'num_nodes': num_nodes,
            },
            output
            )
        print(f'Finished _prepare_graph_tensors on {gene}')

    @time_decorator(print_args=True)
    def generate_graphs(self) -> None:
        """Constructs graphs in parallel"""
        ### base reference
        gencode_ref = f'{self.parse_dir}/attributes/gencode_reference.pkl'

        ### retrieve interaction-based edges
        polyadenylation = self._interaction_preprocess()

        ### prepare nested dict for node features
        reference_attrs = self._prepare_reference_attributes(
            gencode_ref=gencode_ref,
            polyadenylation=polyadenylation,
        )

        genes_to_construct = [
            gene for gene in self.genes
            if not (os.path.exists(f'{self.graph_dir}/{gene}_{self.tissue}')
            and os.stat(f'{self.graph_dir}/{gene}_{self.tissue}').st_size > 0)
        ]

        interaction_file = f'{self.interaction_dir}/interaction_edges.txt'

        ### prepare list of uniq interaction edges
        cmd = f"awk '{{print $1 \"\\n\" $2}}' {interaction_file} \
            | sort -u \
            > {self.interaction_dir}/uniq_interaction_nodes.txt"

        subprocess.run(cmd, stdout=None, shell=True)

        ### read interaction file into a list
        with open(interaction_file, newline='') as file:
            interaction_edges = [line for line in csv.reader(file, delimiter='\t')]

        print(f'total graphs to construct - {len(self.genes)}')
        print(f'starting construction on {len(genes_to_construct)} genes')

        ### parse graph into tensors and save
        pool = Pool(processes=24)
        pool.starmap(
            self._prepare_graph_tensors,
            zip(genes_to_construct,
            repeat(reference_attrs),
            repeat(interaction_edges),
        ))
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

    genes = filtered_genes(f"{params['dirs']['root_dir']}/{params['resources']['tissue']}/gene_regions_tpm_filtered.bed")

    ### instantiate object
    graphconstructingObject = GraphConstructor(
        params=params,
        genes=genes,
        )

    ### run pipeline!
    graphconstructingObject.generate_graphs()


if __name__ == '__main__':
    main()