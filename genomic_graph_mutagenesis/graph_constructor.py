#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] PRIORITY ** Fix memory leak! 
# - [ ] Fix filepaths. They are super ugly! 
# - [ ] one-hot encode node_feat type?
#

"""Create base graph structure from interaction-type omics data"""

import argparse
import csv
from itertools import repeat
from multiprocessing import Pool
import os
import pickle
import subprocess
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pybedtools

from utils import _tpm_filter_gene_windows, dir_check_make, parse_yaml, time_decorator


class GraphConstructor:
    """Object to construct tensor based graphs from parsed bedfiles
    
    The baseline graph structure is build from the following in order:
        Curated protein-protein interactions from the integrated interactions
        database V 2021-05
        TF-gene circuits from Marbach et al.
        TF-gene interactions from TFMarker. We keep "TF" and "I Marker" type relationships
        Enhancer-gene networks from FENRIR
        Enhancer-enhancer networks from FENRIR

        Alternative polyadenylation targets from APAatlas

    Args:
        params: configuration vals from yaml 
        genes:
        graph_type: {'local', 'concatenated'}

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
        Parses graph data into tensors representing edge and note feats and
        indexes
    generate graphs:
        Lorem

    # Helpers
        ATTRIBUTES --
        HISTONE_IDXS --
        NODES --
        NODE_FEATS --
        ONEHOT_EDGETYPE --
    """

    # ATTRIBUTES = ['cpg', 'ctcf', 'dnase', 'enh', 'enhbiv', 'enhg', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'het', 'line', 'ltr', 'microsatellites', 'phastcons', 'polr2a', 'reprpc', 'rnarepeat', 'simplerepeats', 'sine', 'tssa', 'tssaflnk', 'tssbiv', 'txflnk', 'tx', 'txwk', 'znf']  # no gc; hardcoded in as initial file
    ATTRIBUTES = ['cpg', 'ctcf', 'dnase', 'h3k27ac', 'h3k27me3', 'h3k36me3', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'line', 'ltr', 'microsatellites', 'phastcons', 'polr2a', 'rnarepeat', 'simplerepeats', 'sine']  # no gc; hardcoded in as initial file
    NODES = ['chromatinloops', 'cpgislands', 'enhancers', 'histones', 'mirnatargets', 'polyasites', 'promoters', 'rbpbindingsites', 'tads', 'tfbindingclusters', 'tss']  # no gencode; hardcoded in as initial file 
    NODE_FEATS = ['start', 'end', 'size', 'gc'] + ATTRIBUTES

    ONEHOT_EDGETYPE = {
        'local': [1,0,0,0,0],
        'enhancer-enhancer': [0,1,0,0,0],
        'enhancer-gene': [0,0,1,0,0],
        'circuits': [0,0,0,1,0],
        'ppi': [0,0,0,0,1],
    }

    def __init__(
        self,
        params: Dict[str, Dict[str, str]],
        genes: List[any],
        graph_type: str,
        ):
        """Initialize the class"""

        self.genes = genes
        self.graph_type = graph_type

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
        self.interaction_dir = f"{self.root_dir}/{self.tissue}/interaction"
        self.shared_interaction_dir = f'{self.shared_dir}/interaction'

        if self.graph_type == 'local':
            self.graph_dir = f"{self.parse_dir}/local_graphs"
        else:
            self.graph_dir = f"{self.parse_dir}/graphs"

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
    
    def _base_graph(
        self,
        edges: List[str]
        ):
        """Create a graph from list of edges"""
        G = nx.Graph()
        G.add_edges_from((tup[0], tup[1]) for tup in edges)
        return G 

    @time_decorator(print_args=True)
    def _iid_ppi(
        self,
        interaction_file: str,
        tissue: str,
        ) -> List[Tuple[str, str, float, str]]:
        """Protein-protein interactions from the Integrated Interactions
        Database v 2021-05"""
        df = pd.read_csv(interaction_file, delimiter='\t')
        df = df[['symbol1', 'symbol2', 'evidence_type', 'n_methods', tissue]]
        t_spec_filtered = df[
            (df[tissue] > 0)
            & (df['n_methods'] >= 3)
            & (df['evidence_type'].str.contains('exp'))
            ]
        edges = list(
                zip(*map(t_spec_filtered.get, ['symbol1', 'symbol2']), repeat(-1), repeat('ppi'))
                )
        return [
            (
                self.genesymbol_to_gencode[edge[0]],
                self.genesymbol_to_gencode[edge[1]],
                edge[2],
                edge[3],
            )
            for edge in edges
            if edge[0] in self.genesymbol_to_gencode.keys()
            and edge[1] in self.genesymbol_to_gencode.keys()
        ]

    @time_decorator(print_args=True)
    def _marbach_regulatory_circuits(
        self,
        interaction_file: str
        ) -> List[Tuple[str, str, float, str]]:
        """Regulatory circuits from Marbach et al., Nature Methods, 2016. Each
        network is in the following format:
            col_1   TF
            col_2   Target gene
            col_3   Edge weight 
        """
        with open(interaction_file, newline = '') as file:
            return [
                (f'{self.genesymbol_to_gencode[line[0]]}_tf', self.genesymbol_to_gencode[line[1]], line[2], 'circuits')
                for line in csv.reader(file, delimiter='\t')
                if line[0] in self.genesymbol_to_gencode.keys() and line[1] in self.genesymbol_to_gencode.keys()
                ]

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
        e_dict_unfiltered = {
            enhancer:e_dict[e_dict_unlifted[enhancer]]
            for enhancer in e_dict_unlifted
            if e_dict_unlifted[enhancer] in e_dict.keys()
            }
        return {
            k:v for k,v in e_dict_unfiltered.items()
            if 'alt' not in v
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
        score_filter: int,
        ) -> List[Tuple[str, str, float, str]]:
        """Convert each enhancer-enhancer link to hg38 and return a formatted tuple"""
        e_e_liftover, scores = [], []
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            for line in file_reader:
                scores.append(int(line[2]))
                if line[0] in self.e_indexes.keys() and line[1] in self.e_indexes.keys():
                    e_e_liftover.append((self.e_indexes[line[0]], self.e_indexes[line[1]]))

        cutoff = np.percentile(scores, score_filter)
        return [
            (f"enhancers_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
            f"enhancers_{self._format_enhancer(line[1], 0)}_{self._format_enhancer(line[1], 1)}",
            -1,
            'enhancer-enhancer',)
            for line in e_e_liftover
            if int(line[2]) >= cutoff 
        ]

    @time_decorator(print_args=True)
    def _fenrir_enhancer_gene(
        self,
        interaction_file: str,
        score_filter: int,
        ) -> List[Tuple[str, str, float, str]]:
        """Convert each enhancer-gene link to hg38 and ensemble ID, return a formatted tuple"""
        e_g_liftover, scores = [], []
        with open(interaction_file, newline='') as file:
            file_reader = csv.reader(file, delimiter='\t')
            next(file_reader)
            for line in file_reader:
                scores.append(int(line[3]))
                if line[0] in self.e_indexes.keys() and line[2] in self.genesymbol_to_gencode.keys():
                    e_g_liftover.append((self.e_indexes[line[0]], self.genesymbol_to_gencode[line[2]]))

        cutoff = np.percentile(scores, score_filter)
        return [
            (f"enhancers_{self._format_enhancer(line[0], 0)}_{self._format_enhancer(line[0], 1)}",
            line[1],
            -1,
            'enhancer-gene')
            for line in e_g_liftover
            if int(line[3]) >= cutoff
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
            ppi_edges = self._iid_ppi(
                interaction_file=f"{self.interaction_dir}/{self.interaction_files['ppis']}",
                tissue=self.ppi_tissue
                )
            e_e_edges = self._fenrir_enhancer_enhancer(
                f"{self.interaction_dir}"
                f"/{self.tissue_specific['enhancers_e_e']}",
                score_filter=250
                )
            e_g_edges = self._fenrir_enhancer_gene(
                f"{self.interaction_dir}"
                f"/{self.tissue_specific['enhancers_e_g']}",
                score_filter=250
                )
            circuit_edges = self._marbach_regulatory_circuits(
                f"{self.interaction_dir}"
                f"/{self.interaction_files['circuits']}"
                )
            # interaction_edges = e_e_edges + e_g_edges + ppi_edges + giant_edges + circuit_edges
            interaction_edges = e_e_edges + e_g_edges + ppi_edges + circuit_edges
            with open(all_interaction_file, 'w+') as output:
                writer = csv.writer(output, delimiter='\t')
                writer.writerows(interaction_edges)
        else:
            pass

        polyadenylation = self._polyadenylation_targets(
            f"{self.interaction_dir}"
            f"/{self.interaction_files['polyadenylation']}"
            )
        
        base_graph = self._base_graph(edges=interaction_edges)
        return base_graph, polyadenylation
        
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

        if self.graph_type == 'local':
            edges = local_edges
            output_file = f'{self.parse_dir}/local_graphs/{gene}_{self.tissue}'
        else:
            edges = local_edges + edges_to_add
            output_file = f'{self.graph_dir}/{gene}_{self.tissue}'

        edges_reindexed, node_idxs, num_nodes = _reindex_nodes(edges)
        graph_only_refs = {node_idxs[node]:reference_attrs[node] for node in node_idxs}

        with open(f'{output_file}', 'wb') as output:
            pickle.dump({
            'edge_index': np.array([[edge[0] for edge in edges_reindexed], [edge[1] for edge in edges_reindexed]]),
            'edge_feat': np.array([self.ONEHOT_EDGETYPE[edge[2]] for edge in edges_reindexed]),
            'node_feat': np.array([[int(x) for x in list(graph_only_refs[key].values())[2:]]
            for key in graph_only_refs.keys()]),
            'num_nodes': num_nodes,
            },
            output
            )
        print(f'Finished _prepare_graph_tensors on {gene}')

    @time_decorator(print_args=True)
    def generate_graphs(self) -> None:
        """Constructs graphs in parallel"""
        # retrieve interaction-based edges
        base_graph, polyadenylation = self._interaction_preprocess()

        # prepare nested dict for node features
        reference_attrs = self._prepare_reference_attributes(
            gencode_ref=f'{self.parse_dir}/attributes/gencode_reference.pkl',
            polyadenylation=polyadenylation,
        )

        genes_to_construct = [
            gene for gene in self.genes
            if not (os.path.exists(f'{self.graph_dir}/{gene}_{self.tissue}')
            and os.stat(f'{self.graph_dir}/{gene}_{self.tissue}').st_size > 0)
        ]

        # prepare list of uniq interaction edges
        interaction_file = f'{self.interaction_dir}/interaction_edges.txt'
        cmd = f"awk '{{print $1 \"\\n\" $2}}' {interaction_file} \
            | sort -u \
            > {self.interaction_dir}/uniq_interaction_nodes.txt"

        subprocess.run(cmd, stdout=None, shell=True)

        # read interaction file into a list
        with open(interaction_file, newline='') as file:
            interaction_edges = [line for line in csv.reader(file, delimiter='\t')]

        print(f'total graphs to construct - {len(self.genes)}')
        print(f'starting construction on {len(genes_to_construct)} genes')

        # parse graph into tensors and save
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

    # get filtered genes
    _, tpm_filtered_genes = _tpm_filter_gene_windows(
        gencode=f"shared_data/local/{params['shared']['gencode']}",
        tissue=params['resources']['tissue'],
        tpm_file=params['resources']['tpm'],
        slop=False,
        chromfile=params['resources']['chromfile'],
        window=params['resources']['window'],
        )
    
    # instantiate object
    graphconstructingObject = GraphConstructor(
        params=params,
        genes=tpm_filtered_genes,
        graph_type='concatenated'
        )

    # run pipeline!
    graphconstructingObject.generate_graphs()


if __name__ == '__main__':
    main()