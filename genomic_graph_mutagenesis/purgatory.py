#! /usr/bin/env python
# -*- coding: utf-8 -*-  
#

"""Purgatory zone for bits of leftover code"""


"""From prepare_bedfiles.py
"""
    # @time_decorator(print_args=True)
    # def _chromhmm_to_attribute(self, bed: str) -> None:
    #     """
    #     Split chromhmm to individual files of the following for node attributes:
    #     ['Enh', 'EnhBiv', 'EnhG', 'TssA', 'TssAFlnk', 'TssBiv', 'TxFlnk', 'Tx', 'TxWk', 'ZNF']
    #     """
    #     segmentations = [
    #         '1_TssA', '2_TssAFlnk', '3_TxFlnk', '4_Tx', '5_TxWk', '6_EnhG', '7_Enh', '8_ZNF', '9_Het', '10_TssBiv', '12_EnhBiv', 'ReprPC'
    #         ]

    #     for segmentation in segmentations:
    #         if segmentation == 'ReprPC':
    #             seg = segmentation
    #         else:
    #             seg = segmentation.split('_')[1]
    #         cmd = f"grep {segmentation} {self.root_tissue}/unprocessed/{bed} \
    #             > {self.root_tissue}/local/{seg.casefold()}_hg38.bed"
    #         self._run_cmd(cmd)


"""From graph_constructor.py
"""
    # # get filtered genes
    # _, tpm_filtered_genes = _tpm_filter_gene_windows(
    #     gencode=f"shared_data/local/{params['shared']['gencode']}",
    #     tissue=params['resources']['tissue'],
    #     tpm_file=params['resources']['tpm'],
    #     slop=False,
    #     chromfile=params['resources']['chromfile'],
    #     window=params['resources']['window'],
    #     )


    # @time_decorator(print_args=False)
    # def _prepare_reference_attributes(
    #     self,
    #     gencode_ref: str, 
    #     polyadenylation: List[str],
    #     ) -> Dict[str, Dict[str, Any]]:
    #     """Add polyadenylation to gencode ref dict used to fill """
    #     ref = pickle.load(open(f'{gencode_ref}', 'rb'))
    #     for gene in ref:
    #         if gene in polyadenylation:
    #             ref[gene]['polyadenylation'] = 1
    #         else:
    #             pass

    #     for node in self.NODES:
    #         ref_for_concat = pickle.load(
    #             open(f'{self.parse_dir}/attributes/{node}_reference.pkl', 'rb')
    #         )
    #         ref.update(ref_for_concat)
    #     return ref

    # @time_decorator(print_args=False)
    # def _prepare_graph_tensors(
    #     self,
    #     gene: str,
    #     reference_attrs: Dict[str, Dict[str, Any]],
    #     interaction_edges: List[Any],
    #     ) -> Any:
    #     """_lorem ipsum"""
    #     print(f'starting _prepare_graph_tensors on {gene}')

    #     def _reindex_nodes(edges):
    #         """_lorem"""
    #         uniq_nodes = sorted(
    #             set([edge[0] for edge in edges]+[edge[1] for edge in edges])
    #             )
    #         node_idxs = {node: id for id, node in enumerate(uniq_nodes)}
    #         edges_reindexed = list(
    #             map(lambda edge: [node_idxs[edge[0]], node_idxs[edge[1]], edge[2]], edges)
    #             )
    #         return sorted(edges_reindexed), node_idxs, len(uniq_nodes)

    #     gene_edges = f'{self.parse_dir}/edges/genes/{gene}'
    #     ### fast uniq_nodes 
    #     uniq_local_sort = f"awk '{{print $4 \"\\n\" $8}}' {gene_edges} \
    #         | sort -u"
    #     proc = subprocess.Popen(uniq_local_sort, shell=True, stdout=subprocess.PIPE)
    #     uniq_local = proc.communicate()[0]

    #     with open(f'{self.interaction_dir}/uniq_interaction_nodes.txt') as f:
    #         interaction_nodes = [line.rstrip('\n') for line in f.readlines()]

    #     nodes_to_add = set(str(uniq_local).split('\\n')).intersection(interaction_nodes)

    #     edges_to_add = [
    #         [line[0], line[1], line[3]] for line in
    #         filter(
    #             lambda interaction: interaction[0] in nodes_to_add or interaction[1] in nodes_to_add,
    #             interaction_edges
    #         )
    #     ]

    #     with open(gene_edges, newline='') as file:
    #         local_edges = [
    #             [line[3], line[7], 'local']
    #             for line in csv.reader(file, delimiter='\t')]

    #     edges = local_edges + edges_to_add
    #     output_file = f'{self.graph_dir}/{gene}_{self.tissue}'

    #     edges_reindexed, node_idxs, num_nodes = _reindex_nodes(edges)
    #     graph_only_refs = {node_idxs[node]:reference_attrs[node] for node in node_idxs}

    #     with open(f'{output_file}', 'wb') as output:
    #         pickle.dump({
    #         'edge_index': np.array([[edge[0] for edge in edges_reindexed], [edge[1] for edge in edges_reindexed]]),
    #         'edge_feat': np.array([self.ONEHOT_EDGETYPE[edge[2]] for edge in edges_reindexed]),
    #         'node_feat': np.array([[int(x) for x in list(graph_only_refs[key].values())[2:]]
    #         for key in graph_only_refs.keys()]),
    #         'num_nodes': num_nodes,
    #         },
    #         output
    #         )
    #     print(f'Finished _prepare_graph_tensors on {gene}')



        # # prepare nested dict for node features
        # reference_attrs = self._prepare_reference_attributes(
        #     gencode_ref=f'{self.parse_dir}/attributes/gencode_reference.pkl',
        # )

        # genes_to_construct = [
        #     gene for gene in self.genes
        #     if not (os.path.exists(f'{self.graph_dir}/{gene}_{self.tissue}')
        #     and os.stat(f'{self.graph_dir}/{gene}_{self.tissue}').st_size > 0)
        # ]

        # # prepare list of uniq interaction edges
        # interaction_file = f'{self.interaction_dir}/interaction_edges.txt'
        # cmd = f"awk '{{print $1 \"\\n\" $2}}' {interaction_file} \
        #     | sort -u \
        #     > {self.interaction_dir}/uniq_interaction_nodes.txt"

        # subprocess.run(cmd, stdout=None, shell=True)

        # # read interaction file into a list
        # with open(interaction_file, newline='') as file:
        #     interaction_edges = [line for line in csv.reader(file, delimiter='\t')]

        # print(f'total graphs to construct - {len(self.genes)}')
        # print(f'starting construction on {len(genes_to_construct)} genes')

        # # parse graph into tensors and save
        # pool = Pool(processes=24)
        # pool.starmap(
        #     self._prepare_graph_tensors,
        #     zip(genes_to_construct,
        #     repeat(reference_attrs),
        #     repeat(interaction_edges),
        # ))
        # pool.close()