# genes = {line[3]:line[9].split(';')[3].split('\"')[1] for line in a}
# genes_cleaned = [line[3].split('.')[0] for line in a]


# df = pd.DataFrame(columns=['ensembl', 'gene_symbol', 'entrez', 'symbol_2'])

# for idx, gene in enumerate(genes.items()):
#     meta = mg.query(gene[1], fields=['symbol', 'entrezgene'], species='human', verbose=False)
#     print(gene[1])
#     print(meta)
#     print('\n')
#     try:
#         result = meta['hits'][0]
#         for col in ['symbol', 'entrezgene']:
#             if col not in result:
#                 result[col] = 'NA'
#         df.loc[idx] = [gene[0], gene[1], result['entrezgene'], result['symbol'],]
#     except IndexError:
#         df.loc[idx] = [gene[0], gene[1], 'NA', 'NA',]

# df.to_pickle('id_table.txt')

import csv
import pandas as pd
import pybedtools

from mygene import MyGeneInfo

mg = MyGeneInfo()

def genes_from_gencode(gencode):
    """returns a dict as well as a list of cleaned genes"""
    symbols, ref = [], {}
    a = pybedtools.BedTool(gencode)
    for line in a:
        ref[line[9].split(';')[3].split('\"')[1]] = line[3]
        symbols.append(line[9].split(';')[3].split('\"')[1])
    return ref, [symbol for symbol in symbols if symbols.count(symbol) == 1]
    # return {line[9].split(';')[3].split('\"')[1]:line[3] for line in a}
    # return {line[3]:line[9].split(';')[3].split('\"')[1] for line in a}, [line[3].split('.')[0] for line in a]

def get_entrez_to_symbol_ref(edge_list):
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


def read_giant(graph):
    edges, edge1, edge2 = [], [], []
    with open(graph, newline='') as file:
        lines = csv.reader(file, delimiter='\t')
        for line in lines:
            if line[2] == '1':
                edges.append(line)
                edge1.append(line[0])
                edge2.append(line[1])
    return edges, set(edge1+edge2)

def convert_giant(edges, symbol_ref, ensembl_ref):
    def convert_genes(edges, ref):
        return [
            (ref[edge[0]],
            ref[edge[1]],
            -1,
            'giant',)
            for edge in edges
            if edge[0] in ref.keys()
            and edge[1] in ref.keys()
        ]

    giant_symbols = convert_genes(edges, symbol_ref)
    giant_filtered = [edge for edge in giant_symbols if edge[0] != 'NA' and edge[1] != 'NA']
    return convert_genes(giant_filtered, ensembl_ref)


graph = 'liver.dat'
gencode = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed'
ensembl_ref, _ = genes_from_gencode(gencode)
edges, edge_list = read_giant(graph)
symbol_ref = get_entrez_to_symbol_ref(edge_list)
giant_converted = convert_giant(edges, symbol_ref, ensembl_ref)


tpm_file = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/tpm/'
tisues = ['brain_hippocampus.tpm.txt', 'breast_mammary_tissue.tpm.txt', 'heart_left_ventricle.tpm.txt', 'liver.tpm.txt', 'lung.tpm.txt', 'pancreas.tpm.txt', 'muscle_skeletal.tpm.txt']
tpm_files = [tpm_file+x for x in tisues]


allgenes = []
for tpm_file in tpm_files:
    tpm_filtered_genes = _filter_low_tpm(
        tissue='brain_hippocampus',
        file=tpm_file,
        return_list=True,
    )
    allgenes = allgenes + tpm_filtered_genes

test = [x for x in allgenes if x in dupe_genes.keys()]
