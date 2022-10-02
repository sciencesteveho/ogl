#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Look into synder paper for mass spec values as a potential target

"""Get dataset train/val/split split"""


import csv
from operator import concat
import pickle
import pandas as pd
import numpy as np
import random

from typing import List, Tuple

from cmapPy.pandasGEXpress.parse_gct import parse

from utils import genes_from_gff, time_decorator, TISSUE_TPM_KEYS


DATA_SPLITS = ['train', 'test', 'validation']

TISSUE_PARAMS = {
    'mammary': ('breast_mammary_tissue', 'breast', 'breast_mammary_tissue'),  
    'hippocampus': ('brain_hippocampus', 'brain_cortex', 'brain_hippocampus'),
    'left_ventricle': ('heart_left_ventricle', 'heart_ventricle', 'heart_left_ventricle'),
    'liver': ('liver', 'liver', 'liver'),  
    'lung': ('lung', 'lung', 'lung'),  
    'pancreas': ('pancreas', 'pancreas', 'pancreas'),  
    'skeletal_muscle': ('muscle_skeletal', 'muscle_skeletal', 'muscle_skeletal'),  
    }  # tissue: (tpm_key, protein_key, filtered_tpm_filename)


@time_decorator(print_args=True)
def _filter_low_tpm(
    tissue: str,
    file: str,
    return_list: False,
    ) -> List[str]:
    """Remove genes expressing less than 0.10 TPM across 20% of samples"""
    df = pd.read_table(file, index_col=0, header=[2])
    sample_n = len(df.columns)
    df['total'] = df.select_dtypes(np.number).gt(0.10).sum(axis=1)
    df['result'] = df['total'] >= (.2 * sample_n)
    if return_list == False:
        return [
            f'{gene}_{tissue}' for gene
            in list(df.loc[df['result'] == True].index)
        ]
    else:
        return list(df.loc[df['result'] == True].index)


@time_decorator(print_args=True)
def _chr_split_train_test_val(genes, test_chrs, val_chrs):
    """
    Create a list of training, split, and val IDs
    """
    return {
        'train': [gene for gene in genes if genes[gene] not in test_chrs + val_chrs],
        'test': [gene for gene in genes if  genes[gene] in test_chrs],
        'validation': [gene for gene in genes if genes[gene] in val_chrs],
    }


def _tpm_all_tissue_median(gct_file):
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT"""
    df = parse(gct_file).data_df
    median_series = pd.Series(df.median(axis=1), name='all_tissues').to_frame()
    median_series.to_pickle('gtex_tpm_median_across_all_tissues.pkl')


def _tissue_std_dev_and_mean() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get means and standard deviation for TPM for specific tissue
    Calulates fold change of median of tissue relative to all_tissue_median
    """


def _protein_abundance_all_tissue_median(protein_file: str):
    '''
    For now, "BRAIN-CORTEX" values are being used for hippocampus. We choose cortex due
    to similarites shown between the tissues in GTEx Consortium, Science, 2020.
    Values are log2 so we inverse log them (2^x)
    '''
    df = pd.read_csv(
        protein_file,
        sep=',',
        index_col ='gene.id.full'
    ).drop(columns=['gene.id'])

    df = df.apply(np.exp2).fillna(0)  # relative abundances are log2
    return pd.Series(df.median(axis=1), name='all_tissues').to_frame()


def _protein_std_dev_and_mean(protein_median_file: str) -> pd.DataFrame:
    """Get means and standard deviation for protein abundance for specific tissue
    Calulates fold change of median of tissue relative to all_tissue_median
    """
    tissues = ['Heart Ventricle', 'Brain Cortex', 'Breast', 'Liver', 'Lung', 'Muscle Skeletal', 'Pancreas']
    df = pd.read_csv(
        protein_median_file,
        sep=',',
        index_col ='gene.id.full',
        usecols=['gene.id.full'] + tissues
    )

    return df.apply(np.exp2).fillna(0)  # relative abundances are log2


@time_decorator(print_args=True)
def _fold_change_median(tissue_df, all_median_df, type=None) -> pd.DataFrame:
    """_lorem"""
    if type == 'tpm':
        regex = (('- ', ''), ('-', ''), ('(', ''), (')', ''), (' ', '_'))
    else:
        regex = ((' ', '_'), ('', ''))  # second added as placeholder so we can use *r

    df = pd.concat([tissue_df, all_median_df], axis=1)
    for tissue in list(df.columns)[:-1]:
        tissue_rename = tissue.casefold()
        for r in regex:
            tissue_rename = tissue_rename.replace(*r)
        df.rename(columns={f'{tissue}': f'{tissue_rename}'}, inplace=True)
        df[f'{tissue_rename}_foldchange'] = (df[f'{tissue_rename}']+0.01) / (df['all_tissues']+0.01) # add .01 to TPM to avoid negative infinity
    return df.apply(lambda x: np.log1p(x))


@time_decorator(print_args=True)
def _get_dict_with_target_array(split_dict, tissue, tpmkey, prokey, tpm_targets_df, protein_targets_df):
    """_lorem"""
    new = {}
    for gene in split_dict:
        new[gene+'_'+tissue] = np.array([
            tpm_targets_df[tpmkey].loc[gene],  # median tpm in the tissue
            tpm_targets_df[tpmkey+'_foldchange'].loc[gene], # fold change
            protein_targets_df[prokey].loc[gene] if gene in protein_targets_df.index else -1,
            protein_targets_df[prokey+'_foldchange'].loc[gene] if gene in protein_targets_df.index else -1,
        ])
    return new


@time_decorator(print_args=True)
def _combine_tissue_dicts(
    tissue_params: dict,
    split: dict,
    tpm_targets_df: pd.DataFrame,
    protein_targets_df: pd.DataFrame
    ):
    """_lorem"""
    for idx, tissue in enumerate(tissue_params):
        if idx == 0:
            all_dict = _get_dict_with_target_array(
                split_dict=split,
                tissue=tissue,
                tpmkey=tissue_params[tissue][0],
                prokey=tissue_params[tissue][1],
                tpm_targets_df=tpm_targets_df,
                protein_targets_df=protein_targets_df  
            )
        else:
            update_dict = _get_dict_with_target_array(
                split_dict=split,
                tissue=tissue,
                tpmkey=tissue_params[tissue][0],
                prokey=tissue_params[tissue][1],
                tpm_targets_df=tpm_targets_df,
                protein_targets_df=protein_targets_df  
            )
            all_dict.update(update_dict)
    return all_dict


def tissue_targets(
    split: dict,
    tissue_params: dict,
    tpm_pkl: str,
    tpm_median_file: str,
    protein_file: str,
    protein_median_file: str,
    ):
    """_lorem"""

    # proteins 
    pro_median_df = _protein_std_dev_and_mean(protein_median_file)
    pro_all_median = _protein_abundance_all_tissue_median(protein_file)
    protein_targets_df = _fold_change_median(pro_median_df, pro_all_median, type='protein')

    # expression TPMs 
    with open(tpm_pkl, 'rb') as file:
        tpm_all_median = pickle.load(file)

    tpm_median_df = parse(tpm_median_file).data_df
    tpm_targets_df = _fold_change_median(tpm_median_df, tpm_all_median, type='tpm')

    return {
        data_split: _combine_tissue_dicts(
            tissue_params,
            split[data_split],
            tpm_targets_df,
            protein_targets_df,
        ) for data_split in DATA_SPLITS
    }


def filtered_targets(
    tissue_params,
    targets,
    ):
    """Filters a dict of every possible target (all 56200 genes)
    Takes the pre-tpm filtered list of genes in each directory and concats into a list
    Keeps targets if they exist within this concatenated list"""
    def filtered_genes(tpm_filtered_genes: str) -> List[str]:
        with open(tpm_filtered_genes, newline='') as file:
            return [f'{line[3]}_{tissue}' for line in csv.reader(file, delimiter='\t')]

    for idx, tissue in enumerate(tissue_params):
        if idx ==0:
            genes = filtered_genes(f'{tissue}/gene_regions_tpm_filtered.bed')
        else:
            update_genes = filtered_genes(f'{tissue}/gene_regions_tpm_filtered.bed')
            genes += update_genes

    for key in targets.keys():
        targets[key] = {gene: targets[key][gene] for gene in targets[key].keys() if gene in genes}
    return targets


def max_node_filter(max_nodes, filtered_stats, targets, randomizer=False):
    """_lorem"""
    filtered_targets = {gene:value for gene, value in filtered_stats.items() if value[0] <= max_nodes}
    print(f'max_nodes = {max([value[0] for idx, value in filtered_targets.items()])} for max_nodes {max_nodes}')
    filtered_genes = list(filtered_targets.keys())
    filtered_dict = {}
    if randomizer:
        randomkeys = {'train': 0, 'test': 1, 'validation': 2}
        random.shuffle(filtered_genes)
        randomized = np.array_split(filtered_genes, 3)
        alltargets = {**targets['train'], **targets['test'], **targets['validation']}
        for key in randomkeys:
            filtered_dict[key] = {gene: value for gene, value in alltargets.items() if gene in randomized[randomkeys[key]]}
        with open(f'targets_filtered_random_{max_nodes}.pkl', 'wb') as output:
            pickle.dump(filtered_dict, output)
    else:
        for key in ['train', 'test', 'validation']:
            filtered_dict[key] = {gene: value for gene, value in targets[key].items() if gene in filtered_genes}
        with open(f'targets_filtered_{max_nodes}.pkl', 'wb') as output:
            pickle.dump(filtered_dict, output)


def concat_graph_stats(tissue_params):
    """combines the graph stats files, each of which are dicts of the following format
        gene: (num_nodes, num_edges)
    contains a minor nested function to add tissue name to the dict key
    """
    def open_graph_stats(tissue):
        with open(f'/ocean/projects/bio210019p/stevesho/data/preprocess/check_num_nodes/num_nodes_{tissue}.pkl', 'rb') as file:
            return {f'{key}_{tissue}':value for key,value in pickle.load(file).items()}
    for idx, tissue in enumerate(tissue_params):
        if idx == 0:
            graph_stats = open_graph_stats(tissue)
        else:
            update_stats = open_graph_stats(tissue)
            graph_stats.update(update_stats)
    return graph_stats


def main() -> None:
    """Pipeline to generate dataset split and target values"""

    # split genes in train, test, validation
    genes = genes_from_gff('/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed')
    test_chrs = ['chr8', 'chr9']
    val_chrs = ['chr7', 'chr13']

    # split genes by chr holdouts
    split = _chr_split_train_test_val(
        genes=genes,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )

    # save split partition
    directory = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data'
    with open(f"{directory}/graph_partition_test_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}.pkl", 'wb') as output:
        pickle.dump(split, output)

    # get targets
    targets = tissue_targets(
        split=split,
        tissue_params=TISSUE_PARAMS,
        tpm_pkl='/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/gtex_tpm_median_across_all_tissues.pkl',
        tpm_median_file='GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct',
        protein_file='protein_relative_abundance_all_gtex.csv',
        protein_median_file='protein_relative_abundance_median_gtex.csv',
        )  # 333886/31283/28231 train/test/validation, total = 393400

    # filter targets by tpm
    filtered_targets = filtered_targets(
        TISSUE_PARAMS,
        targets
        )  # 159424/14395/13161 train/test/validation, total = 186980

    # concatenate and save a file with all num_nodes and num_edges
    with open('filtered_stats.pkl', 'wb') as output:
        pickle.dump(concat_graph_stats(TISSUE_PARAMS), output)

    # save targets
    with open('filtered_targets_7_tissues_v3.pkl', 'wb') as output:
        pickle.dump(filtered_targets, output)


    # code to get max_node filtered targets for end-to-end debugging
    # open files
    with open('filtered_stats.pkl', 'rb') as file:
        filtered_stats = pickle.load(file)

    with open('filtered_targets_7_tissues_v3.pkl', 'rb') as file:
        targets = pickle.load(file)

    for num in [500, 1000, 2500, 5000, 10000]:
        max_node_filter(
            max_nodes=num,
            filtered_stats=filtered_stats,
            targets=targets,
            randomizer=True
        )


if __name__ == '__main__':
    main()


# def print_stats(num):
#     with open(f'targets_filtered_{num}.pkl', 'rb') as file:
#         targets = pickle.load(file)
#     print(f'max_nodes = {num}')
#     print(f"train = {len(targets['train'])}")
#     print(f"test = {len(targets['test'])}")
#     print(f"validation = {len(targets['validation'])}")
#     print('\n')

# for num in [500, 1000, 2500, 5000, 10000]:
#     print_stats(num)


# max_nodes = 500
# train = 3098
# test = 883
# validation = 257

# max_nodes = 1000
# train = 7839
# test = 1664
# validation = 780

# max_nodes = 2500
# train = 25519
# test = 4350
# validation = 2881

# max_nodes = 5000
# train = 55217
# test = 7088
# validation = 6716

# max_nodes = 10000
# train = 100490
# test = 10221
# validation = 10205

# import pandas as pd
# from cmapPy.pandasGEXpress.parse_gct import parse
# gct_file = 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'
# df = parse(gct_file).data_df
# '''        .apply(lambda x: boxcox1p(x,0.25))'''