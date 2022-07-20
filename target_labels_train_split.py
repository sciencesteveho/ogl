#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Look into synder paper for mass spec values as a potential target

"""Get dataset train/val/split split"""


import pickle
from pickletools import read_unicodestring4
import pandas as pd
import numpy as np

from typing import Tuple

from scipy.special import boxcox
from cmapPy.pandasGEXpress.parse_gct import parse

from utils import genes_from_gff, time_decorator, TISSUE_TPM_KEYS



@time_decorator(print_args=True)
def chr_split_train_test_val(genes, test_chrs, val_chrs):
    """
    create a list of training, split, and val IDs
    """
    return {
        'train': [gene for gene in genes if genes[gene] not in test_chrs + val_chrs],
        'test': [gene for gene in genes if  genes[gene] in test_chrs],
        'validation': [gene for gene in genes if genes[gene] in val_chrs],
    }


def tpm_all_tissue_median(gct_file):
    """Get the median TPM per gene across ALL samples within GTEx V8 GCT"""
    df = parse(gct_file).data_df
    median_series = pd.Series(df.median(axis=1), name='all_tissues').to_frame()
    median_series.to_pickle('gtex_tpm_median_across_all_tissues.pkl')


def tissue_std_dev_and_mean() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get means and standard deviation for TPM for specific tissue
    Calulates fold change of median of tissue relative to all_tissue_median
    """


def protein_abundance_all_tissue_median(protein_file: str):
    '''
    For now, "BRAIN-CORTEX" values are being used for hippocampus due to the
    similarites shown between the tissues in GTEx Consortium, Science, 2020.
    Values are log2 so we inverse log them (2^x)
    '''
    df = pd.read_csv(
        protein_file,
        sep=',',
        index_col ='gene.id.full'
    ).drop(columns=['gene.id'])

    df = df.apply(np.exp2).fillna(0)  # relative abundances are log2
    return pd.Series(df.median(axis=1), name='all_tissues').to_frame()


def protein_std_dev_and_mean(protein_median_file: str) -> pd.DataFrame:
    """Get means and standard deviation for protein abundance for specific tissue
    Calulates fold change of median of tissue relative to all_tissue_median
    """
    tissues = ['Heart Ventricle', 'Brain Cortex', 'Breast']
    df = pd.read_csv(
        protein_median_file,
        sep=',',
        index_col ='gene.id.full',
        usecols=['gene.id.full'] + tissues
    )

    return df.apply(np.exp2).fillna(0)  # relative abundances are log2


def fold_change_median(tissue_df, all_median_df, type=None) -> pd.DataFrame:
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


def get_dict_with_target_array(split_dict, tissue, tpmkey, prokey, tpm_targets_df, protein_targets_df):
    new = {}
    for gene in split_dict:
        new[gene+'_'+tissue] = np.array([
            tpm_targets_df[tpmkey].loc[gene],  # median tpm in the tissue
            tpm_targets_df[tpmkey+'_foldchange'].loc[gene], # fold change
            protein_targets_df[prokey].loc[gene] if gene in protein_targets_df.index else -1,
            protein_targets_df[prokey+'_foldchange'].loc[gene] if gene in protein_targets_df.index else -1,
        ])
    return new

def main() -> None:
    """Pipeline to generate dataset split and target values"""
    genes = genes_from_gff('/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/interaction/gencode_v26_genes_only_with_GTEx_targets.bed')
    test_chrs=['chr8', 'chr9']
    val_chrs=['chr7', 'chr13']

    split = chr_split_train_test_val(
        genes=genes,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
    )

    directory = '/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data'
    with open(f"{directory}/graph_partition_test_{('-').join(test_chrs)}_val_{('-').join(val_chrs)}.pkl", 'wb') as output:
        pickle.dump(split, output)

    tpm_file = 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'
    tpm_median_file = 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct'
    protein_file = 'protein_relative_abundance_all_gtex.csv'
    protein_median_file = 'protein_relative_abundance_median_gtex.csv'

    ### proteins 
    pro_median_df = protein_std_dev_and_mean(protein_median_file)
    pro_all_median = protein_abundance_all_tissue_median(protein_file)
    protein_targets_df = fold_change_median(pro_median_df, pro_all_median, type='protein')

    ### expression TPMs 
    with open('gtex_tpm_median_across_all_tissues.pkl', 'rb') as file:
        tpm_all_median = pickle.load(file)

    tpm_median_df = parse(tpm_median_file).data_df
    tpm_targets_df = fold_change_median(tpm_median_df, tpm_all_median, type='tpm')

    mammary_train = get_dict_with_target_array(
        split['train'],
        tissue = 'mammary',
        tpmkey = 'breast_mammary_tissue',
        prokey = 'breast',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )

    hippocampus_train = get_dict_with_target_array(
        split['train'],
        tissue = 'hippocampus',
        tpmkey = 'brain_hippocampus',
        prokey = 'brain_cortex',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )

    left_ventricle_train = get_dict_with_target_array(
        split['train'],
        tissue = 'left_ventricle',
        tpmkey = 'heart_left_ventricle',
        prokey = 'heart_ventricle',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )
        
    mammary_train.update(hippocampus_train)
    mammary_train.update(left_ventricle_train)
    train_dict = mammary_train
    print(len(mammary_train.keys()))

    mammary_test = get_dict_with_target_array(
        split['test'],
        tissue = 'mammary',
        tpmkey = 'breast_mammary_tissue',
        prokey = 'breast',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )

    hippocampus_test = get_dict_with_target_array(
        split['test'],
        tissue = 'hippocampus',
        tpmkey = 'brain_hippocampus',
        prokey = 'brain_cortex',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )

    left_ventricle_test = get_dict_with_target_array(
        split['test'],
        tissue = 'left_ventricle',
        tpmkey = 'heart_left_ventricle',
        prokey = 'heart_ventricle',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )
        
    mammary_test.update(hippocampus_test)
    mammary_test.update(left_ventricle_test)
    test_dict = mammary_test
    print(len(mammary_test.keys()))

    mammary_validation = get_dict_with_target_array(
        split['validation'],
        tissue = 'mammary',
        tpmkey = 'breast_mammary_tissue',
        prokey = 'breast',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )

    hippocampus_validation = get_dict_with_target_array(
        split['validation'],
        tissue = 'hippocampus',
        tpmkey = 'brain_hippocampus',
        prokey = 'brain_cortex',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )

    left_ventricle_validation = get_dict_with_target_array(
        split['validation'],
        tissue = 'left_ventricle',
        tpmkey = 'heart_left_ventricle',
        prokey = 'heart_ventricle',
        tpm_targets_df=tpm_targets_df,
        protein_targets_df=protein_targets_df
        )
        
    mammary_validation.update(hippocampus_validation)
    mammary_validation.update(left_ventricle_validation)
    validation_dict = mammary_validation
    print(len(mammary_validation.keys()))

    targets = {
        'train': mammary_train,
        'test': mammary_test,
        'validation': mammary_validation
    }

    

if __name__ == '__main__':
    main()

import pandas as pd
from cmapPy.pandasGEXpress.parse_gct import parse
gct_file = 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct'
df = parse(gct_file).data_df
'''        .apply(lambda x: boxcox1p(x,0.25))'''