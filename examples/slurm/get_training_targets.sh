#!/bin/bash

#SBATCH --job-name=<job_name>
#SBATCH --mail-user=<user>
#SBATCH --account=<account>
#SBATCH --mail-type=FAIL

# Number of nodes requested
#SBATCH --ntasks-per-node=<ntasks_per_node>

# Partition
#SBATCH -p <partition>

# Time
#SBATCH -t <time

# output to a designated folder
#SBATCH -o <output_file>

#echo commands to stdout
set -x

# load modules according to your cluster's configuration
# module load anaconda3
conda activate ogl

yaml=$1
tpm=$2
filter=$3
splitname=$4
rna_seq_flag=$5

if [ "$rna_seq_flag" == "--rna_seq" ]; then
    python -u omics_graph_learning/omics_graph_learning/dataset_split.py --experiment_config ${yaml} --tpm_filter ${tpm} --percent_of_samples_filter ${filter} --split_name ${splitname} --rna_seq
else
    python -u omics_graph_learning/omics_graph_learning/dataset_split.py --experiment_config ${yaml} --tpm_filter ${tpm} --percent_of_samples_filter ${filter} --split_name ${splitname}
fi
