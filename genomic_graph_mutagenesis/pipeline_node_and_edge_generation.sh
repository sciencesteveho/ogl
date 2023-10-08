#!/bin/bash
#
# This script automates the process of node and edge generation per experimental
# setup, submitting a job to the cluster per tissue.

#SBATCH --job-name=pipelineA
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=bio210019p

# Number of cores requested
#SBATCH --ntasks-per-node=24

# Partition
#SBATCH -p RM-shared

# Time
#SBATCH -t 48:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

experiment_yaml=$1
tissue_yaml=$2

module load AI/anaconda3-tf2.2020.11
module load cuda/11.1.1
module load cudnn/8.0.4
conda activate /ocean/projects/bio210019p/stevesho/gnn

python -u genomic_graph_mutagenesis/genomic_graph_mutagenesis/pipeline_node_and_edge_generation.py \
    --experiment_config ${experiment_yaml} \
    --tissue_config ${tissue_yaml}