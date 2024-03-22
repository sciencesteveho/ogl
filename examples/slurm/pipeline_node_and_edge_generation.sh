#!/bin/bash
#
# This script automates the process of node and edge generation per experimental
# setup, submitting a job to the cluster per tissue.

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


experiment_yaml=$1
tissue_yaml=$2


python -u omics_graph_learning/omics_graph_learning/pipeline_node_and_edge_generation.py \
    --experiment_config ${experiment_yaml} \
    --tissue_config ${tissue_yaml}

