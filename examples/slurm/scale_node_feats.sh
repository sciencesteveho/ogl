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

gtype=$1
config=$2
splitname=$3


python -u omics_graph_learning/omics_graph_learning/scale_node_feats.py -g $1 --experiment_config $2 --split_name $3
