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


# Set arguments!
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment_config) experiment_config="$2"; shift ;;
        --model) model="$2"; shift ;;
        --target) target="$2"; shift ;;
        --gnn_layers) gnn_layers="$2"; shift ;;
        --linear_layers) linear_layers="$2"; shift ;;
        --activation) activation="$2"; shift ;;
        --dimensions) dimensions="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --learning_rate) learning_rate="$2"; shift ;;
        --optimizer) optimizer="$2"; shift ;;
        --dropout) dropout="$2"; shift ;;
        --graph_type) graph_type="$2"; shift ;;
        --split_name) split_name="$2"; shift ;;
        --bool_flags) bool_flags="$2"; shift ;;
        --heads) heads="$2"; shift ;;
        --total_random_edges) total_random_edges="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set up the python command
train_command="python -u omics_graph_learning/omics_graph_learning/gnn.py \
    --experiment_config ${experiment_config} \
    --model ${model} \
    --target ${target} \
    --gnn_layers ${gnn_layers} \
    --linear_layers ${linear_layers} \
    --activation ${activation} \
    --dimensions ${dimensions} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --optimizer ${optimizer} \
    --dropout ${dropout} \
    --graph_type ${graph_type} \
    --split_name ${split_name} \
    ${bool_flags}"

# Add heads and total_random_edges to the command only if they are set
[ -n "$heads" ] && train_command+=" --heads ${heads}"
[ -n "$total_random_edges" ] && train_command+=" --total_random_edges ${total_random_edges}"

train_command+= "--heads 2 --residual"

echo $train_command
eval $train_command
