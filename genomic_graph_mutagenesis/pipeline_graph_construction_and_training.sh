#!/bin/bash
#
# Submit jobs for graph construction, graph concatenation, scaler creation,
# feature scaling, and gnn training.
experiment_yaml=$1

# create graphs and concat via EM partition
construct_id=$(
    sbatch \
    --parsable \
    graph_constructor_and_concat.sh \
    full \
    --experiment_config $experiment_yaml
)

# create scaler after concat is finished
slurmids=()
for num in {0..38..1}; do
    ID=$(sbatch \
        --parsable \
        --dependency=afterok:${construct_id} \
        make_scaler.sh \
        $num \
        full \
        $experiment_yaml)
    slurmids+=($ID)
done

# scale node feats after every scaler job is finished
scale_id=$(sbatch \
    --parsable \
    --dependency=afterok:$(echo ${slurmids[*]} | tr ' ' :) \
    scale_node_feats.sh \
    full \
    --experiment_config $experiment_yaml)

# create training targets
sbatch dataset_split.py

# train neural network
sbatch \
    run_gnn.sh \
    GraphSage \
    3 \
    256 \
    neighbor \
    0.001 \
    512 \
    True \
    full \ 
    False 