#!/bin/bash
#
# Submit jobs for graph construction, graph concatenation, scaler creation,
# feature scaling, and gnn training.
experiment_yaml=$1

# create graphs and concat via EM partition
sbatch graph_constructor_and_concat.sh \
    full \
    --experiment_config $experiment_yaml

# create scaler after concat is finished
# sbatch --dependency=afterok:$(echo ${slurmids[*]} | tr ' ' :) 
for i in {0..38..1}; do
    sbatch make_scaler.sh \
        $i \
        $experiment_yaml \
        full
done

# scale node feats after every scaler job is finished
# sbatch --dependency=afterok:$(echo ${slurmids[*]} | tr ' ' :) 
sbatch scale_node_features.sh \
    full \
    --experiment_config $experiment_yaml