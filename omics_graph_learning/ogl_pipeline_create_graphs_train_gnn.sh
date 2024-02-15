#!/bin/bash
#
# Submit jobs for graph construction, graph concatenation, scaler creation,
# feature scaling, and gnn training. This script will first check if a final,
# scaled graph is already present. If it is, it will skip all jobs and move
# straight to GNN training. If it is not, it will submit all jobs in the
# pipeline.

# Parse command-line arguments for GNN training
parse_arguments() {
    # Initialize variables with default values
    experiment_yaml=""
    partition=""
    model=""
    layers=""
    dimensions=""
    epochs=""
    loader=""
    learning_rate=""
    batch_size=""
    idx=""
    graph_type=""
    zero_nodes=""
    randomize_node_feats=""
    early_stop=""
    expression_only=""
    randomize_edges=""
    total_random_edges=""

    # Use getopt with the additional --options and --longoptions flags
    options=$(getopt --options "e:p:m:l:d:o:r:b:i:g:z:n:s:h:p" --longoptions "experiment_yaml:,partition:,model:,layers:,dimensions:,epochs:,loader:,learning_rate:,batch_size:,idx:,graph_type:,zero_nodes:,randomize_node_feats:,early_stop:,expression_only:,randomize_edges:,total_random_edges:,help:" --name "$0" -- "$@")

    # Check for getopt errors
    if [ $? -ne 0 ]; then
        echo "Usage: $0 [--experiment_yaml] [--partition] [--model] [--layers] [--dimensions] [--epochs] [--loader] [--learning_rate] [--batch_size] [--idx] [--graph_type] [--zero_nodes] [--randomize_node_feats] [--early_stop] [--expression_only] [--randomize_edges] [--total_random_edges]"
        exit 1
    fi

    eval set -- "$options"

    while true; do
        case "$1" in
            -e|--experiment_yaml)
                experiment_yaml="$2"
                shift 2
                ;;
            -p|--partition)
                partition="$2"
                shift 2
                ;;
            -m|--model)
                model="$2"
                shift 2
                ;;
            -l|--layers)
                layers="$2"
                shift 2
                ;;
            -d|--dimensions)
                dimensions="$2"
                shift 2
                ;;
            -q|--epochs)
                epochs="$2"
                shift 2
                ;;
            -o|--loader)
                loader="$2"
                shift 2
                ;;
            -r|--learning_rate)
                learning_rate="$2"
                shift 2
                ;;
            -b|--batch_size)
                batch_size="$2"
                shift 2
                ;;
            -i|--idx)
                idx="$2"
                shift 2
                ;;
            -g|--graph_type)
                graph_type="$2"
                shift 2
                ;;
            -z|--zero_nodes)
                zero_nodes="$2"
                shift 2
                ;;
            -n|--randomize_node_feats)
                randomize_node_feats="$2"
                shift 2
                ;;
            -s|--early_stop)
                early_stop="$2"
                shift 2
                ;;
            -p|--expression_only)
                expression_only="$2"
                shift 2
                ;;
            -r|--randomize_edges)
                randomize_edges="$2"
                shift 2
                ;;
            -t|--total_random_edges)
                total_random_edges="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [--experiment_yaml] [--partition] [--model] [--layers] [--dimensions] [--epochs] [--loader] [--learning_rate] [--batch_size] [--idx] [--graph_type] [--zero_nodes] [--randomize_node_feats] [--early_stop] [--expression_only] [--randomize_edges] [--total_random_edges]"
                exit 0
                ;;
            --)
                shift
                break
                ;;
        esac
    done
}

# Call the function to parse command-line arguments
parse_arguments "$@"

# Set conda environment
module load anaconda3/2022.10
conda activate /ocean/projects/bio210019p/stevesho/gnn

# Check if final graph is already made and set up variables
working_directory=$(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['working_directory'])")
experiment_name=$(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['experiment_name'])")
tissues=($(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['tissues'])" | tr -d "[],'"))
final_graph=${working_directory}/${experiment_name}/graphs/${experiment_name}_full_graph_scaled.pkl

# start running pipeline!
if ! [ -f ${final_graph} ]; then
    echo "Final graph not found. Submitting pipeline jobs."

    # temporary fix
    # use EM partition for deeploop
    if [ ${partition} == "EM" ]; then
        node_and_edge_generator=pipeline_node_and_edge_generation_em.sh
    else
        node_and_edge_generator=pipeline_node_and_edge_generation.sh
    fi

    # Parse nodes and edges
    pipeline_a_ids=()
    for tissue in ${tissues[@]}; do
        ID=$(sbatch \
            --parsable \
            ${node_and_edge_generator} \
            ${experiment_yaml} \
            omics_graph_learning/configs/${tissue}.yaml
        )
        pipeline_a_ids+=($ID)
    done

    # Create graphs from nodes and edges individual tissues and combine into a single
    # representation
    # if [ ${partition} == "EM" ]; then
    #     constructor=graph_constructor_and_concat_em.sh
    # else
    #     constructor=graph_constructor_and_concat.sh
    # fi
    constructor=graph_concat.sh

    construct_id=$(
        sbatch \
        --parsable \
        --dependency=afterok:$(echo ${pipeline_a_ids[*]} | tr ' ' :) \
        ${constructor} \
        full \
        ${experiment_yaml}
    )

    # Get training targets by splitting dataset (genes)
    split_id=$(sbatch \
        --parsable \
        --dependency=afterok:${construct_id} \
        get_training_targets.sh \
        ${experiment_yaml}
    )

    # Create scalers after concat is finished. One scaler is made per node feature
    # and each scaler is made independently.
    slurmids=()
    for num in {0..38..1}; do
        ID=$(sbatch \
            --parsable \
            --dependency=afterok:${split_id} \
            make_scaler.sh \
            $num \
            full \
            ${experiment_yaml})
        slurmids+=($ID)
    done

    # Scale node feats after every scaler job is finished
    scale_id=$(sbatch \
        --parsable \
        --dependency=afterok:$(echo ${slurmids[*]} | tr ' ' :) \
        scale_node_feats.sh \
        full \
        ${experiment_yaml})


#     # train GNN after scaler job is finished
#     sbatch \
#         --dependency=afterok:${scale_id} \
#         train_gnn.sh \
#         ${experiment_yaml} \
#         ${model} \
#         ${layers} \
#         ${dimensions} \
#         ${epochs} \
#         ${loader} \
#         ${learning_rate} \
#         ${batch_size} \
#         ${idx} \
#         ${graph_type} \
#         ${zero_nodes} \
#         ${randomize_node_feats} \
#         ${early_stop} \
#         ${expression_only} \
#         ${randomize_edges} \
#         ${total_random_edges}

# else
#     echo "Final graph found. Going straight to GNN training."
#     # Train graph neural network

#     sbatch \
#         train_gnn.sh \
#         ${experiment_yaml} \
#         ${model} \
#         ${layers} \
#         ${dimensions} \
#         ${epochs} \
#         ${loader} \
#         ${learning_rate} \
#         ${batch_size} \
#         ${idx} \
#         ${graph_type} \
#         ${zero_nodes} \
#         ${randomize_node_feats} \
#         ${early_stop} \
#         ${expression_only} \
#         ${randomize_edges} \
#         ${total_random_edges}
# fi