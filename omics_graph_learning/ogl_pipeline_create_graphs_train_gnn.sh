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
    target=""
    tpm_filter=""
    percent_of_samples_filter=""
    layers=""
    dimensions=""
    epochs=""
    learning_rate=""
    batch_size=""
    graph_type=""
    zero_nodes=""
    randomize_node_feats=""
    early_stop=""
    randomize_edges=""
    total_random_edges=""

    # Use getopt with the additional --options and --longoptions flags
    options=$(getopt --options "e:p:m:t:x:y:l:d:q:r:b:g:z:n:s:r:p" --longoptions "experiment_yaml:,partition:,model:,target:,tpm_filter:,percent_of_samples_filter:,layers:,dimensions:,epochs:,learning_rate:,batch_size:,graph_type:,zero_nodes:,randomize_node_feats:,early_stop:,randomize_edges:,total_random_edges:,help:" --name "$0" -- "$@")

    # Check for getopt errors
    if [ $? -ne 0 ]; then
        echo "Usage: $0 [--experiment_yaml] [--partition] [--model] [--target] [--tpm_filter] [--percent_of_samples_filter] [--layers] [--dimensions] [--epochs] [--learning_rate] [--batch_size] [--graph_type] [--zero_nodes] [--randomize_node_feats] [--early_stop]  [--randomize_edges] [--total_random_edges]"
        exit 1
    fi

    eval set -- "$options"

    while true; do
        case "$1" in
        -e | --experiment_yaml)
            experiment_yaml="$2"
            shift 2
            ;;
        -p | --partition)
            partition="$2"
            shift 2
            ;;
        -m | --model)
            model="$2"
            shift 2
            ;;
        -t | --target)
            target="$2"
            shift 2
            ;;
        -x | --tpm_filter)
            tpm_filter="$2"
            shift 2
            ;;
        -y | --percent_of_samples_filter)
            percent_of_samples_filter="$2"
            shift 2
            ;;
        -l | --layers)
            layers="$2"
            shift 2
            ;;
        -d | --dimensions)
            dimensions="$2"
            shift 2
            ;;
        -q | --epochs)
            epochs="$2"
            shift 2
            ;;
        -r | --learning_rate)
            learning_rate="$2"
            shift 2
            ;;
        -b | --batch_size)
            batch_size="$2"
            shift 2
            ;;
        -g | --graph_type)
            graph_type="$2"
            shift 2
            ;;
        -z | --zero_nodes)
            zero_nodes="$2"
            shift 2
            ;;
        -n | --randomize_node_feats)
            randomize_node_feats="$2"
            shift 2
            ;;
        -s | --early_stop)
            early_stop="$2"
            shift 2
            ;;
        -r | --randomize_edges)
            randomize_edges="$2"
            shift 2
            ;;
        -p | --total_random_edges)
            total_random_edges="$2"
            shift 2
            ;;
        -h | --help)
            echo "Usage: $0 [--experiment_yaml] [--partition] [--model] [--target] [--tpm_filter] [--percent_of_samples_filter] [--layers] [--dimensions] [--epochs] [--learning_rate] [--batch_size] [--graph_type] [--zero_nodes] [--randomize_node_feats] [--early_stop] [--randomize_edges] [--total_random_edges]"
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

# Set up variables and find out if final graph is already made
working_directory=$(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['working_directory'])")
experiment_name=$(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['experiment_name'])")
tissues=($(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['tissues'])" | tr -d "[],'"))
split_name=$(python splitname.py --experiment_config ${experiment_yaml} --tpm_filter ${tpm_filter} --percent_of_samples_filter ${percent_of_samples_filter})

# set up variables for graph checking
final_graph=${working_directory}/${experiment_name}/graphs/${split_name}/${experiment_name}_tpm_${tpm_filter}_percent_of_samples_${percent_of_samples_filter}_${graph_type}_graph_scaled.pkl
intermediate_graphs=${working_directory}/${experiment_name}/graphs/${experiment_name}_${graph_type}.pkl

# Start running pipeline
if [ -f "${final_graph}" ]; then
    if [ -f "${intermediate_graph}" ]; then
        echo "Intermediate graph found. Running dataset split, scaler, and training."

        # Get training targets by splitting dataset (genes)
        split_id=$(
            sbatch --parsable \
                get_training_targets.sh \
                ${experiment_yaml}
        )

    else
        echo "No intermediates found. Running entire pipeline!"

        # Determine node and edge generator script
        if [ "${partition}" == "EM" ]; then
            node_and_edge_generator=pipeline_node_and_edge_generation_em.sh
        else
            node_and_edge_generator=pipeline_node_and_edge_generation.sh
        fi

        # Parse nodes and edges
        pipeline_a_ids=()
        for tissue in "${tissues[@]}"; do
            ID=$(
                sbatch --parsable \
                    "${node_and_edge_generator}" \
                    "${experiment_yaml}" \
                    "omics_graph_learning/configs/${tissue}.yaml"
            )
            pipeline_a_ids+=("${ID}")
        done

        # Concatenate graphs
        constructor=graph_concat.sh
        construct_id=$(
            sbatch --parsable \
                --dependency=afterok:"$(echo "${pipeline_a_ids[*]}" | tr ' ' ':')" \
                "${constructor}" \
                full \
                "${experiment_yaml}"
        )

        # Get training targets after graph construction
        split_id=$(
            sbatch --parsable \
                --dependency=afterok:"${construct_id}" \
                get_training_targets.sh \
                "${experiment_yaml}"
        )
    fi

    # Create scalers after concat is finished. One scaler per node feature.
    slurmids=()
    for num in {0..38}; do
        ID=$(
            sbatch --parsable \
                --dependency=afterok:"${split_id}" \
                make_scaler.sh \
                "${num}" \
                full \
                "${experiment_yaml}"
        )
        slurmids+=("${ID}")
    done

    # Scale node feats after every scaler job is finished
    scale_id=$(
        sbatch --parsable \
            --dependency=afterok:"$(echo "${slurmids[*]}" | tr ' ' ':')" \
            scale_node_feats.sh \
            full \
            "${experiment_yaml}"
    )

    # Train GNN after scaler job is finished
    sbatch --dependency=afterok:"${scale_id}" \
        train_gnn.sh \
        "${experiment_yaml}" \
        "${model}" \
        "${target}" \
        "${tpm_filter}" \
        "${percent_of_samples_filter}" \
        "${layers}" \
        "${dimensions}" \
        "${epochs}" \
        "${learning_rate}" \
        "${batch_size}" \
        "${graph_type}" \
        "${zero_nodes}" \
        "${randomize_node_feats}" \
        "${early_stop}" \
        "${randomize_edges}" \
        "${total_random_edges}"

else
    echo "Final graph found. Going straight to GNN training."

    # Train graph neural network
    sbatch \
        train_gnn.sh \
        "${experiment_yaml}" \
        "${model}" \
        "${target}" \
        "${tpm_filter}" \
        "${percent_of_samples_filter}" \
        "${layers}" \
        "${dimensions}" \
        "${epochs}" \
        "${learning_rate}" \
        "${batch_size}" \
        "${graph_type}" \
        "${zero_nodes}" \
        "${randomize_node_feats}" \
        "${early_stop}" \
        "${randomize_edges}" \
        "${total_random_edges}"
fi
