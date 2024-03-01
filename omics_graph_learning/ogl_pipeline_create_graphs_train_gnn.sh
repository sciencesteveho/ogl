#!/bin/bash
#
# Submit jobs for graph construction, graph concatenation, scaler creation,
# feature scaling, and gnn training. This script will perform checks for
# intermediate and final graphs before running the pipeline and submit jobs as
# necessary.

# Logging function to track progress
log_progress() {
    local message="$1"
    echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] $message"
}

# Parse command-line arguments for GNN training
parse_arguments() {
    # Initialize variables with default values
    experiment_yaml=""
    partition=""
    model=""
    target=""
    tpm_filter=""
    percent_of_samples_filter=""
    gnn_layers=""
    linear_layers=""
    activation=""
    dimensions=""
    epochs=""
    learning_rate=""
    optimizer=""
    dropout=""
    heads=""
    batch_size=""
    graph_type=""

    # Use getopt with adjusted flags for optional boolean arguments
    options=$(getopt --options q:w:e:r:t:y:u:i:o:p:a:s:d:f:g:k:ljzxcvbh --longoptions experiment_yaml:,partition:,model:,target:,tpm_filter:,percent_of_samples_filter:,gnn_layers:,linear_layers:,activation:,dimensions:,epochs:,learning_rate:,optimizer:,dropout:,heads:,batch_size:,graph_type:,residual,zero_nodes,randomize_node_feats,early_stop,randomize_edges,total_random_edges,help --name "$0" -- "$@")

    # Check for getopt errors
    if [ $? -ne 0 ]; then
        echo "Usage: $0 [--experiment_yaml] [--partition] [--model] [--target] [--tpm_filter] [--percent_of_samples_filter] [--gnn_layers] [--linear_layers] [--dimensions] [--epochs] [--learning_rate] [--optimizer] [--dropout] [--heads] [--batch_size] [--graph_type] [--residual] [--zero_nodes] [--randomize_node_feats] [--early_stop]  [--randomize_edges] [--total_random_edges]"
        exit 1
    fi

    # Handle potential getopt errors and invalid options
    if [[ $? -ne 0 ]]; then
        log_progress "Invalid option provided."
        exit 1
    fi

    eval set -- "$options"
    log_progress "Starting to parse arguments..."

    while true; do
        case "$1" in
        -q | --experiment_yaml)
            experiment_yaml="$2"
            log_progress "Setting experiment_yaml to $2"
            shift 2
            ;;
        -w | --partition)
            partition="$2"
            log_progress "Setting partition to $2"
            shift 2
            ;;
        -e | --model)
            model="$2"
            log_progress "Setting model to $2"
            shift 2
            ;;
        -r | --target)
            target="$2"
            log_progress "Setting target to $2"
            shift 2
            ;;
        -t | --tpm_filter)
            tpm_filter="$2"
            log_progress "Setting tpm_filter to $2"
            shift 2
            ;;
        -y | --percent_of_samples_filter)
            percent_of_samples_filter="$2"
            log_progress "Setting percent_of_samples_filter to $2"
            shift 2
            ;;
        -u | --gnn_layers)
            gnn_layers="$2"
            log_progress "Setting gnn_layers to $2"
            shift 2
            ;;
        -i | --linear_layers)
            linear_layers="$2"
            log_progress "Setting linear_layers to $2"
            shift 2
            ;;
        -o | --activation)
            activation="$2"
            log_progress "Setting activation to $2"
            shift 2
            ;;
        -p | --dimensions)
            dimensions="$2"
            log_progress "Setting dimensions to $2"
            shift 2
            ;;
        -a | --epochs)
            epochs="$2"
            log_progress "Setting epochs to $2"
            shift 2
            ;;
        -s | --learning_rate)
            learning_rate="$2"
            log_progress "Setting learning_rate to $2"
            shift 2
            ;;
        -d | --optimizer)
            optimizer="$2"
            log_progress "Setting optimizer to $2"
            shift 2
            ;;
        -f | --batch_size)
            batch_size="$2"
            log_progress "Setting batch_size to $2"
            shift 2
            ;;
        -g | --dropout)
            dropout="$2"
            log_progress "Setting dropout to $2"
            shift 2
            ;;
        -j | --heads)
            heads="$2"
            log_progress "Setting heads to $2"
            shift 2
            ;;
        -k | --graph_type)
            graph_type="$2"
            log_progress "Setting graph_type to $2"
            shift 2
            ;;
        -l | --residual)
            residual=true
            log_progress "Setting residual"
            shift
            ;;
        -z | --zero_nodes)
            zero_nodes=true
            log_progress "Setting zero_nodes"
            shift
            ;;
        -x | --randomize_node_feats)
            randomize_node_feats=true
            log_progress "Setting randomize_node_feats"
            shift
            ;;
        -c | --early_stop)
            early_stop=true
            log_progress "Setting early_stop"
            shift
            ;;
        -v | --randomize_edges)
            randomize_edges=true
            log_progress "Setting randomize_edges"
            shift
            ;;
        -b | --total_random_edges)
            total_random_edges="$2"
            log_progress "Setting total_random_edges to $2"
            shift 2
            ;;
        -h | --help)
            echo "Usage: $0 [--experiment_yaml] [--partition] [--model] [--target] [--tpm_filter] [--percent_of_samples_filter] [--gnn_layers] [--linear_layers] [--dimensions] [--epochs] [--learning_rate] [--optimizer] [--dropout] [--heads] [--batch_size] [--graph_type] [--residual] [--zero_nodes] [--randomize_node_feats] [--early_stop] [--randomize_edges] [--total_random_edges"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        esac
    done

    # Construct a string of boolean flags for argparse
    bool_flags=""
    if [[ $residual = true ]]; then bool_flags="--residual $bool_flags"; fi
    if [[ $zero_nodes = true ]]; then bool_flags="--zero_nodes $bool_flags"; fi
    if [[ $randomize_node_feats = true ]]; then bool_flags="--randomize_node_feats $bool_flags"; fi
    if [[ $early_stop = true ]]; then bool_flags="--early_stop $bool_flags"; fi
    if [[ $randomize_edges = true ]]; then bool_flags="--randomize_edges $bool_flags"; fi
}

# Call the function to parse command-line arguments
parse_arguments "$@"
log_progress "Command-line arguments parsed, with boolean flags: ${bool_flags}"

# Set conda environment
module load anaconda3/2022.10
conda activate /ocean/projects/bio210019p/stevesho/gnn

# Set up variables and find out if final graph is already made
splitname_script=omics_graph_learning/omics_graph_learning/splitname.py
working_directory=$(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['working_directory'])")
experiment_name=$(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['experiment_name'])")
tissues=($(python -c "import yaml; print(yaml.safe_load(open('${experiment_yaml}'))['tissues'])" | tr -d "[],'"))
split_name=$(python ${splitname_script} --experiment_config ${experiment_yaml} --tpm_filter ${tpm_filter} --percent_of_samples_filter ${percent_of_samples_filter})
log_progress "Working directory: ${working_directory}\nExperiment name: ${experiment_name}\nTissues: ${tissues[*]}\nSplit name: ${split_name}"

# set up variables for graph checking
final_graph=${working_directory}/${experiment_name}/graphs/${split_name}/${experiment_name}_tpm_${tpm_filter}_percent_of_samples_${percent_of_samples_filter}_${graph_type}_graph_scaled.pkl
intermediate_graphs=${working_directory}/${experiment_name}/graphs/${experiment_name}_${graph_type}.pkl
log_progress "Conda environment and python arguments parsed."

# set up training script
train="train_gnn.sh \
    ${experiment_yaml} \
    ${gnn_layers} \
    ${linear_layers} \
    ${activation} \
    ${dimensions} \
    ${epochs} \
    ${batch_size} \
    ${learning_rate} \
    ${optimizer} \
    ${dropout} \
    ${heads} \
    ${graph_type} \
    ${split_name} \
    ${bool_flags}"

if [[ -n $total_random_edges ]]; then
    train="${train} ${total_random_edges}"
fi

# Start running pipeline
log_progress "Checking for final graph: ${final_graph}"
if [ -f "${final_graph}" ]; then
    log_progress "Checking for intermediate graph: ${intermediate_graphs}"
    if [ -f "${intermediate_graph}" ]; then
        log_progress "Intermediate graph found. Running dataset split, scaler, and training."

        # Get training targets by splitting dataset (genes)
        split_id=$(
            sbatch --parsable \
                get_training_targets.sh \
                ${experiment_yaml}
        )

    else
        log_progress "No intermediates found. Running entire pipeline!"

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
                "${experiment_yaml}" \
                "${tpm_filter}" \
                "${percent_of_samples_filter}" \
                "${split_name}"
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
                "${experiment_yaml}" \
                "${split_name}"
        )
        slurmids+=("${ID}")
    done

    # Scale node feats after every scaler job is finished
    scale_id=$(
        sbatch --parsable \
            --dependency=afterok:"$(echo "${slurmids[*]}" | tr ' ' ':')" \
            scale_node_feats.sh \
            full \
            "${experiment_yaml}" \
            "${split_name}"
    )

    # Train GNN after scaler job is finished
    sbatch --dependency=afterok:"${scale_id} ${train}"

else
    log_progress "Final graph found. Going straight to GNN training."
    sbatch ${train}  # Train graph neural network
fi
