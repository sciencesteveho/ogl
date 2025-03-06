#!/bin/bash

#SBATCH --job-name=filter_hicdc
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of cores requested
#SBATCH --ntasks-per-node=8

# Partition
#SBATCH -p RM-shared

# Request time
#SBATCH -t 48:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

# tissue=$1  # name of tissue or cell line
# binsize=$2  # resolution of .hic to work with

# tissue=liver
# binsize=40000

# =============================================================================
# Filter using HiCDC+
# =============================================================================

# working_dir=/Users/steveho/hic
# script_dir=/Users/steveho/omics_graph_learning/omics_graph_learning/discretionary_preprocessing

# function run_hicdcplus () {
#     local tissue=$1
#     local binsize=$2
#     local gen_ver="hg38"

#     /Library/Frameworks/R.framework/Resources/bin/Rscript ${script_dir}/hicdcplus.r \
#         --tissue $1 \
#         --working_dir $working_dir \
#         --gen_ver $gen_ver \
#         --binsize $binsize \
#         --ncore 4
# }


# for cell in gm12878 h1-esc hepg2 hmec imr90 k562 nhek adrenal aorta hippocampus liver left_ventricle lung ovary pancreas skeletal_muscle small_intestine spleen; do
#     case "$cell" in
#         gm12878|h1-esc|hepg2|hmec|imr90|k562|nhek)
#             resolution=5000
#             ;;
#         adrenal|aorta|hippocampus|liver|left_ventricle|lung|ovary|pancreas|skeletal_muscle|small_intestine|spleen)
#             resolution=40000
#             ;;
#         *)
#             echo "Invalid cell type: $cell"
#             continue
#             ;;
#     esac

#     echo "Running hicdcplus for $cell with resolution $resolution"
#     # run_hicdcplus $cell $resolution  <-- Uncomment this line to run the script
# done

# =============================================================================
# Write files out to separate chromosomes
# =============================================================================
# Set your qvalue cutoff here
tissue=$1
QVALUE_CUTOFF=$2

working_dir=/ocean/projects/bio210019p/stevesho/raw_tissue_hic/contact_matrices
INPUT_FILE=${working_dir}/fdr_filtered/${tissue}_result.txt
FILE_PREFIX=${working_dir}/fdr_filtered/${tissue}/${tissue}

# Make the directory to place filtered files if it doesn't exist
mkdir -p ${working_dir}/fdr_filtered/${tissue}

# Create an array to keep track of which chromosome files have been
# written
tail -n +2 "$INPUT_FILE" \
    | awk -v cutoff="$QVALUE_CUTOFF" -v prefix="$FILE_PREFIX" '$10 < cutoff' \
    | sort -k1,1 -k2,2n \
    | cut -f1,2,3,4,5,6 \
    > ${working_dir}/fdr_filtered/${tissue}/${tissue}_${QVALUE_CUTOFF}_filtered.txt

echo "$tissue hi-c contacts filtered at q-val = $QVALUE_CUTOFF."


# q_vals=(0.1 0.01 0.001)
# for tissue in h1-esc hepg2 hmec imr90 k562 nhek gm12878 adrenal aorta hippocampus liver left_ventricle lung ovary pancreas skeletal_muscle small_intestine spleen; do
#     for qval in "${q_vals[@]}"; do
#         sbatch filter_hicdc_qvals.sh $tissue $qval
#     done
# done

# declare -A chr_written
# tail -n +2 "$INPUT_FILE" | awk -v cutoff="$QVALUE_CUTOFF" -v prefix="$FILE_PREFIX" '{
#     chrFile = prefix "_" $1 "_" cutoff ".tsv"
#     if ($10 < cutoff) {
#         if (!(chrFile in chr_written)) {
#             print header > chrFile
#             chr_written[chrFile] = 1
#         }
#         print >> chrFile
#     }
# }' header="$HEADER"

# echo "Files have been created for each chromosome with qvalue below $QVALUE_CUTOFF."

# for qval in 0.1 0.01 0.001; do
#     sbatch filter_hicdc.sh $qval leftventricle_result.txt leftventricle
# done

# # =============================================================================
# # Concat to get counts
# # =============================================================================
# Define arrays or space-separated strings of all unique prefixes and q-values
# prefixes=("leftventricle")  # Add all possible prefixes
# prefixes=("k562")  # Add all possible prefixes
# qvals=("0.001" "0.01")  # Add all possible q-values

# # Loop through each prefix
# for prefix in "${prefixes[@]}"; do
#     # Loop through each q-value
#     for qval in "${qvals[@]}"; do
#         output_file="${prefix}_all_chr_${qval}.tsv"
#         # Loop through files and concatenate without headers
#         for file in ${prefix}_chr*_"${qval}".tsv; do
#             tail -n +2 "$file"  # Skip header of every file
#         done | sort -k1,1 -k2,2n -V | cut -f1,2,3,4,5,6 > "${output_file}"
#     done
# done
# # Add executable permissions to the script if needed


# For posterity, the follow was used to filter the hic results
# for tis in adrenal aorta hippocampus liver left_ventricle liver lung ovary pancreas skeletal_muscle small_intestine spleen;
# do
#     sbatch filter_hicdc_tissues.sh $tis 40000
# done

# for cell in gm12878 h1-esc hepg2 hmec imr90 k562 nhek;
# do
#     sbatch filter_hicdc.sh $cell 5000
# done
