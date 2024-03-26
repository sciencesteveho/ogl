#!/bin/bash

#SBATCH --job-name=mat_to_hic
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of cores requested
#SBATCH --ntasks-per-node=16

# Partition
#SBATCH -p RM-shared

# Request time
#SBATCH -t 24:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x

# =============================================================================
# Set up variables
# =============================================================================
module load anaconda3
conda activate /jet/home/stevesho/.conda/envs/hiclift

tissue=$1
juicer=/ocean/projects/bio210019p/stevesho/hic/juicer_tools.3.0.0.jar
run_juicer="java -Xms512m -Xmx2048m -jar ${juicer}"
hg38_chrom_sizes=/ocean/projects/bio210019p/stevesho/resources/hg38.chrom.sizes.txt
src_dir=/ocean/projects/bio210019p/stevesho/data/preprocess/omics_graph_learning/omics_graph_learning/discretionary_preprocessing
base_dir="/ocean/projects/bio210019p/stevesho/data/preprocess/raw_files/chromatin_loops/hic"
resource_dir="/ocean/projects/bio210019p/stevesho/resources"

# Append specific directory names to the base directory
working_dir="${base_dir}/coolers"
final_dir="${base_dir}/contact_matrices"
tmp_dir="${working_dir}/tmp" 

# Set-up array for tissue names
declare -A tissue_names

# Assign values to the array
tissue_names[aorta]="AO"
tissue_names[hippocampus]="HC"
tissue_names[lung]="LG2"
tissue_names[pancreas]="PA2"
tissue_names[skeletal_muscle]="PO1"
tissue_names[small_intestine]="SB"
tissue_names[liver]="LI"
tissue_names[left_ventricle]="LV"

# =============================================================================
# Convert individual chr matrices to unified cooler
# =============================================================================
echo "Tissue: $tissue"
echo "Tissue name from array: ${tissue_names[$tissue]}"
python ${src_dir}/mat_to_cooler.py -t "${tissue_names[$tissue]}" -o ${tissue}
echo "Python script exit status: $?"

# # =============================================================================
# # Liftover cooler to hg38
# # =============================================================================
HiCLift \
    --input ${tmp_dir}/${tissue}.cool \
    --input-format cooler \
    --out-pre ${working_dir}/${tissue}_hg38 \
    --output-format hic \
    --chain-file ${resource_dir}/hg19ToHg38.over.chain \
    --out-chromsizes ${resource_dir}/hg38.chrom.sizes.autosomes.txt \
    --in-assembly hg19 \
    --out-assembly hg38 \
    --memory 28G \
    --nproc 12 \
    --logFile ${tmp_dir}/hiclift.log

echo "Finished liftover"

# # # =============================================================================
# # # Convert cooler to hic
# # # Adapted from Charlotte West: https://www.biostars.org/p/360254/
# # # =============================================================================
conda activate /jet/home/stevesho/.conda/envs/deeploop
function cool_to_hic () {
    local tissue=$1
    local savedir=$final_dir  # Assuming savedir should point to final_dir.

    # Convert cooler to ginteractions format
    hicConvertFormat -m "${tmp_dir}/${tissue}.cool" \
        --outFileName "${tmp_dir}/${tissue}.ginteractions" \
        --inputFormat cool \
        --outputFormat ginteractions

    # Add dummy variables
    awk -F "\t" '{print 0, $1, $2, 0, 0, $4, $5, 1, $7}' "${tmp_dir}/${tissue}.ginteractions" > "${tmp_dir}/${tissue}_hg38.ginteractions.short"

    # Sort by chromosomes
    sort -k2,2d -k6,6d "${savedir}/${tissue}_hg38.ginteractions.short" > "${savedir}/${tissue}_hg38.ginteractions.short.sorted"

    # Convert ginteractions to hic file using the juicer pre command
    $run_juicer pre -r 40000 "${savedir}/${tissue}_hg38.ginteractions.short.sorted" "${savedir}/${tissue}.hic" "${hg38_chrom_sizes}"
}

cool_to_hic ${tissue}