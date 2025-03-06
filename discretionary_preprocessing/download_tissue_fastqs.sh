#!/bin/bash

#SBATCH --job-name=download_sra
#SBATCH --mail-user=stevesho@umich.edu
#SBATCH --account=bio210019p
#SBATCH --mail-type=FAIL

# Number of cores requested
#SBATCH --ntasks-per-node=4

# Partition
#SBATCH -p RM-shared

# Request time
#SBATCH -t 48:00:00

# output to a designated folder
#SBATCH -o slurm_outputs/%x_%j.out

#echo commands to stdout
set -x


bwamem=/ocean/projects/bio210019p/stevesho/resources/software/bwa-0.7.17/bwa
module load sra-toolkit/2.10.9

# =============================================================================
# Setting up variables to track time
# =============================================================================
SECONDS=0

function convertsecs() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d\n" $hours $minutes $seconds
}


# Function to echo script progress to stdout
log_progress() {
    echo -e "[$(date +%Y-%m-%dT%H:%M:%S%z)] $1"
}

# =============================================================================
# Downloading SRA data
# =============================================================================
# Accession numbers
# Add ovary,
accessions=(
    "SRR4271997" "SRR4271998" "SRR4271999" "SRR4272000" # Hippocampus
    "SRR4272004" "SRR4272005" "SRR4272006" # Lung
    "SRR4272015" "SRR4272016" # Pancreas
    "SRR4272018" "SRR4272019" "SRR4272020" # Psoas
    "SRR4272031" "SRR4272032" "SRR4272033" "SRR4272034" # Small_intestine
    "SRR1501147" "SRR1501148" "SRR1501149" "SRR1501150" "SRR1501151" # Liver
    "SRR1501138" "SRR1501139" "SRR1501140" "SRR1501141" # Aorta
    "SRR1501142" "SRR1501143" "SRR1501144" "SRR1501145" "SRR1501146" # Left_ventricle
    "SRR4272035" "SRR4272036" "SRR4272037" # Spleen
    "SRR4272007" "SRR4272008" "SRR4272009" "SRR4272010" # Ovary
    "SRR4271980" "SRR4271981" "SRR4271982" "SRR4271983" # Adrenal
)

# Set the output directory
outdir="sra_data"
mkdir -p "$outdir"

# =============================================================================
# Download one accession
# =============================================================================
download_sra() {
    local acc="$1"  # Accession number
    echo "Downloading $acc..."
    fastq-dump --split-files  --outdir "$outdir" --gzip --skip-technical --readids --dumpbase --split-files --clip "$acc"
}

# =============================================================================
# Uncomment the following line to download all accessions
# =============================================================================
download_sra $1
log_progress "Downloaded $1 in $(convertsecs $SECONDS)"
