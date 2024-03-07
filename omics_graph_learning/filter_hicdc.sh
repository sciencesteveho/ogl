#!/bin/bash

#SBATCH --job-name=filter_hicdc
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


# sort --parallel=8 -S 80%

# Set your qvalue cutoff here
QVALUE_CUTOFF=$1
INPUT_FILE=$2
FILE_PREFIX=$3

# Assuming the input file is named "input.tsv"
# and the first line of the file contains headers.
HEADER=$(head -n 1 "$INPUT_FILE")

# Create an associative array to keep track of which chromosome files have been written
declare -A chr_written

tail -n +2 "$INPUT_FILE" | awk -v cutoff="$QVALUE_CUTOFF" -v prefix="$FILE_PREFIX" '{
    chrFile = prefix "_" $1 "_" cutoff ".tsv"
    if ($10 < cutoff) {
        if (!(chrFile in chr_written)) {
            print header > chrFile
            chr_written[chrFile] = 1
        }
        print >> chrFile
    }
}' header="$HEADER"

echo "Files have been created for each chromosome with qvalue below $QVALUE_CUTOFF."