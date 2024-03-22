#!/usr/bin/env python3

import argparse
import datetime
import os
import subprocess
import sys

import yaml


# =============================================================================
# Helper functions
# =============================================================================
def log_progress(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{timestamp}] {message}")


def run_command(command, get_output=False):
    """Runs a shell command."""
    if get_output:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            shell=True,
        )
        return result.stdout.decode("utf-8").strip()
    else:
        subprocess.run(command, shell=True, check=True)


# =============================================================================
# Argument parsing
# =============================================================================
parser = argparse.ArgumentParser(
    description="Wrapper script for submitting jobs to SLURM."
)
# Add all the arguments from bash script here
parser.add_argument(
    "-q", "--experiment_yaml", help="Path to experiment YAML file", required=True
)
parser.add_argument(
    "-w", "--partition", help="Partition for SLURM scheduling", required=True
)
# Continue to add all other arguments here...
# Note: For flags that are true/false, use `action='store_true'`
parser.add_argument("-l", "--residual", help="Use residual", action="store_true")

# Add more arguments as per the bash script...

args = parser.parse_args()

# Set conda environment
os.system("module load anaconda3/2022.10")
os.system("conda activate /ocean/projects/bio210019p/stevesho/ogl")

# =============================================================================
# Main logic
# =============================================================================
experiment_yaml = args.experiment_yaml

# Parse experiment config from YAML
with open(experiment_yaml) as f:
    exp_config = yaml.safe_load(f)
working_directory = exp_config["working_directory"]
experiment_name = exp_config["experiment_name"]
tissues = exp_config["tissues"]

# Proceed with the rest of your script, converting bash commands into Python functions
# ...

# SLURM job submission example:
command = "sbatch --parsable ..."
job_id = run_command(command, get_output=True)

# Then you could pass job_id to subsequent jobs as dependencies using the --dependency flag

# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    # Your main script execution based on the commands that would have been the bash script
    pass
