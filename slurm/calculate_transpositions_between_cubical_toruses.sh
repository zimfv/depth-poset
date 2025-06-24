#!/bin/bash
#SBATCH --ntasks=1
#
#SBATCH --job-name=transpositions_during_homotopies_between_cubical_toruses
#SBATCH --output=logs/transpositions_during_homotopies_between_cubical_toruses/output_%j.log
#
#SBATCH --time=96:00:00
#SBATCH --mem=8G

param_file="params/scores_on_cubical_torus.txt"

IFS= read -r param << (sed -n "${SLURM_ARRAY_TASK_ID}p" "$param_file")

eval set -- $param
path0="$1"
path1="$2"

source venv/bin/activate

python scripts/calculate_transpositions_during_homotopy_between_cubical_toruses.py "$path0" "$path1"

deactivate
