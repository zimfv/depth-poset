#!/bin/bash
#SBATCH --ntasks=1
#
#SBATCH --job-name=BCTE-homotopy
#SBATCH --output=logs/transpositions_during_homotopies_between_extended_cubical_toruses/output_%j.log
#
#SBATCH --time=72:00:00
#SBATCH --mem=16G

echo "SLURM_ARRAY_TASK_ID = '$SLURM_ARRAY_TASK_ID'"

param_file="params/transpositions_during_homotopies_between_extended_cubical_toruses.txt"
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$param_file")
echo "LINE: $line"

eval "set -- $line"
path0=$1
path1=$2
echo "PARAMS:"
echo "path0=$path0"
echo "path1=$path1"

echo "DATETIME:"
date

source venv/bin/activate

python scripts/calculate_transpositions_during_homotopy_between_extended_cubical_toruses_facilitated.py "$path0" "$path1"

deactivate
