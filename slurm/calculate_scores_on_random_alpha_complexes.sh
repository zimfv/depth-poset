#!/bin/bash

mkdir -p logs

#SBATCH --job-name=depth_stats_of_random_alpha_complex
#SBATCH --output=logs/output_%j.log

#SBATCH -c 1
#SBATCH --time=12:00:00
#SBATCH --mem=8G

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

param_file="params.txt"
param=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $param_file)

dim=$(echo $param | cut -d ' ' -f 1)
n=$(echo $param | cut -d ' ' -f 2)

source venv/bin/activate

python scripts/calculate_scores_on_random_alpha_complexes.py $dim $n

deactivate
