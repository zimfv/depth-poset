#!/bin/bash

cd ..
mkdir -p logs

#SBATCH --job-name=depth_stats_of_random_alpha_complex
#SBATCH --output=logs/output_%A_%a.log
#SBATCH --error=logs/error_%A_%a.log

#SBATCH -c 1
#SBATCH --time=12:00:00
#SBATCH --mem=8G

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-799

start_n=12
end_n=148
step_n=8
start_d=1
end_d=4
repeat=10

params=()
for n in $(seq $start_n $step_n $end_n); do
    for d in $(seq $start_d $end_d); do
        for _ in $(seq 1 $repeat); do
            params+=("$d $n")
        done
    done
done

total_jobs=${#params[@]}

param="${params[$SLURM_ARRAY_TASK_ID]}"

source venv/bin/activate

python scripts/calculate_scores_on_random_alpha_complexes.py $param

deactivate
