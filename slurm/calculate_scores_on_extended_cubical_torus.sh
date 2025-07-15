#!/bin/bash
#SBATCH --ntasks=1
#
#SBATCH --job-name=DepthStatsRCTE
#SBATCH --output=logs/scores_on_extended_cubical_torus/output_%j.log
#
#SBATCH --time=72:00:00
#SBATCH --mem=16G

echo "SLURM_ARRAY_TASK_ID = '$SLURM_ARRAY_TASK_ID'"

param_file="params/scores_on_extended_cubical_torus.txt"
param=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $param_file)

dim=$(echo $param | cut -d ' ' -f 1)
n=$(echo $param | cut -d ' ' -f 2)
echo "PARAMS:"
echo "dim=$dim"
echo "n=$n"

source venv/bin/activate

python -u scripts/calculate_scores_on_extended_cubical_torus.py $dim $n

deactivate
