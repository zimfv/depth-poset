# system imports
import sys
from pathlib import Path

# change root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# main imports
import os
import subprocess
import itertools


# define params
repeat_params = [
    {'dim': 2, 'ns': [32], 'repeat': 3},
    {'dim': 2, 'ns': [36, 40], 'repeat': 10},
    {'dim': 3, 'ns': [6], 'repeat': 7},
    {'dim': 3, 'ns': [7, 8, 9, 10, 11, 12], 'repeat': 10},
]
dims = list(itertools.chain.from_iterable([[params_dict['dim'] for _ in range(len(params_dict['ns'])*params_dict['repeat'])] for params_dict in repeat_params]))
ns = list(itertools.chain.from_iterable([[n for n, _ in itertools.product(params_dict['ns'], range(params_dict['repeat']))] for params_dict in repeat_params]))

params_txt = '\n'.join([f'{dim} {n}' for dim, n in zip(dims, ns)]) + '\n'

# save params to file
if not os.path.exists('params'):
    os.makedirs('params')
with open('params/scores_on_cubical_torus.txt', 'w') as file:
    file.write(params_txt)

# create logs directory if not exists
if not os.path.exists('logs/scores_on_cubical_torus'):
    os.makedirs('logs/scores_on_cubical_torus')

# Submit the SLURM array job and capture output
result = subprocess.run(
    ['sbatch', f'--array=0-{len(ns) - 1}', 'slurm/calculate_scores_on_cubical_torus.sh'],
    capture_output=True,
    text=True
)

# Check for errors
if result.returncode != 0:
    print("Error submitting job:")
    print(result.stderr)
else:
    print("Submission output:")
    print(result.stdout)
