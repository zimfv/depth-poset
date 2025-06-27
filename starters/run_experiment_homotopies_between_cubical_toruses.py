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

# import not-defult packages
import numpy as np
import pandas as pd
import pickle as pkl

# execution_parameters
run_slurm = True
run_native = False

max_cases_per_size = None
max_number_of_cells = None

both_directions = False

# define file paths
directory = "results/scores-on-barycentric-cubical-toruses"
paths = np.sort([f'{directory}/{f}' for f in os.listdir(directory)])

paths = paths[pd.Series(paths).apply(lambda s: s.split('.')[0].split('/')[-1]).values > '2024-12-26 18:04:00']

# load the cases
df_cases = []
for path in paths:
    with open(path, 'rb') as file:
        df_cases.append(pkl.load(file))
        df_cases[-1].update({'path': path})

df_cases = pd.DataFrame(df_cases)
df_cases = df_cases[['dim', 'n', 'path']]
df_cases = df_cases.groupby(['dim', 'n']).agg(list)

print(f'Torus Sizes/Dimensions Distribution:')
print(df_cases.map(len).reset_index(drop=False).pivot_table(columns='n', index='dim', values='path').fillna(0).astype(int))

# define the pairs of cases
if both_directions:
    df_pairs = df_cases.map(lambda l: [(i0, i1) for i0, i1 in itertools.product(l, repeat=2) if i0 != i1])
else:
    df_pairs = df_cases.map(lambda l: [(i0, i1) for i0, i1 in itertools.combinations(l, 2)])
df_pairs = df_pairs.explode('path', ignore_index=False).reset_index(drop=False)
df_pairs = df_pairs[~pd.isna(df_pairs['path'])]
df_pairs['input0'] = df_pairs['path'].apply(lambda x: x[0])
df_pairs['input1'] = df_pairs['path'].apply(lambda x: x[1])
df_pairs = df_pairs.drop(columns='path')

# filter pairs, which are already calculated 
df_pairs['index0'] = df_pairs['input0'].apply(lambda s: os.path.splitext(os.path.basename(s))[0])
df_pairs['index1'] = df_pairs['input1'].apply(lambda s: os.path.splitext(os.path.basename(s))[0])
df_pairs['result'] = df_pairs.apply(lambda row: f'{row["index0"]} and {row["index1"]}.pkl', axis=1)
try:
    files_exist = os.listdir('results/transpositions-during-linear-homotopy-between-barycentric-cubical-toruses')
except FileNotFoundError:
    files_exist = []
df_pairs = df_pairs[~df_pairs['result'].isin(files_exist)]
df_pairs = df_pairs.drop(columns=['index0', 'index1', 'result'])

if max_cases_per_size is not None:
    df_pairs = [df_pairs[(df_pairs['dim'] == dim)&(df_pairs['n'] == n)] for dim, n in df_pairs[['dim', 'n']].drop_duplicates().values]
    for i in range(len(df_pairs)):
        if len(df_pairs[i]) > max_cases_per_size:
            df_pairs[i] = df_pairs[i].loc[np.random.choice(df_pairs[i].index.values, max_cases_per_size, replace=False)]
    df_pairs = pd.concat(df_pairs)

if max_number_of_cells is not None:
    df_pairs = df_pairs[(2*df_pairs['n'])**df_pairs['dim'] <= max_number_of_cells]

# sort by the number of cells
number_of_cells = (2 * df_pairs['n']) ** df_pairs['dim']
df_pairs = df_pairs.loc[number_of_cells.sort_values().index]

print(f'\nThe distribution of {len(df_pairs)} pairs to calculate homotopies:')
print(df_pairs.groupby(['dim', 'n'])['input0'].count().reset_index().pivot_table(columns='n', index='dim', values='input0').fillna(0).astype(int))

params_txt = '\n'.join([f'"{path0}" "{path1}"' for path0, path1 in df_pairs[['input0', 'input1']].values]) + '\n'

# save params to file
if not os.path.exists('params'):
    os.makedirs('params')
with open('params/transpositions_during_homotopies_between_cubical_toruses.txt', 'w') as file:
    file.write(params_txt)

# create logs directory if not exists
if not os.path.exists('logs/transpositions_during_homotopies_between_cubical_toruses'):
    os.makedirs('logs/transpositions_during_homotopies_between_cubical_toruses')


# Submit the SLURM array job and capture output
if run_slurm:
    result = subprocess.run(
        ['sbatch', f'--array=0-{len(df_pairs) - 1}', 'slurm/calculate_transpositions_between_cubical_toruses.sh'],
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

# Submit the jobs in native way
if run_native:
    for i, (input0, input1) in enumerate(df_pairs[['input0', 'input1']].values):
        print(f'Runing Process {i + 1}/{len(df_pairs)}')
        result = subprocess.run(
            ['python', 'scripts/calculate_transpositions_during_homotopy_between_cubical_toruses.py', input0, input1],
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
