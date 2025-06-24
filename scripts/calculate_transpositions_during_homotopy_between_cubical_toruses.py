# system imports
import sys
from pathlib import Path

# change root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# main imports
import inspect
import os

import argparse

import math
import pickle as pkl
import numpy as np
import pandas as pd

from src.complexes import CubicalTorusComplex
from src.utils import get_cross_parameters
from src.transpositions import Transposition

from src.profiling import Timer
from tqdm import tqdm


# define the similarity scores
from src import depth_poset_similarity_scores
similarity_scores = [obj for name, obj in inspect.getmembers(depth_poset_similarity_scores, inspect.isfunction) if inspect.getmodule(obj) == depth_poset_similarity_scores]


def read_torus_from_file(path: str) -> CubicalTorusComplex:
    # 
    with open(path, 'rb') as file:
        data = pkl.load(file)
    ctc = data['complex']
    return ctc


def get_flat_index(coords, dim, shape, add=True):
    """
    Преобразует многомерный индекс в плоский.
    """
    poly_shape = (math.comb(len(shape), dim), ) + shape
    flat_index = np.ravel_multi_index(coords, poly_shape)
    if add:
        flat_index += int(np.sum([math.comb(len(shape), i) for i in range(dim)])*np.prod(shape))
    return flat_index


def get_flat_add(dim, shape):
    """
    """
    return int(np.sum([math.comb(len(shape), i) for i in range(dim)])*np.prod(shape))


def collect_transpositions_during_homotopy(ctc0: CubicalTorusComplex, ctc1: CubicalTorusComplex, similarity_scores: list=[]) -> pd.DataFrame:
    """
    """
    if ctc0.shape != ctc1.shape:
        msg = f'The Cubical Torus Complexes should have the same shape, but ctc0.shape={ctc0.shape} and ctc1.shape={ctc1.shape}.'
        raise ValueError(msg)
    shape = ctc0.shape
    print(f'Collecting the transpositions during linear homotopy between 2 Cubical Torus Complexes shape {shape}.')

    with Timer() as timer:
        fvals0 = ctc0.filtration_values
        fvals1 = ctc1.filtration_values
        cross_parameters = [get_cross_parameters(m0.flatten(), m1.flatten()) for m0, m1 in zip(fvals0, fvals1)]
        eps = np.concatenate([m.flatten() for m in cross_parameters] + [[0, 1]])
        eps = np.unique(eps[~np.isnan(eps)])
        eps = 0.8*(eps[1:] - eps[:-1]).min()

        cross_parameters_id_flat = [np.unique(np.sort(np.argwhere(~np.isnan(i)), axis=1), axis=0) for i in cross_parameters]

        df_transpositions = pd.DataFrame()
        df_transpositions['id0_flat'] = np.concatenate(cross_parameters_id_flat)[:, 0]
        df_transpositions['id1_flat'] = np.concatenate(cross_parameters_id_flat)[:, 1]
        df_transpositions['dim'] = np.concatenate([d*np.ones(len(i), dtype=int) for d, i in enumerate(cross_parameters_id_flat)])
        df_transpositions['shape'] = df_transpositions['dim'].apply(lambda x: math.comb(len(shape), x))
        df_transpositions['shape'] = df_transpositions['shape'].apply(lambda x: (x, ) + shape)
        df_transpositions['id0_volume'] = df_transpositions.apply(lambda row: np.unravel_index(row['id0_flat'], row['shape']), axis=1)
        df_transpositions['id1_volume'] = df_transpositions.apply(lambda row: np.unravel_index(row['id1_flat'], row['shape']), axis=1)

        df_transpositions['time'] = df_transpositions.apply(lambda row: cross_parameters[row['dim']][row['id0_flat'], row['id1_flat']], axis=1)

        print(f'The time values and coresponding indices of transpositions have been found in {timer.elapsed():.4f} seconds.')
        timer.checkpoint()

        df_transpositions['id0_flat'] = df_transpositions.apply(lambda row: row['id0_flat'] + get_flat_add(row['dim'], shape), axis=1)
        df_transpositions['id1_flat'] = df_transpositions.apply(lambda row: row['id1_flat'] + get_flat_add(row['dim'], shape), axis=1)

        order = ctc0.get_order(sort_with_filtration=False)
        df_transpositions['cell0'] = df_transpositions['id0_flat'].apply(lambda i: order[i])
        df_transpositions['cell1'] = df_transpositions['id1_flat'].apply(lambda i: order[i])

        print(f'The cells have been defined in {timer.elapsed():.4f} seconds.')
        timer.checkpoint()

        tqdm.pandas(desc="Geting Complexes")
        df_transpositions['complex'] = df_transpositions.progress_apply(lambda row: CubicalTorusComplex(shape).assign_filtration([m0*(1 - (row['time'] - eps)) + m1*(row['time'] - eps) for m0, m1 in zip(fvals0, fvals1)]), axis=1)
        tqdm.pandas(desc="Calculating Orders")
        df_transpositions['order&dims'] = df_transpositions['complex'].progress_apply(lambda x: x.get_order(sort_with_filtration=True, return_filtration=False, return_dims=True))
        df_transpositions['order'] = df_transpositions['order&dims'].apply(lambda x: x[0])
        df_transpositions['dims'] = df_transpositions['order&dims'].apply(lambda x: x[1])
        df_transpositions = df_transpositions.drop(columns='order&dims')

        print(f'The complexes and the orders during homotopy have been found in {timer.elapsed():.4f} seconds.')
        timer.checkpoint()

        df_transpositions['id0_order'] = df_transpositions.apply(lambda row: row['order'].index(row['cell0']), axis=1)
        df_transpositions['id1_order'] = df_transpositions.apply(lambda row: row['order'].index(row['cell1']), axis=1)


        to_reverse = df_transpositions['id1_order'] - df_transpositions['id0_order'] == -1
        i0_vals = df_transpositions.loc[to_reverse, 'id0_order']
        i1_vals = df_transpositions.loc[to_reverse, 'id1_order']
        df_transpositions.loc[to_reverse, 'id0_order'] = i1_vals
        df_transpositions.loc[to_reverse, 'id1_order'] = i0_vals

        assert (df_transpositions['id1_order'] - df_transpositions['id0_order'] == 1).all()

        print(f'The orders during homotopy have been found in {timer.elapsed():.4f} seconds.')
        timer.checkpoint()

        tqdm.pandas(desc="Calculating Border Matrices")
        df_transpositions['border matrix'] = df_transpositions['complex'].progress_apply(lambda x: x.get_border_matrix(sort_with_filtration=True))

        print(f'The ordered border matrices during homotopy have been found in {timer.elapsed():.4f} seconds.')
        timer.checkpoint()

        tqdm.pandas(desc="Calculating Depth Posets")
        df_transpositions['dp'] = df_transpositions['complex'].progress_apply(lambda x: x.get_depth_poset(sort_with_filtration=True))

        print(f'The depth posets during homotopy have been found in {timer.elapsed():.4f} seconds.')
        timer.checkpoint()

        tqdm.pandas(desc="Compile Transpositions")
        df_transpositions['transposition'] = df_transpositions.progress_apply(
            lambda row: 
                Transposition(border_matrix=row['border matrix'], 
                              index0=row['id0_order'], 
                              index1=row['id1_order'], 
                              order=row['order'], 
                              dims=row['dims'], 
                              dp=row['dp']),
            axis=1
        )
        print(f'The transpositions themselves during homotopy have been found in {timer.elapsed():.4f} seconds.')
        timer.checkpoint()

        df_transpositions = df_transpositions.drop(columns=['dim'] + list(np.concatenate([[f'id{i}_flat', f'id{i}_volume', f'id{i}_order', f'cell{i}'] for i in range(2)])))
        df_transpositions = pd.concat([pd.DataFrame(df_transpositions['transposition'].apply(lambda tr: tr.to_dict()).to_list(), index=df_transpositions.index), df_transpositions], axis=1)

        # calculate the similarity scores
        for score in similarity_scores:
            score_vals = [score(dp0, dp1) for dp0, dp1 in zip(df_transpositions['dp'].values[:-1], df_transpositions['dp'].values[1:])]
            score_vals = np.append(None, score_vals)
            df_transpositions[score.__name__] = score_vals
            print(f'The similarity score {score.__name__} values for transpositions during homotopy have been found in {timer.elapsed():.4f} seconds.')
            timer.checkpoint()

    print(f'Everything is found in found in {timer.get_duration():.4f} seconds.')
    print(f'\ndf_transpositions.shape = {df_transpositions.shape}')
    return df_transpositions


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Collect the transpositions between during the homotopy 2 cubical toruses.")
    parser.add_argument("path0", help="Path to the first Cubical Torus Complex")
    parser.add_argument("path1", help="Path to the second Cubical Torus Complex")
    args = parser.parse_args()

    # get complexes
    ctc0 = read_torus_from_file(args.path0)
    ctc1 = read_torus_from_file(args.path1)

    # define complex indices
    complex_index0 = os.path.splitext(os.path.basename(args.path0))[0]
    complex_index1 = os.path.splitext(os.path.basename(args.path1))[0]

    # get dataframe of transpositions
    df_transpositions = collect_transpositions_during_homotopy(ctc0, ctc1, similarity_scores=similarity_scores)
    # add data about complex parameters
    df_transpositions.insert(0, 'complex_index0', complex_index0)
    df_transpositions.insert(1, 'complex_index1', complex_index1)
    df_transpositions.insert(2, 'complex_dim', ctc0.dim)
    df_transpositions.insert(3, 'complex_shape', [ctc0.shape]*len(df_transpositions))
    
	# create the directory if not exist and save file
    path = f"results/transpositions-during-linear-homotopy-between-barycentric-cubical-toruses/{complex_index0} and {complex_index1}.pkl"
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

	# save to file
    df_transpositions.to_pickle(path)
    print(f"The result is saved to path:\n{path}")



if __name__ == "__main__":
    main()