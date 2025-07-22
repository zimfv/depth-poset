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
from scipy.sparse import csr_matrix

from src.complexes import CubicalTorusComplexExtended as CubicalTorusComplex
from src.complexes import AuxiliaryCell
from src.utils import get_cross_parameters
from src.transpositions import Transposition

from src.profiling import Timer
#from tqdm import tqdm


# define the similarity scores
from src import depth_poset_similarity_scores
#similarity_scores = [obj for name, obj in inspect.getmembers(depth_poset_similarity_scores, inspect.isfunction) if inspect.getmodule(obj) == depth_poset_similarity_scores]
similarity_scores = [
    depth_poset_similarity_scores.birth_relation_cell_similarity, 
    depth_poset_similarity_scores.death_relation_cell_similarity, 
    depth_poset_similarity_scores.poset_closure_arcs_cell_similarity, 
    depth_poset_similarity_scores.poset_reduction_arcs_cell_similarity,    
]

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
        cells0, dims0, fvals0 = ctc0.get_order(sort_with_filtration=False, return_filtration=True, return_dims=True)
        cells1, dims1, fvals1 = ctc1.get_order(sort_with_filtration=False, return_filtration=True, return_dims=True)
        assert list(dims0) == list(dims1)
        assert list(cells0) == list(cells1) 
        cells, dims = cells0, dims0
        del cells0, cells1, dims0, dims1

        print(f'len(fvals0) = {len(fvals0)}')
        print(f'len(fvals1) = {len(fvals1)}')
        
        # find the cross_parameters (times) and coresponding filtration values
        cross_parameters = get_cross_parameters(fvals0, fvals1, filter_outside=True)
        cross_parameters[np.tril_indices_from(cross_parameters)] = np.nan
        print(f'cross_parameters.shape = {cross_parameters.shape}')
        cross_values = (1 - cross_parameters)*fvals0 + cross_parameters*fvals1
        print(f'cross_values.shape = {cross_values.shape}')
        

        # find the indices of non-filtered cells and sort them by time 
        transposition_indices = np.argwhere(~np.isnan(cross_parameters))
        transposition_times = cross_parameters[transposition_indices[:, 0], transposition_indices[:, 1]]
        transposition_indices = transposition_indices[np.argsort(transposition_times)]
        
        print(f'The indices of the transposing cells have been found and sorted in {timer.elapsed():.4f} seconds.')
        print(f'There will be {len(transposition_indices)} transpositions.')
        timer.checkpoint()

        # define the initial order, dims and border matrix
        current_order, current_dims = ctc0.get_order(sort_with_filtration=True, return_filtration=False, return_dims=True)
        current_border_matrix = csr_matrix(ctc0.get_border_matrix(sort_with_filtration=True, dtype=int))

        # define the initial depth poset
        dp_next = ctc0.get_depth_poset()

        # consecutively define the transpositions
        df = []
        for i, (i0, i1) in enumerate(transposition_indices):
            transposition = Transposition(border_matrix=current_border_matrix, 
                                          index0=current_order.index(cells[i0]), 
                                          index1=current_order.index(cells[i1]), 
                                          order=current_order, dims=current_dims)
            df.append(
                {
                    'time': cross_parameters[i0, i1], 
                    'value': cross_values[i0, i1], 
                }
            )
            df[-1].update(transposition.to_dict())

            # compare depth posets
            if len(similarity_scores) > 0:
                # if I am sure, that it's correct
                dp_current, dp_next = dp_next, transposition.next_depth_poset()
                for score in similarity_scores:
                    with Timer() as timer_score:
                        val = score(dp_current, dp_next)
                        df[-1].update({score.__name__: val})
                    print(f'{i}/{len(transposition_indices)} - The similarity score {score} have been found in {timer_score.duration:.4f} seconds.')

            # update the order, dims and border matrix
            current_order = transposition.next_order()
            current_dims = transposition.next_dims()
            current_border_matrix = transposition.next_border_matrix()
        print(f'{len(df)} transpositions were found in {timer.elapsed():.4f} seconds.')
    
    df = pd.DataFrame(df)
    print(f'So the total duration of seeking transpositions info was {timer.duration:.4f} seconds.')
    return df

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Collect the transpositions between during the homotopy 2 cubical toruses.")
    parser.add_argument("path0", help="Path to the first Cubical Torus Complex")
    parser.add_argument("path1", help="Path to the second Cubical Torus Complex")
    args = parser.parse_args()
    
    print(f'The input files are:\npath0="{args.path0}"\npath1="{args.path1}"\n')

    # get complexes
    ctc0 = read_torus_from_file(args.path0)
    ctc1 = read_torus_from_file(args.path1)

    # define complex indices
    complex_index0 = os.path.splitext(os.path.basename(args.path0))[0]
    complex_index1 = os.path.splitext(os.path.basename(args.path1))[0]

    with Timer() as timer:
        # collect transpositions
        df = collect_transpositions_during_homotopy(ctc0, ctc1, similarity_scores=similarity_scores)
    
	# create the directory if not exist and save file
    path = f"results/transpositions-during-linear-homotopy-between-extended-barycentric-cubical-toruses/{complex_index0} and {complex_index1}.pkl"
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

	# save to file
    with open(path, 'wb') as file:
        pkl.dump(
            {
                'complex_index0': complex_index0, 
                'complex_index1': complex_index1, 
                'complex_dim': ctc0.dim, 
                'complex_shape': ctc0.shape, 
                'calculation time': timer.duration,
                'transpositions': df, 
                #'depth posets': dps,
            }, file
        )
    print(f"The result is saved to path:\n{path}")

    # show the size of the result
    size = os.path.getsize(path)
    print(f'The size of the result file is: {size} bytes.')


if __name__ == "__main__":
    main()