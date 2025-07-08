# system imports
import sys
from pathlib import Path

# change root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# main imports
import itertools
import inspect
import time
import os

import argparse

import numpy as np
import pickle as pkl

from src.complexes import CubicalTorusComplex
from src.depth import DepthPoset
from src import poset_scores, node_scores

from datetime import datetime

from src.profiling import Timer
from tqdm import tqdm


# select poset scores to check
#poset_scores_to_check = [obj for name, obj in inspect.getmembers(poset_scores, inspect.isfunction) if obj.__module__ == poset_scores.__name__]
poset_scores_to_check = [poset_scores.number_of_nodes, 
                         poset_scores.number_of_relations, 
                         poset_scores.number_of_components, 
                         poset_scores.number_of_minimal_nodes, 
                         poset_scores.number_of_maximal_nodes, 
                         poset_scores.number_of_edges_in_reduction, 
                         poset_scores.number_of_edges_in_closure,  
                         poset_scores.height, 
                         poset_scores.width, 
                         ]

# select nodes number and nodes scores to check
number_nodes_to_check = 16
#node_scores_to_check = [obj for name, obj in inspect.getmembers(node_scores, inspect.isfunction) if obj.__module__ == node_scores.__name__]
node_scores_to_check = []

def calculate_result_for_random_barycentric_fitration_on_cubical_torus(n, dim):
    # calculate the dictionary of values and scores for case of torus dimension dim builded from n^dim cubes.
    shape = tuple([int(n) for i in range(dim)])
    result = {'n': n, 'dim': dim, 'shape': shape}
    print(f'Calculate the scores for the barycentric model on the {dim}-dimensional torus with sides splited to {n} cells.')
    with Timer() as timer:
        # get a cubical complex with random filtration
        ctc = CubicalTorusComplex(shape=shape)
        ctc.assign_random_barycentric_filtration()
        result.update({'complex': ctc})
        print(f"Generated the {dim}-dimensional torus of shape-{shape} with random barycentric filtration in {timer.elapsed():.4f} seconds.")
        timer.checkpoint()

        # find depth poset
        depth_poset = ctc.get_depth_poset()
        result.update({'depth poset': depth_poset})
        number_of_nodes = len(depth_poset.nodes)
        number_of_edges = len(depth_poset.get_transitive_closure().edges)
        print(f"Found the depth poset for the given complex in {timer.elapsed():.4f} seconds.")
        print(f'There are {number_of_nodes} nodes and {number_of_edges} edges.')
        timer.checkpoint()

        # define full depth poset and subposets
        poset_dict = {'full': depth_poset}
        for sdim in range(dim):
            poset_dict.update({f'subposet dim={sdim}': depth_poset.subposet_dim(sdim)})
        poset_dict.update({'column reduction': depth_poset.get_column_bottom_to_top_reduction()})
        for sdim in range(dim):
            poset_dict.update({f'column reduction subposet dim={sdim}': poset_dict['column reduction'].subposet_dim(sdim)})
        poset_dict.update({'row reduction': depth_poset.get_row_left_to_right_reduction()})
        for sdim in range(dim):
            poset_dict.update({f'row reduction subposet dim={sdim}': poset_dict['row reduction'].subposet_dim(sdim)})
        print(f"Found the subposets of the depth poset in {timer.elapsed():.4f} seconds.")
        timer.checkpoint()

        # calculate poset score values
        print(f'Calculating the poset scores for the depth poset and its subposets...')
        poset_scores_values = []
        n_iters = len(poset_dict)*len(poset_scores_to_check)
        i_iter = 0
        #with tqdm(total=len(poset_dict)*len(poset_scores_to_check), desc="Poset scores") as pbar:
        with Timer() as timer1:
            for key, subposet in poset_dict.items():
                poset_scores_values_i = {'object': key}
                for score in poset_scores_to_check:
                    #pbar.set_postfix_str(f'Calculate {score.__name__.ljust(20)} for {key.ljust(16)}.')
                    #pbar.refresh()
                    score_value = score(subposet)
                    poset_scores_values_i.update({score.__name__: score_value})

                    i_iter += 1
                    print(f"Poset scores {i_iter}/{n_iters}: Calculated the score {score.__name__} for the object {key} in {timer1.elapsed():.4f} seconds.")
                    timer1.checkpoint()
                    #pbar.update()
                poset_scores_values.append(poset_scores_values_i)
        result.update({'poset scores': poset_scores_values})
        print(f"All poset scores have been calculated in {timer.elapsed():.4f} seconds.")
        timer.checkpoint()
        

        # choose the nodes to check and sort them by dimension
        nodes_to_check = np.random.choice(depth_poset.nodes, min(number_nodes_to_check, len(depth_poset.nodes)), replace=False)
        nodes_to_check = nodes_to_check[np.argsort([node.dim for node in nodes_to_check])]

        # calculate node scores
        print(f'Calculating the node scores for the depth poset and its subposets...')
        node_scores_values = []
        n_iters = len(nodes_to_check)*len(node_scores_to_check)*2*np.sum([key.find('dim=') == -1 for key in poset_dict.keys()])
        i_iter = 0
        #with tqdm(total=len(nodes_to_check)*len(node_scores_to_check)*2*np.sum([key.find('dim=') == -1 for key in poset_dict.keys()]), desc="Node scores") as pbar:
        with Timer() as timer2:
            for inode, node in enumerate(nodes_to_check):
                keys = [key for key in poset_dict.keys() if key.find('dim') == -1 or key.find(f'dim={node.dim}') != -1]
                for key in keys:
                    subposet = poset_dict[key]
                    node_scores_values_i = {'object': key, 'node': node}
                    for score in node_scores_to_check:
                        #pbar.set_postfix_str(f'Calculate {score.__name__.ljust(20)} for node {inode}/{len(nodes_to_check)} in {key.ljust(16)}.')
                        #pbar.refresh()
                        score_value = score(subposet, node)
                        node_scores_values_i.update({score.__name__: score_value})

                        i_iter += 1
                        print(f"Node scores {i_iter}/{n_iters}: Calculated the score {score.__name__} for the object {key} in {timer1.elapsed():.4f} seconds.")
                        timer1.checkpoint()
                        #pbar.update()
                    node_scores_values.append(node_scores_values_i)
        result.update({'node scores': node_scores_values})
        print(f"All node scores have been calculated in {timer.elapsed():.4f} seconds.")
        timer.checkpoint()
    
    print(f'Everything is found in found in {timer.get_duration():.4f} seconds.')

    return result

def main():
	# parse arguments
	parser = argparse.ArgumentParser(description="Check scores of Depth Poset for cubical complex with random barycentric filtration")
	parser.add_argument("dim", type=int, help="Dimension")
	parser.add_argument("n", type=int, help="Number of points in a cloud")
	args = parser.parse_args()

	# calculate the scores dataframe
	result = calculate_result_for_random_barycentric_fitration_on_cubical_torus(args.n, args.dim)

	# create the directory if not exist and save file
	path = f"results/scores-on-barycentric-cubical-toruses/{datetime.now()}.pkl"
	directory = os.path.dirname(path)
	if directory and not os.path.exists(directory):
		os.makedirs(directory)

	# save to file
	with open(path, "wb") as file:
		pkl.dump(result, file)
	print(f"The result is saved to path {path}")


if __name__ == "__main__":
	main()