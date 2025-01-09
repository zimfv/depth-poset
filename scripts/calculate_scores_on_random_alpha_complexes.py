# system imports
import sys
from pathlib import Path

# change root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# main imports
import itertools
import time
import os

import argparse

import numpy as np
import pickle as pkl

from gudhi import AlphaComplex, SimplexTree
from src.depth import DepthPoset
from src import poset_scores, node_scores

from datetime import datetime

from tqdm import tqdm


# select poset scores to check
poset_scores_to_check = [
    poset_scores.number_of_nodes, 
    poset_scores.number_of_minimal_nodes, 
    poset_scores.number_of_maximal_nodes, 
    poset_scores.height, 
    poset_scores.width, 
    poset_scores.minimum_maximal_chain, 
    poset_scores.avarage_maximal_chain, 
]


# select nodes number and nodes scores to check
number_nodes_to_check = 16
node_scores_to_check = [
    node_scores.incomparable_number, 
    node_scores.ancestors_number, 
    node_scores.ancestors_height, 
    node_scores.ancestors_width, 
    node_scores.successors_number, 
    node_scores.successors_height, 
    node_scores.successors_width, 
]


def calculate_result_for_random_alpha_complex(n, dim):
	# calculate the dictionary of values and scores for case with n points of given dimension
	result = {'n': n, 'dim': dim}

	# generate a cloud of points
	points = np.random.random([n, dim])
	result.update({'points': points})
	print(f"Generated the cloud of n={n} points dim={dim}.")

	# generate SimplexTree
	stree = AlphaComplex(points).create_simplex_tree()
	result.update({'stree': stree}) # should we save SimplexTree?
	print(f"Found the simplicial complex as SimplexTree structure.")


	# find depth poset
	depth_poset = DepthPoset.from_simplex_tree(stree)
	result.update({'depth poset': depth_poset})	
	print(f"Found the depth poset for given simplicial complex.")

	# define full septh poset and subposets
	poset_dict = {'full': depth_poset}
	for sdim in range(dim):
		poset_dict.update({f'subposet dim={sdim}': depth_poset.subposet_dim(sdim)})

	# calculate poset score values
	poset_scores_values = []
	with tqdm(total=len(poset_dict)*len(poset_scores_to_check), desc="Poset scores") as pbar:
		for key, subposet in poset_dict.items():
			poset_scores_values_i = {'object': key}
			for score in poset_scores_to_check:
				pbar.set_postfix_str(f'Calculate {score.__name__.ljust(20)} for {key.ljust(16)}.')
				pbar.refresh()
				score_value = score(subposet)
				poset_scores_values_i.update({score.__name__: score_value})
				pbar.update()
			poset_scores_values.append(poset_scores_values_i)
	result.update({'poset scores': poset_scores_values})

	# choose the nodes to check and sort them by dimension
	nodes_to_check = np.random.choice(depth_poset.nodes, min(number_nodes_to_check, len(depth_poset.nodes)), replace=False)
	nodes_to_check = nodes_to_check[np.argsort([node.dim for node in nodes_to_check])]

	# calculate node scores
	node_scores_values = []
	with tqdm(total=len(nodes_to_check)*len(node_scores_to_check)*2, desc="Node scores") as pbar:
		for inode, node in enumerate(nodes_to_check):
			for key in ["full", f"subposet dim={node.dim}"]:
				subposet = poset_dict[key]
				node_scores_values_i = {'object': key, 'node': node}
				for score in node_scores_to_check:
					pbar.set_postfix_str(f'Calculate {score.__name__.ljust(20)} for node {inode}/{len(nodes_to_check)} in {key.ljust(16)}.')
					pbar.refresh()
					score_value = score(subposet, node)
					node_scores_values_i.update({score.__name__: score_value})
					pbar.update()
				node_scores_values.append(node_scores_values_i)
	result.update({'node scores': node_scores_values})
	return result



def main():
	# parse arguments
	parser = argparse.ArgumentParser(description="Check scores of Depth Poset for random cloud of points.")
	parser.add_argument("dim", type=int, help="Dimension")
	parser.add_argument("n", type=int, help="Number of points in a cloud")
	args = parser.parse_args()

	# calculate the scores dataframe
	result = calculate_result_for_random_alpha_complex(args.n, args.dim)

	# create the directory if not exist and save file
	path = f"results/scores-on-random-alpha-complexes/{datetime.now()}.pkl"
	directory = os.path.dirname(path)
	if directory and not os.path.exists(directory):
		os.makedirs(directory)

	# save to file
	with open(path, "wb") as file:
		pkl.dump(result, file)
	print(f"The result is saved to path {path}")


if __name__ == "__main__":
	main()