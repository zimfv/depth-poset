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

import numpy as np
import pickle as pkl

from gudhi import AlphaComplex, SimplexTree
from src.depth import DepthPoset
from src import poset_scores, node_scores

from tqdm import tqdm
from datetime import datetime


ns = np.arange(4, 13, 8)
dims = np.arange(1, 4)
repeat = 10
number_nodes_to_check = 10

path_template = "results/scores-on-random-alpha-complexes/{0}.pkl"

poset_scores_to_check = [
    poset_scores.number_of_nodes, 
    poset_scores.number_of_minimal_nodes, 
    poset_scores.number_of_maximal_nodes, 
    poset_scores.height, 
    poset_scores.width, 
    poset_scores.minimum_maximal_chain, 
    poset_scores.avarage_maximal_chain, 
]

node_scores_to_check = [
    node_scores.incomparable_number, 
    node_scores.incestors_number, 
    node_scores.incestors_height, 
    node_scores.incestors_width, 
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

	# generate SimplexTree
	stree = AlphaComplex(points).create_simplex_tree()
	#result.update({'stree': stree}) # should we save SimplexTree?

	# find depth poset
	depth_poset = DepthPoset.from_simplex_tree(stree)
	result.update({'depth poset': depth_poset})	

	# find poset scores for full depth poset
	for score in poset_scores_to_check:
		score_value = score(depth_poset)
		result.update({score.__name__: {"full": score_value}})

	# find poset scores for subposets of different dimensions
	for sdim in range(dim):
		subposet = depth_poset.subposet_dim(sdim)
		for score in poset_scores_to_check:
			score_value = score(subposet)
			result[score.__name__].update({sdim: score_value})

	# choose the nodes to check and sort them by dimension
	nodes_to_check = np.random.choice(depth_poset.nodes, min(number_nodes_to_check, len(depth_poset.nodes)), replace=False)
	nodes_to_check = nodes_to_check[np.argsort([node.dim for node in nodes_to_check])]
	

	for score in node_scores_to_check:
		result.update({score.__name__: {node: {} for node in nodes_to_check}})

	# calculate node scores for the full poset
	for node in nodes_to_check:
		for score in node_scores_to_check:
			score_value = score(depth_poset, node)
			result[score.__name__][node].update({'full': score_value})


	# calculate node scores for subposets
	sdim = -1
	for node in nodes_to_check:
		if node.dim != sdim:
			sdim = node.dim
			subposet = depth_poset.subposet_dim(sdim)
			for score in node_scores_to_check:
				score_value = score(subposet, node)
				result[score.__name__].update({node: {sdim: score_value}})
	return result


with tqdm(total=len(ns)*len(dims)*repeat) as pbar:
	for n, dim in itertools.product(ns, dims):
		for i in range(repeat):
			result = calculate_result_for_random_alpha_complex(n, dim)

			# create the directory if not exist and save file
			path = path_template.format(datetime.now())
			directory = os.path.dirname(path)
			if directory and not os.path.exists(directory):
				os.makedirs(directory)

			# save file
			with open(path, "wb") as file:
				pkl.dump(result, file)

			pbar.update()