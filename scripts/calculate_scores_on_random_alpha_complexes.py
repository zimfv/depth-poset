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


ns = np.arange(4, 13, 8)
dims = np.arange(1, 3)

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

]

with tqdm(total=len(ns)*len(dims)) as pbar:
	for n, dim in itertools.product(ns, dims):
		result = {'n': n, 'dim': dim}

		# generate a cloud of points
		pbar.set_postfix_str(f'dim={dim}, n={n}: generating points')
		pbar.refresh()
		points = np.random.random([n, dim])
		result.update({'points': points})

		# generate SimplexTree
		pbar.set_postfix_str(f'dim={dim}, n={n}: generating SimplexTree')
		pbar.refresh()
		stree = AlphaComplex(points).create_simplex_tree()
		result.update({'stree': stree})

		# searching depth poset
		pbar.set_postfix_str(f'dim={dim}, n={n}: finding DepthPoset')
		pbar.refresh()
		depth_poset = DepthPoset.from_simplex_tree(stree)
		result.update({'depth poset': depth_poset})

		# find poset scores on full poset
		result.update({('poset scores', 'full'): {}})
		for score in poset_scores_to_check:
			pbar.set_postfix_str(f'dim={dim}, n={n}: calculate {score.__name__} on full poset')
			pbar.refresh()
			score_value = score(depth_poset)
			result[('poset scores', 'full')].update({score.__name__: score_value})

		# find poset scores on subposets
		pass

		# find node scores on full poset
		pass

		# find node scores on subposets
		pass

		# save the result
		pass

		pbar.update()