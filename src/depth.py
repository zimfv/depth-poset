import itertools
import numpy as np
import networkx as nx
from src.poset import Poset

from gudhi import SimplexTree


def reduct_column_bottom_to_top(delta):
	"""
	"""
	delta0 = np.array(delta)
	b0 = []
	i = 0
	alpha = {}
	while (delta0 != 0).any():
		# let delta0[s, t] be leftmost non-zero entry in last row, alpha_i = (s, t)
		ss, ts = np.where(delta0 != 0)
		s = int(np.max(ss))
		t = int(np.min(ts[ss == s]))
		alpha.update({i: (s, t)})
		# while there exists y > t such that delta0[s, y] = 1
		while (delta0[s, t+1:] != 0).any():
			# add column t to column y in delta0; append (t, y) to b0
			y = t + 1 + int(np.where(delta0[s, t+1:] != 0)[0][0])
			delta0[:, y] = (delta0[:, t] + delta0[:, y])%2
			b0.append((t, y))
		# delete rows s and t and columns s and t from delta0
		delta0[(s, t), :] = 0
		delta0[:, (s, t)] = 0
        
		i += 1
	return alpha, b0


def reduct_row_left_to_right(delta):
	"""
	"""
	delta1 = np.array(delta)
	b1 = []
	j = 0
	omega = {}
	while (delta1 != 0).any():
		# let delta1[s, t] lowest non-zero entry in first, omega_i = (s, t)
		ss, ts = np.where(delta1 != 0)
		t = int(np.min(ts))
		s = int(np.max(ss[ts == t]))
		omega.update({j: (s, t)})
		# while there exists x < s such that delta1[x, t] = 1
		while (delta1[:s, t] != 0).any():
			# add row s to row x in delta1; append (s, x) to b1
			x = int(np.where(delta1[:s, t] != 0)[0][0])
			delta1[x, :] = (delta1[s, :] + delta1[x, :])%2
			b1.append((s, x))
		# delete rows s and t and columns s and t from delta1
		delta1[(s, t), :] = 0
		delta1[:, (s, t)] = 0
        
		j += 1
	return omega, b1


def get_shallow_pairs_relations(delta):
	"""
	"""
	alpha, b0 = reduct_column_bottom_to_top(delta)
	omega, b1 = reduct_row_left_to_right(delta)
    
	bd = list(alpha.values())

	r = []
	for iphi, phi in enumerate(bd):
		for ipsi, psi in enumerate(bd):
			if (phi[1], psi[1]) in b0 or (phi[0], psi[0]) in b1:
				r.append((iphi, ipsi))

	# transitive closure
	g = nx.DiGraph()
	g.add_edges_from(r)
	r = list(nx.transitive_closure(g).edges())
    
	return bd, r


def find(l, v):
	"""
	Returns the index of the first inclusion of element v in list l
	"""
	for i, e in enumerate(l):
		if e == v:
			return i
	return -1


def get_ordered_border_matrix_from_simplex_tree(stree: SimplexTree):
	"""
	Returns the ordered border matrix from SimplexTree with given filtration
	"""
	simplices = [tuple(simplex) for simplex, filtration_value in stree.get_filtration()]
    
	col_indices = []
	row_indices = []
	for i, simplex in enumerate(simplices):
		for subsimplex in itertools.permutations(simplex, len(simplex) - 1):
			j = find(simplices, subsimplex)
			if j != -1:
				col_indices.append(j)
				row_indices.append(i)
	data = np.ones(len(row_indices))
	shape=[stree.num_simplices(), stree.num_simplices()]
	matrix = np.zeros(shape, dtype=int)
	matrix[col_indices, row_indices] = 1
	return simplices, matrix


class ShallowPair:
	"""
	"""
	def __init__(self, birth_index, death_index, birth_value, death_value, dim, source=None):
		self.birth_index = birth_index
		self.death_index = death_index
		self.birth_value = birth_value
		self.death_value = death_value
		self.dim = dim
		self.source = source

	def __repr__(self):
		return f'ShallowPair([{self.birth_value:.4f}, {self.death_value:.4f}], dim={self.dim})' # should there be simplier and shorter representation

	def __eq__(self, other):
		if not isinstance(other, ShallowPair):
			return NotImplemented
		return (self.birth_index == other.birth_index) and (self.death_index == other.death_index)

	def __hash__(self):
		return hash((self.birth_index, self.death_index))


class DepthPoset(Poset):
	"""
	"""
	@classmethod
	def from_border_matrix(cls, border_matrix, dims, filter_values=None, sources=None):
		"""
		"""
		if filter_values is None:
			filter_values = np.arange(len(border_matrix))
		if sources is None:
			sources = [None for i in range(len(border_matrix))]

		bd, r = get_shallow_pairs_relations(border_matrix)

		nodes = [ShallowPair(birth_index=bdi[0], death_index=bdi[1], birth_value=filter_values[bdi[0]], death_value=filter_values[bdi[1]], 
							 dim=dims[bdi[0]], source=sources[bdi[0]]) for bdi in bd]
		edges = [(nodes[e0], nodes[e1]) for e0, e1 in r]
		return DepthPoset(nodes=nodes, edges=edges)

	@classmethod
	def from_simplex_tree(cls, stree: SimplexTree):
		"""
		"""
		simplices, matrix = get_ordered_border_matrix_from_simplex_tree(stree)
		dims = [len(simplex) - 1 for simplex in simplices]
		filter_values = [value for key, value in stree.get_filtration()]
		return DepthPoset.from_border_matrix(matrix, dims, filter_values, sources=simplices)

	def get_dim(self):
		"""
		Returns the dimension of the poset
		"""
		return max([node.dim for node in self.nodes])

	def get_filtration_values(self):
		"""
		Returns the array of filtration values
		"""
		return np.unique([[node.birth_value, node.death_value] for node in self.nodes])


	def persistant_layout(self, by_index=False):
		"""
		Returns the dict: keys are nodes and values are their positions.
		The x-position is the birth, and the y-position is the death.

		Parameters:
		-----------
		by_index: bool
			Returns the index of the birth/death diltration values, except the real values
		"""
		if by_index:
			return {node: (node.birth_index, node.death_index) for node in self.nodes}
		else:
			return {node: (node.birth_value, node.death_value) for node in self.nodes}

	def subposet_dim(self, dim: int):
		"""
		Returns the subset, of given dimension

		Parameters:
		-----------
		dim: int
			The dimension of nodes
		"""
		node_condition = lambda node: node.dim == dim
		return self.subposet(node_condition=node_condition)
