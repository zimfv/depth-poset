import itertools
import numpy as np
import networkx as nx
from src.poset import Poset

from gudhi import SimplexTree


def reduct_column_bottom_to_top(delta, stop_condition=None):
	"""
	Returns the result of algorithm 1: Bottom to Top Column Reduction
	from the article The Poset of Cancellations in a Filtered Complex
	by Herbert Edelsbrunner, Michał Lipiński, Marian Mrozek, Manuel Soriano-Trigueros
	https://arxiv.org/abs/2311.14364

	Parameters:
	-----------
	delta : np.array shape (n, n) of ones and zeros
		Border matrix

	stop_condition: None or function
		Returns the intermediary delta for the condition, if it's not None

	Returns:
	--------
	alpha : list of tuples
		The birth-death index pairs, coresponding the shallow pairs

	b0 : list of tuples 
		The death-death index pairs, coresponding the relations 

	delta0 : np.array shape (n, n) of ones and zeros
		Modified border matrix, the intermediary delta for the condition, if it's not None
	"""
	delta0 = np.array(delta)
	b0 = []
	i = 0
	alpha = {}
	while (delta0 != 0).any():
		if stop_condition is not None:
			if stop_condition(alpha, b0, delta0):
				return alpha, b0, delta0
		# let delta0[s, t] be leftmost non-zero entry in last row, alpha_i = (s, t)
		ss, ts = np.where(delta0 != 0)
		s = int(np.max(ss))
		t = int(np.min(ts[ss == s]))
		alpha.update({i: (s, t)})
		# while there exists y > t such that delta0[s, y] = 1

		while (delta0[s, t+1:] != 0).any():
			# add column t to column y in delta0; append (t, y) to b0
			y = t + 1 + int(np.where(delta0[s, t+1:] != 0)[0][0])
			b0.append((t, y))
			if stop_condition is not None:
				if stop_condition(alpha, b0, delta0):
					return alpha, b0, delta0
			delta0[:, y] = (delta0[:, t] + delta0[:, y])%2

		if stop_condition is not None:
			if stop_condition(alpha, b0, delta0):
				return alpha, b0, delta0

		# delete rows s and t and columns s and t from delta0
		delta0[(s, t), :] = 0
		delta0[:, (s, t)] = 0

		i += 1

	if stop_condition is not None:
		return alpha, b0, delta0
	return alpha, b0


def reduct_row_left_to_right(delta, stop_condition=None):
	"""
	Returns the result of algorithm 2: Left to Right Row Reduction
	from the article The Poset of Cancellations in a Filtered Complex
	by Herbert Edelsbrunner, Michał Lipiński, Marian Mrozek, Manuel Soriano-Trigueros
	https://arxiv.org/abs/2311.14364

	Parameters:
	-----------
	delta : np.array shape (n, n) of ones and zeros
		Border matrix

	stop_condition: None or function
		Returns the intermediary delta for the condition, if it's not None

	Returns:
	--------
	omega : list of tuples
		The birth-death index pairs, coresponding the shallow pairs

	b1 : list of tuples 
		The death-death index pairs, coresponding the relations 

	delta1 : np.array shape (n, n) of ones and zeros
		Modified border matrix, the intermediary delta for the condition, if it's not None
	"""
	delta1 = np.array(delta)
	b1 = []
	j = 0
	omega = {}
	while (delta1 != 0).any():
		if stop_condition is not None:
			if stop_condition(omega, b1, delta1):
				return omega, b1, delta1
		# let delta1[s, t] lowest non-zero entry in first, omega_i = (s, t)
		ss, ts = np.where(delta1 != 0)
		t = int(np.min(ts))
		s = int(np.max(ss[ts == t]))
		omega.update({j: (s, t)})
		# while there exists x < s such that delta1[x, t] = 1
		while (delta1[:s, t] != 0).any():
			# add row s to row x in delta1; append (s, x) to b1
			x = int(np.where(delta1[:s, t] != 0)[0][0])
			b1.append((s, x))
			if stop_condition is not None:
				if stop_condition(omega, b1, delta1):
					return omega, b1, delta1
			delta1[x, :] = (delta1[s, :] + delta1[x, :])%2


		if stop_condition is not None:
			if stop_condition(omega, b1, delta1):
				return omega, b1, delta1
		
		# delete rows s and t and columns s and t from delta1
		delta1[(s, t), :] = 0
		delta1[:, (s, t)] = 0

		j += 1
	if stop_condition is not None:
		return omega, b1, delta1
	return omega, b1


def get_shallow_pairs_relations_from_reductions(alpha, b0, omega, b1):
	"""
	"""
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


def get_shallow_pairs_relations(delta):
	"""
	"""
	alpha, b0 = reduct_column_bottom_to_top(delta)
	omega, b1 = reduct_row_left_to_right(delta)

	return get_shallow_pairs_relations_from_reductions(alpha, b0, omega, b1)


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
		return f'ShallowPair([{self.birth_value:.4f}, {self.death_value:.4f}]{f", source={self.source}" if self.source is not None else ""}, dim={self.dim})'

	def __eq__(self, other):
		if not isinstance(other, ShallowPair):
			return NotImplemented
		if self.source is not None and other.source is not None:
			return self.source == other.source
		return (self.birth_index == other.birth_index) and (self.death_index == other.death_index)

	def __hash__(self):
		return hash((self.birth_index, self.death_index))

	def __str__(self):
		# 
		try:
			strs = []
			for simplex in self.source:
				try:
					if len(simplex) == 1:
						strs.append(str(simplex[0]))
					else:
						strs.append(str(simplex).replace(' ', ''))
				except TypeError:
						strs.append(str(simplex))
			return ','.join(strs)
		except Exception:
			return self.__repr__()



class DepthPoset(Poset):
	"""
	"""
	@classmethod
	def from_border_matrix(cls, border_matrix, dims, filter_values=None, sources=None):
		"""
		"""
		if filter_values is None:
			filter_values = np.arange(len(border_matrix))
		#if sources is None:
		#	sources = [None for i in range(len(border_matrix))]

		alpha, b0 = reduct_column_bottom_to_top(border_matrix)
		omega, b1 = reduct_row_left_to_right(border_matrix)
		bd, r = get_shallow_pairs_relations_from_reductions(alpha, b0, omega, b1)

		nodes = [ShallowPair(birth_index=bdi[0], death_index=bdi[1], birth_value=filter_values[bdi[0]], death_value=filter_values[bdi[1]], 
							 dim=dims[bdi[0]], source=None if sources is None else (sources[bdi[0]], sources[bdi[1]])) for bdi in bd]
		edges = [(nodes[e0], nodes[e1]) for e0, e1 in r]

		obj = cls(nodes=nodes, edges=edges)
		obj._b0_set = set((e0, e1) for e0, e1 in b0) # define the reduct_column_bottom_to_top pairs
		obj._b1_set = set((e0, e1) for e0, e1 in b1) # define the reduct_row_left_to_right pairs

		return obj

	@classmethod
	def from_simplex_tree(cls, stree: SimplexTree, remove_zero_persistant_pairs: bool=True):
		"""
		"""
		simplices, matrix = get_ordered_border_matrix_from_simplex_tree(stree)
		dims = [len(simplex) - 1 for simplex in simplices]
		filter_values = [value for key, value in stree.get_filtration()]

		if remove_zero_persistant_pairs:
			node_condition = lambda node: node.birth_value != node.death_value
		else:
			node_condition = lambda node: True

		return cls.from_border_matrix(matrix, dims, filter_values, sources=simplices).subposet(node_condition=node_condition)

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

	def get_column_bottom_to_top_reduction(self):
		"""
		Returns the subposet, which is the origin, given by algorithm 1 Column Bottom to Top Reduction
		"""
		try:
			condition = lambda edge: (edge[0].death_index, edge[1].death_index) in self._b0_set
			return self.subposet(edge_condition=condition)
		except AttributeError:
			raise AttributeError("The Dpeth poset should be defined from border matrix.")

	def get_row_left_to_right_reduction(self):
		"""
		Returns the subposet, which is the origin, given by algorithm 2 Row Left to Right Reduction
		"""
		try:
			condition = lambda edge: (edge[0].birth_index, edge[1].birth_index) in self._b1_set
			return self.subposet(edge_condition=condition)
		except AttributeError:
			raise AttributeError("The Dpeth poset should be defined from border matrix.")

	def get_labels(self):
		"""
		Returns the dict labaling the Nodes
		"""
		return {node: str(node) for node in self.nodes}
