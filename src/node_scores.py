from src.poset import Poset
from src.poset_scores import height as poset_height
from src.poset_scores import width as poset_width

def ancestors_number(poset: Poset, node) -> int:
	"""
	Returns the number of nodes higher than given
	"""
	s = 0
	for other in poset.nodes:
		s += int(poset.higher(other, node))
	return s


def successors_number(poset: Poset, node) -> int:
	"""
	Returns the number of nodes higher than given
	"""
	s = 0
	for other in poset.nodes:
		s += int(poset.lower(other, node))
	return s


def incomparable_number(poset: Poset, node) -> int:
	"""
	Returns the number of incomparable elements for given node
	"""
	s = 0
	for other in poset.nodes:
		s += int(not poset.comparable(other, node))
	return s

def ancestors_height(poset: Poset, node) -> int:
	"""
	Returns the size of maximum chain of subposet of nodes higher or equal than given
	"""
	node_condition = lambda other: poset.higher(other, node)
	subposet = poset.subposet(node_condition=node_condition)
	return poset_height(subposet)

def successors_height(poset: Poset, node) -> int:
	"""
	Returns the size of maximum chain of subposet of nodes lower or equal than given
	"""
	node_condition = lambda other: poset.lower(other, node)
	subposet = poset.subposet(node_condition=node_condition)
	return poset_height(subposet)

def ancestors_width(poset: Poset, node) -> int:
	"""
	Returns the size of maximum chain of subposet of nodes higher or equal than given
	"""
	node_condition = lambda other: poset.higher(other, node)
	subposet = poset.subposet(node_condition=node_condition)
	return poset_width(subposet)

def successors_width(poset: Poset, node) -> int:
	"""
	Returns the size of maximum chain of subposet of nodes lower or equal than given
	"""
	node_condition = lambda other: poset.lower(other, node)
	subposet = poset.subposet(node_condition=node_condition)
	return poset_width(subposet)