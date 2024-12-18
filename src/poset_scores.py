import itertools
import networkx as nx
from src.poset import Poset


def number_of_nodes(poset: Poset) -> int:
    """
    Returns the number of nodes in the poset
    """
    return len(poset.nodes)


def number_of_minimal_nodes(poset: Poset) -> int:
    """
    Returns the number of minimal nodes.
    """
    return len([i for i in poset.iterate_minimal_nodes()])


def number_of_maximal_nodes(poset: Poset) -> int:
    """
    Returns the number of maximal nodes.
    """
    return len([i for i in poset.iterate_maximal_nodes()])


def number_of_edges_in_reduction(poset: Poset) -> int:
    """
    Returns the number of nodes in the poset transitive reduction
    """
    return len(poset.get_transitive_reduction().edges)


def number_of_edges_in_closure(poset: Poset) -> int:
    """
    Returns the number of nodes in the poset transitive closure
    """
    return len(poset.get_transitive_closure().edges)


def height(poset: Poset) -> int:
    """
    Returns the poset height - the length of the longest chain
    """
    g = poset.get_transitive_reduction()
    return nx.dag_longest_path_length(g)


def width(poset: Poset) -> int:
    """
    Returns the poset width - the length of the longest antichain (subset, s.t. all elements are pairwise incomparable)
    The algorithm is based on Dilworth's theorem and it's proof via Kőnig's theorem
    https://en.wikipedia.org/wiki/Dilworth%27s_theorem
    """
    g = poset.get_transitive_closure()

    b = nx.DiGraph()
    b.add_nodes_from([(0, node) for node in g.nodes] + [(1, node) for node in g.nodes])
    b.add_edges_from([((0, node0), (1, node1)) for node0, node1 in g.edges])

    matching = nx.bipartite.hopcroft_karp_matching(b, top_nodes=[(0, node) for node in g.nodes])
    matching = [(key[1], value[1]) for key, value in matching.items() if key[0] == 0]
    return len(g.nodes) - len(matching)

def avarage_maximal_chain(poset: Poset) -> float:
    """
    """
    s = 0
    n = 0
    for chain in poset.maximal_chains():
        s += len(chain)
        n += 1
    return s/n

def minimum_maximal_chain(poset: Poset) -> int:
    """
    """
    return min([len(chain) for chain in poset.maximal_chains()])
    