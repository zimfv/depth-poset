from src.depth import DepthPoset
from src.utils import jacard_index


def poset_node_cell_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of nodes (elements) in the Depth Poset.
    Consider 2 birth-death pairs are similar if they corespond the similar cells.
    """
    set0 = set([node.source for node in dp0.nodes])
    set1 = set([node.source for node in dp1.nodes])
    return jacard_index(set0, set1)

def poset_node_index_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of nodes (elements) in the Depth Poset.
    Consider 2 birth-death pairs are similar if ther indices in the filtration pairs are similar.
    """
    set0 = set([(node.birth_index, node.death_index) for node in dp0.nodes])
    set1 = set([(node.birth_index, node.death_index) for node in dp1.nodes])
    return jacard_index(set0, set1)

def poset_arc_cell_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the Depth Poset.
    Consider 2 birth-death pairs are similar if they corespond the similar cells.
    """
    set0 = set([(node0.source, node1.source) for node0, node1 in dp0.get_transitive_closure().edges])
    set1 = set([(node0.source, node1.source) for node0, node1 in dp1.get_transitive_closure().edges])
    return jacard_index(set0, set1)

def poset_arc_index_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the Depth Poset.
    Consider 2 birth-death pairs are similar if ther indices in the filtration pairs are similar.
    """
    set0 = set([(node0.birth_index, node0.death_index, node1.birth_index, node1.death_index) for node0, node1 in dp0.get_transitive_closure().edges])
    set1 = set([(node0.birth_index, node0.death_index, node1.birth_index, node1.death_index) for node0, node1 in dp1.get_transitive_closure().edges])
    return jacard_index(set0, set1)

def birth_relation_cell_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the birth relation (given by row left to right reduction algorithm).
    Consider 2 birth-death pairs are similar if they corespond the similar cells.
    """
    set0 = set([(n0.source, n1.source) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp0._b1_set)])
    set1 = set([(n0.source, n1.source) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp1._b1_set)])
    return jacard_index(set0, set1)

def birth_relation_index_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the birth relation (given by row left to right reduction algorithm).
    Consider 2 birth-death pairs are similar if ther indices in the filtration pairs are similar.
    """
    set0 = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp0._b1_set)])
    set1 = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp1._b1_set)])
    return jacard_index(set0, set1)

def death_relation_cell_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the death relation (given by column bottom to top reduction algorithm).
    Consider 2 birth-death pairs are similar if they corespond the similar cells.
    """
    set0 = set([(n0.source, n1.source) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp0._b0_set)])
    set1 = set([(n0.source, n1.source) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp1._b0_set)])
    return jacard_index(set0, set1)

def death_relation_index_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the death relation (given by column bottom to top reduction algorithm).
    Consider 2 birth-death pairs are similar if ther indices in the filtration pairs are similar.
    """
    set0 = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp0._b0_set)])
    set1 = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp1._b0_set)])
    return jacard_index(set0, set1)

def relation_cell_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the relation (given by reduction algorithms).
    Consider 2 birth-death pairs are similar if they corespond the similar cells.
    """
    set0b = set([(n0.source, n1.source) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp0._b1_set)])
    set1b = set([(n0.source, n1.source) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp1._b1_set)])
    set0d = set([(n0.source, n1.source) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp0._b0_set)])
    set1d = set([(n0.source, n1.source) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp1._b0_set)])
    set0 = set0b | set0d
    set1 = set1b | set1d    
    return jacard_index(set0, set1)

def relation_index_similarity(dp0: DepthPoset, dp1: DepthPoset) -> float:
    """
    The Jacard index of arcs (edges) in the relation (given by reduction algorithms).
    Consider 2 birth-death pairs are similar if ther indices in the filtration pairs are similar.
    """
    set0b = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp0._b1_set)])
    set1b = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.birth_index, n1.birth_index) in dp1._b1_set)])
    set0d = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp0.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp0._b0_set)])
    set1d = set([(n0.birth_index, n0.death_index, n1.birth_index, n1.death_index) for n0, n1 in dp1.get_transitive_closure().edges if ((n0.death_index, n1.death_index) in dp1._b0_set)])
    set0 = set0b | set0d
    set1 = set1b | set1d    
    return jacard_index(set0, set1)
