import networkx as nx


def get_unique_list(l) -> list:
    """
    Returns the list with unique elements
    """
    r = []
    for i in l:
        if not i in r:
            r.append(i)
    return r

def edges_contain_cycles(edges):
    """
    Returns True if DiGraph defined by edges contains a cycle
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)
    try:
        nx.find_cycle(g, orientation='original')
        return True
    except nx.NetworkXNoCycle:
        return False

def get_edges_transitive_reduction(edges):
    """
    Returns the transitive reduction for given relations
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)
    g = nx.transitive_reduction(g)
    return list(g.edges)
    
def get_edges_transitive_closure(edges):
    """
    Returns the transitive closure for given relations
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)
    g = nx.transitive_closure_dag(g)
    return list(g.edges)


class Poset:
    def add_nodes_from(self, nodes):
        """
        Add elements to poset from list

        Parameters:
        -----------
        nodes: list
            List of elements
        """
        self.nodes = get_unique_list(self.nodes + list(nodes))
    
    def add_edges_from(self, edges):
        """
        Add relations from list of relations

        Parameters:
        -----------
        edges: list of tuples
            List of node pairs in relation. The first node in pair is smaller
        """
        self.add_nodes_from([edge[0] for edge in edges] + [edge[1] for edge in edges])
        new_edges = get_unique_list(self.edges + edges)
        if edges_contain_cycles(new_edges):
            raise ValueError('There should not be a cycle.')
        self.edges = get_edges_transitive_reduction(new_edges)
        
    def __init__(self, nodes=[], edges=[]):
        """
        Inititalize the poset.

        Parameters:
        -----------
        nodes: list
            List of elements

        edges: list of tuples
            List of node pairs in relation. The first node in pair is smaller
        """
        self.nodes = []
        self.edges = []
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

    def get_transitive_reduction(self) -> nx.DiGraph:
        """
        Returns the directed graph, which is the transitive reduction of the poset
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(get_edges_transitive_reduction(self.edges))
        return g
        
    def get_transitive_closure(self) -> nx.DiGraph:
        """
        Returns the directed graph, which is the transitive closure of the poset
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(get_edges_transitive_closure(self.edges))
        return g

    def iterate_minimal_nodes(self):
        """
        Yields the nodes, s.t. there are no nodes less
        """
        for node in self.nodes:
            flag = True
            for edge in self.edges:
                if edge[1] == node:
                    flag = False
                    break
            if flag:
                yield node

    def iterate_maximal_nodes(self):
        """
        Yields the nodes, s.t. there are no nodes bigger
        """
        for node in self.nodes:
            flag = True
            for edge in self.edges:
                if edge[0] == node:
                    flag = False
                    break
            if flag:
                yield node

    def hasse_layout(self, gap=1) -> dict:
        """
        Returns the dict: keys are nodes and values are their positions.
        The positions corresponds a Hasse diagram:
        https://en.wikipedia.org/wiki/Hasse_diagram

        To think: How to decrease the number of intersections

        Parameters:
        -----------
        gap : float
            The gap size between connected components
        """
        if self.number_connected_components() == 1:
            graph = self.get_transitive_reduction()
            levels = {}
            for node in nx.topological_sort(graph):
                predecessors = list(graph.predecessors(node))
                levels[node] = max([levels[p] for p in predecessors], default=-1) + 1
            
            level_groups = {}
            for node, level in levels.items():
                level_groups.setdefault(level, []).append(node)

            pos = {}
            for level, nodes in level_groups.items():
                num_nodes = len(nodes)
                x_coords = range(-num_nodes // 2, num_nodes // 2 + 1)
                for x, node in zip(x_coords, nodes):
                    pos[node] = (x, level)
            return pos
        else:
            poses = []
            total_offset = 0
            for subgraph in self.connected_components():
                subgraph_pos = subgraph.hasse_layout()
                # Shift all x-coordinates of the subgraph by the total offset
                shifted_pos = {node: (x + total_offset, y) for node, (x, y) in subgraph_pos.items()}
                poses.append(shifted_pos)

                # Update total_offset to ensure separation between components
                width = max(x for x, y in subgraph_pos.values()) - min(x for x, y in subgraph_pos.values())
                total_offset += width + gap

            # Merge all positions into a single dictionary
            final_pos = {}
            for pos in poses:
                final_pos.update(pos)

            return final_pos

    def subposet(self, node_condition=None, edge_condition=None):
        """
        Returns the subset, satisfying the conditions:

        Parameters:
        -----------
        node_condition: function node from self.nodes -> bool
            The condition for nodes.

        edge_condition: function edge from self.edges -> bool
            The condition for edges.
        """

        if node_condition is None:
            node_condition = lambda node: True
        if edge_condition is None:
            edge_condition = lambda edge: True

        # Validate conditions
        if not callable(node_condition):
            raise ValueError("node_condition must be a callable function.")
        if not callable(edge_condition):
            raise ValueError("edge_condition must be a callable function.")

        # Filter nodes
        new_nodes = [node for node in self.nodes if node_condition(node)]

        # Filter edges
        new_edges = [edge for edge in self.edges 
                     if edge_condition(edge) and node_condition(edge[0]) and node_condition(edge[1])]

        return Poset(nodes=new_nodes, edges=new_edges)

    def comparable(self, a, b):
        """
        Returns True if elements a and b are comparable.
        Raises ValueError, if a or b not from nodes.
        """
        if (a not in self.nodes) or (b not in self.nodes):
            raise ValueError(f'Both elements {a} and {b} should be the nodes of the Poset')
        if a == b:
            return True
        g = self.get_transitive_closure()
        return ((a, b) in g.edges) or ((b, a) in g.edges)

    def higher(self, a, b):
        """
        Returns True if a > b
        Raises ValueError, if a or b not in nodes.
        """
        if (a not in self.nodes) or (b not in self.nodes):
            raise ValueError(f'Both elements {a} and {b} should be the nodes of the Poset')
        g = self.get_transitive_closure()
        return (b, a) in g.edges

    def lower(self, a, b):
        """
        Returns True if a < b
        Raises ValueError, if a or b not in nodes.
        """
        if (a not in self.nodes) or (b not in self.nodes):
            raise ValueError(f'Both elements {a} and {b} should be the nodes of the Poset')
        g = self.get_transitive_closure()
        return (a, b) in g.edges
        
    def number_connected_components(self):
        """
        Returns the number of connected components.
        """
        g = self.get_transitive_closure().to_undirected()
        return nx.number_connected_components(g)

    def connected_components(self):
        """
        Generate connected components as subgraphs.
        """
        g = self.get_transitive_closure().to_undirected()
        for nodeset in nx.connected_components(g):
            node_condition = lambda node: node in nodeset
            yield self.subposet(node_condition=node_condition)

    def maximal_chains(self):
        """
        Generate maximal chains as lists of nodes.
        """
        g = self.get_transitive_reduction()
        branching = nx.dag_to_branching(g)

        roots = [n for n in branching.nodes if branching.in_degree(n) == 0]
        leaves = [n for n in branching.nodes if branching.out_degree(n) == 0]

        for root in roots:
            for leaf in leaves:
                paths = nx.all_simple_paths(branching, source=root, target=leaf)
                for path in paths:
                    yield [branching.nodes[i]['source'] for i in path]