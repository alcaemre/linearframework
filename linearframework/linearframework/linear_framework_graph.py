"""
Emre Alca
title: linear_framework_graph.py
date: 2024-09-07 14:03:54

holds functions relevant to the creation and use of LinearFrameworkGraph objects
"""

import networkx as nx
import sympy as sp
import numpy as np

def _edge_to_sym_from_edges(edges):
    """takes a list of edges (tuples of 2 vertex id's, in the form ('v_1', 'v_2'))
    and returns a dictionary of the same edges pointing to new sympy symbols.
    These symbols are in the format l_i where i is the order of the initialization of the symbols.

    Args:
        edge_to_weight (dict[tuple[str]: float]): a dictionary of edges to weights

    Returns:
        dict[tuple[str]: sympy.core.symbol.Symbol]: dictionary of edges to symbols (representing weights)
    """
    if not isinstance(edges, list):
        raise NotImplementedError("edges must be a list of tuples of 2 vertex id's, in the form ('v_1', 'v_2')")
    edges = edges
    edge_to_sym = {}
    for i in range(len(edges)):
        edge_to_sym[edges[i]] = sp.symbols(f'l_{i + 1}')
    return edge_to_sym


def _nodes_from_edges(edges):
    """given a list of edges (2-tuples of nodes), 
    makes a list of the unique vertex labels in the set of edges

    Args:
        edges (list[tuple[Any]]): list of edges (2-tuples of nodes)

    Returns:
        list[Any]: list of unique vertex labels in edges
    """
    nodes = []
    for edge in edges:
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[1] not in nodes:
            nodes.append(edge[1])
    return nodes


def _find_terminal_nodes(edges, nodes):
    """given lists of vertices and nodes, finds the nodes with no outgoing edges

    Args:
        edges (list[tuple[Any]]): list of edges
        nodes (list[Any]): list of nodes

    Returns:
        list[Any]: list of terminal nodes
    """
    non_terminal_vertices = []
    for edge in edges:
        if edge[0] not in non_terminal_vertices:
            non_terminal_vertices.append(edge[0])

    terminal_vertices = []
    for node in nodes:
        if node not in non_terminal_vertices:
            terminal_vertices.append(node)

    return terminal_vertices


def _generate_sym_laplacian(edge_to_sym, nodes):
    """given an edge_to_sym dictionary, generates the appropriate Laplacian matrix

    Args:
        edge_to_sym (dict[tuple[Any]: sympy.core.symbol]): dictionary of edges to symbolic weights
        nodes (list[Any]): list of unique nodes in the edges in edge_to_weight

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: symbolic laplacian of the graph represented by edge_to_sym
    """
    sym_lap = []
    for i in range(len(nodes)):
        sym_lap.append([])
        for j in range(len(nodes)):
            sym_lap[i].append(0)
    
    for edge in list(edge_to_sym.keys()):
        i = nodes.index(edge[0])
        j = nodes.index(edge[1])

        sym_lap[i][j] = -edge_to_sym[edge]
        sym_lap[i][i] = sym_lap[i][i] + (- sym_lap[i][j])

    return sp.Matrix(sym_lap)


class LinearFrameworkGraph:
    """
    datatype for calculating symbolic expressions of linear framework results on directed, weighted graphs.
    Since we treat the edge labels as arbitrary

    attributes:
        self.nodes: list of nodes
        self.edges: list of edges
        self.terminal_nodes: list of terminal nodes
        self.edge_to_sym: dictionary from edges to symbolic edge weights
        self.sym_lap: symbolic laplacian generated from edges_to_symbolic_weights

    methods:
        self.

    """
    def __init__(self, edges):
        """initialized a LinearFrameworkGraph
        The input must be a list of tuples with 2 elements.
        Each element in these tuples represents a vertex in the graph
        and two appearing in a tuple as ('v_1', 'v_2') represents vertex 'v_1' having an edge to 'v_2'.

        Args:
            edges (lsit[tuple[Any]]): list of edges
        """
        if not isinstance(edges, list):
            raise NotImplementedError('edges must be a list of tuples with two elements')
        for edge in edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise NotImplementedError("edges must be 2-tuples of nodes in the form (v_1, v_2) for an edge from v_1 to v_2")

        self.edges=edges
        self.edge_to_sym = _edge_to_sym_from_edges(self.edges)
        self.nodes = _nodes_from_edges(self.edges)
        self.terminal_nodes = _find_terminal_nodes(self.edges, self.nodes)

        self.sym_lap = _generate_sym_laplacian(self.edge_to_sym, self.nodes)

        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_edges_from(edges)

    def generate_random_edge_to_weight(self, seed=None):
        """given a list (or other iterable) of edges in the form ('v_1', 'v_2'),
        makes a dictionary with the edges as keys pointing at random weights
        sampled from the range [10**(-3), 10**6]

        Args:
            seed (int, float): seed of random process

        Returns:
            dict[tuple[str]: float]: edges to randomly generated weights
        """
        np.random.seed(seed)
        edge_to_weight = {}
        for edge in self.edges:
            edge_to_weight[(str(edge[0]), str(edge[1]))] = 10 ** ((6 * np.random.rand()) - 3)
        return edge_to_weight