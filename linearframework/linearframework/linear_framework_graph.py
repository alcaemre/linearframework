"""
Emre Alca
title: linear_framework_graph.py
date: 2024-09-07 14:03:54

holds functions relevant to the creation and use of LinearFrameworkGraph objects
"""

import numpy as np


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


def _find_terminal_edges(edges, terminal_nodes):
    terminal_edges = []
    for edge in edges:
        if edge[1] in terminal_nodes:
            terminal_edges.append(edge)

    return terminal_edges


def _generate_laplacian(edge_to_weight, nodes):
    """given an edge_to_sym dictionary, generates the appropriate Laplacian matrix

    Args:
        edge_to_sym (dict[tuple[Any]: sympy.core.symbol]): dictionary of edges to symbolic weights
        nodes (list[Any]): list of unique nodes in the edges in edge_to_weight

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: symbolic laplacian of the graph represented by edge_to_sym
    """
    Lap = np.zeros((len(nodes), len(nodes)))
    
    for edge in list(edge_to_weight.keys()):
        i = nodes.index(edge[0])
        j = nodes.index(edge[1])

        Lap[j][i] = edge_to_weight[edge]
        Lap[i][i] = Lap[i][i] - (Lap[j][i])

<<<<<<< HEAD
    return Lap
=======
    return - sp.Matrix(sym_lap).T
>>>>>>> bd1c3a05bd6580dbaf6aebeed8b29a09753e76f1


def _hill_augmented_edge_to_weight(graph, augmentation_vertex):
    """makes the edge_to_sym dictionary the Hill augmentation to vertex i of graph self.

    Args:
        augmentation_vertex (str): vertex to which we are performing a Hill-augmentation to.

    Returns:
        dict: edge_to_sym dictionary of self Hill-augmented to vertex i
    """
    hill_edge_to_weight = {}
    for edge in graph.edges:
        if edge not in graph.terminal_edges:
            hill_edge_to_weight[edge] = graph.edge_to_weight[edge]

    for terminal_edge in graph.terminal_edges:
        new_hill_edge = (terminal_edge[0], augmentation_vertex)
        if new_hill_edge not in hill_edge_to_weight.keys() and new_hill_edge[0] != new_hill_edge[1]:
            hill_edge_to_weight[new_hill_edge] = graph.edge_to_weight[terminal_edge]
        elif new_hill_edge[0] != new_hill_edge[1]:
            hill_edge_to_weight[new_hill_edge] += graph.edge_to_weight[terminal_edge]
    
    return hill_edge_to_weight
    

def hill_augmented_graph(graph, augmentation_vertex):
        """makes a LinearFrameworkGraph object representing a Hill augmented graph of self with superscript i.
        That is, any terminal edges are redirected into vertex i

        Args:
            graph()
            i (Any): vertex id of desired vertex of augmentation

        Returns:
            LinearFrameworkGraph: hill augmented graph of self with superscript i
        """
        if not isinstance(graph, LinearFrameworkGraph):
            raise NotImplementedError("graph must be a LinearFrameworkGraph")
        if not augmentation_vertex in graph.nodes:
            raise(NotImplementedError("augmentation must be to an existing node"))

        augmented_edge_to_weight = _hill_augmented_edge_to_weight(graph, augmentation_vertex)
        augmented_graph = LinearFrameworkGraph(augmented_edge_to_weight)
        return augmented_graph


def terminalize(graph, node):
    """makes node in graph a terminal node

    Args:
        graph (LinearFrameworkGraph): graph of interest
        node (Any): node to make terminal

    Returns:
        LinearFrameworkGraph: a graph with node made terminal
    """
    if not isinstance(graph, LinearFrameworkGraph):
        raise NotImplementedError("graph must be a LinearFrameworkGraph")
    if not node in graph.nodes:
        raise NotImplementedError("terminal_node must be a node of graph")

    new_edge_to_weight = {}

    for key in graph.edge_to_weight.keys():
        if not key[0] == node:
            new_edge_to_weight[key] = graph.edge_to_weight[key]
    
    terminal_graph = LinearFrameworkGraph(new_edge_to_weight)

    return terminal_graph

class LinearFrameworkGraph:
    """
    datatype for calculating linear framework results on directed, weighted graphs.

    attributes:
        self.nodes: list of nodes
        self.edges: list of edges
        self.terminal_nodes: list of terminal nodes
        self.edge_to_weight: dictionary from edges to edge weights
        self.Lap: symbolic laplacian generated from edge_to_weight

    """
    def __init__(self, edge_to_weight):
        """initializes a LinearFrameworkGraph
        The input can be a list of tuples with 2 elements (edges).
        Each element in these tuples represents a vertex in the graph
        and two appearing in a tuple as ('v_1', 'v_2') represents vertex 'v_1' having an edge to 'v_2'.
        There is also an option to explicitly provide the edge to sym dictionary, 
        but this is mostly for the creation of Hill-augmented graphs rather than explicitly making graphs.

        Args:
            edges (list[tuple[Any]]): list of edges
        """
        if not isinstance(edge_to_weight, dict):
            raise NotImplementedError("edge_to_weight must be a dictionary edges as keys--that is tuples of two objects ('v_1', 'v_2') for an edge from 'v_1' to 'v_2' pointing to float weights")
        edges = list(edge_to_weight.keys())
        if not isinstance(edges[0], tuple) or len(edges[0]) != 2:
            raise NotImplementedError("edges must be 2-tuples of nodes in the form (v_1, v_2) for an edge from v_1 to v_2")

        self.edges = edges

        self.edge_to_weight = edge_to_weight

        self.nodes = _nodes_from_edges(self.edges)
        self.terminal_nodes = _find_terminal_nodes(self.edges, self.nodes)
        self.terminal_edges = _find_terminal_edges(self.edges, self.terminal_nodes)

        self.Lap = _generate_laplacian(self.edge_to_weight, self.nodes)


