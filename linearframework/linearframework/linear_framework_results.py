"""
Emre Alca
title: linear_framework_results.py
date: 2024-08-16 14:59:41

holds the functionality for calculating the kth moment of the first passage time and the steady state probability of a given graph.
This is done by separately calculating the numerator and denominator of each of these quantities.
"""

import sympy as sp
import numpy as np
from math import factorial
from operator import mul
from functools import reduce

from linearframework.linear_framework_graph import LinearFrameworkGraph, hill_augmented_graph, terminalize
import linearframework.ca_recurrence as ca


def steady_states_from_sym_lap(graph):
    """calculates the symbolic formula for the steady states of a graph from its symbolic laplacian using the first-minors MTT.
    This method is much faster than using Q_k matrices, since it circumvents the need for the calculation of Q_k matrices.
    Note that graph can have no more than one terminal vertex for this quantity to be well defined.

    Args:
        graph(LinearFrameworkGraph) a LinearFrameworkGraph with no more than one terminal vertex

    Returns:
        list[sympy.core.mul.Mul]: list of the steady states of each vertex in the graph in canonical order
    """
    if not isinstance(graph, LinearFrameworkGraph):
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no more than one terminal vertex")
    if len(graph.terminal_nodes) > 1:
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no more than one terminal vertex")

    sym_lap = graph.sym_lap

    n = sym_lap.rows
    rhos = []
    for i in range(n):
        rho_i = sym_lap.minor(i,i)
        rhos.append(rho_i)
    
    denominator = sum(rhos)

    nodes = list(graph.nodes)
    steady_states = {}
    for i in range(len(rhos)):
        steady_states[nodes[i]] = rhos[i] / denominator

    return steady_states


def steady_states_from_Q_n_minus_1(graph):
    """calculates the symbolic formula for the steady states of a graph from the diagonal elements of Q_(n-1).
    This method is much slower than the alternative that uses the first-minors MTT.
    It is mostly used for testing to ensure the two outputs are equal.
    Note that graph can have no more than one terminal vertex for this quantity to be well defined.

    Args:
        graph(LinearFrameworkGraph) a LinearFrameworkGraph with no more than one terminal vertex

    Returns:
        list[sympy.core.mul.Mul]: list of the steady states of each vertex in the graph in canonical order
    """
    if not isinstance(graph, LinearFrameworkGraph):
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no more than one terminal vertex")
    if len(graph.terminal_nodes) > 1:
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no more than one terminal vertex")
    
    sym_lap = graph.sym_lap

    n = sym_lap.rows
    Q_n1 = ca.get_sigma_Q_k(sym_lap, n-1)[1]

    rhos = []
    for i in range(n):
        rhos.append(Q_n1.row(i)[i])
    
    denominator = sum(rhos)

    nodes = list(graph.nodes)
    steady_states = {}
    for i in range(len(rhos)):
        steady_states[nodes[i]] = rhos[i] / denominator
    
    return steady_states


def get_vec_indices(ranges):
    """does the combinatorics for ranges.
    """
    operations=reduce(mul,(p[1]-p[0] for p in ranges))-1
    result=[i[0] for i in ranges]
    pos=len(ranges)-1
    increments=0
    indices = []
    indices.append(result.copy())
    while increments < operations:
        if result[pos]==ranges[pos][1]-1:
            result[pos]=ranges[pos][0]
            pos-=1
        else:
            result[pos]+=1
            increments+=1
            pos=len(ranges)-1 #increment the innermost loop
            indices.append(result.copy())
    return indices


def get_j_vecs_from_indices(vec_j, k):
    """does the combinatorics on all possible j-vecs 
    for the k-th moment of the FPT
    """
    ranges = []
    for i in range(k):
        ranges.append((0, len(vec_j)))
    index_vecs = get_vec_indices(ranges)
    j_vecs = []
    for index_vec in index_vecs:
        j_vec = []
        for index in index_vec:
             j_vec.append(vec_j[index])
        j_vecs.append(j_vec)
    return j_vecs


def forest_weight_string_list_to_forest_weight_sym(forest_weight_string_list):
    """transforms a string representing the symbols in a single spanning forest into a sympy symbol

    Args:
        forest_weight_string_list (list[str]): list of strings representing the weights of spanning forests

    Returns:
        list[sympy.matrices.immutable.ImmutableDenseMatrix]: list of symbol products representing the weights of spanning forests
    """
    forest_weight_sym_list = []
    for forest_weight_string in forest_weight_string_list:
        edge_weights = forest_weight_string.split('*')
        edge_symbols = sp.symbols(edge_weights)
        forest_weight_sym_list.append(sp.prod(edge_symbols))
    return forest_weight_sym_list


def _ca_kth_moment_numerator(graph, Q_n_minus_2, source, target, moment):
    """ calculates the numerator of the k-th moment of a graph using the Q_(n-2) matrix given by the CA recurrence.

    Args:
        graph (networkx.classes.digraph.DiGraph): networkx graph of interent
        Q_n_minus_2 (sympy.matrices.dense.MutableDenseMatrix): Q_(n-2) matrix found using the CA recurrence
        source (str): id of source vertex
        target (str): id of target vertex
        moment (int): moment of interest

    Returns:
        sympy.core.add.Add: symbolic expression for the the numerator of the moment of interest.
    """
    if not isinstance(graph, LinearFrameworkGraph):
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no more than one terminal vertex")
    if len(graph.terminal_nodes) > 1:
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no more than one terminal vertex")
    if not isinstance(Q_n_minus_2, sp.matrices.dense.MutableDenseMatrix):
        raise NotImplementedError("Q_n_minus_2 must be a sympy matrix, and a Q_(n-2) matrix")

    if source not in list(graph.nodes):
        raise NotImplementedError("source must be an id of a vertex in graph")
    if target not in list(graph.nodes):
        raise NotImplementedError("target must be an id of a vertex in graph")
    if not isinstance(moment, int) or moment <= 0:
        raise NotImplementedError("moment must be a natural number")

    I = list(graph.nodes)
    I.remove(target)

    j_vecs = get_j_vecs_from_indices(I, moment)

    sum_prods = 0
    for j_vec in j_vecs:
        prod_inner_sums = 1

        for u in range(moment):
            if u == 0:
                j_n_1 = source
            else:
                j_n_1 = j_vec[u-1]
            j_n = j_vec[u]

            jn1_index = graph.nodes.index(j_n_1)
            jn_index = graph.nodes.index(j_n)

            # sum_weights = sum_sym_weights_jq_roots_ij_path(graph, Q_n_minus_2, [j_n, target], j_n_1, j_n)
            # sum_weights = sum_sym_weights_roots_ij_path_from_intersection(graph, Q_n_minus_2, [j_n, target], j_n_1, j_n)
            sum_weights = Q_n_minus_2.row(jn1_index)[jn_index]

            prod_inner_sums *= sum_weights

        sum_prods += prod_inner_sums
    
    return factorial(moment) * sum_prods


def k_moment_fpt_expression(graph, source, target, moment):
    """calculates the symbolic expression of the moment-th moment of the first passage time from source to target
     of the graph represented by edge_to_weight with the symbols of the edge weights being the values in edge_to_sym

    Args:
        graph (LinearFrameworkGraph): LinearFrameworkGraph with no more than one terminal vertex
        source (str): id of source vertex
        target (str): id of target vertex
        moment (int): moment of interest

    Returns:
        sympy.core.add.Add: symbolic expression of the moment-th moment of the graph over the mean of the graph to the power of moment
    """
    if not isinstance(graph, LinearFrameworkGraph):
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no more than one terminal vertex")
    if not len(graph.terminal_nodes) == 0:
        raise NotImplementedError("graph must be a LinearFrameworkGraph with no terminal vertices")
    if source not in list(graph.nodes):
        raise NotImplementedError("source must be an id of a vertex in graph")
    if target not in list(graph.nodes):
        raise NotImplementedError("target must be an id of a vertex in graph")
    if not isinstance(moment, int) or moment <= 0:
        raise NotImplementedError("moment must be a natural number")

    graph = terminalize(graph, target)

    sym_lap = graph.sym_lap
    n = sym_lap.rows
    Q_n_minus_2 = ca.get_sigma_Q_k(sym_lap, n-2)[1]

    numerator = _ca_kth_moment_numerator(graph, Q_n_minus_2, source, target, moment)
    q = list(graph.nodes).index(target)

    sym_lap = graph.sym_lap
    denominator = sym_lap.minor(q, q)
    denominator = denominator ** moment

    return numerator / denominator


def splitting_probability_ca(graph, source, target):
    """Calculates the splitting probability of the graph represented by edge_to_sym from source to target.

    Args:
        graph (LinearFrameworkGraph): LinearFrameworkGraph with no more than one terminal vertex
        source (str): vertex id of source of splitting probability
        target (str): vertex id of target of splitting probability

    Returns:
        sympy.core.mul.Mul: sympy expression of splitting probability
    """
    if not isinstance(graph, LinearFrameworkGraph):
        raise NotImplementedError("graph must be a LinearFrameworkGraph with at least one terminal vertex")
    if len(graph.terminal_nodes) < 1:
        raise NotImplementedError("graph must be a LinearFrameworkGraph with two or more terminal vertices")
    if not isinstance(source, str) or source not in list(graph.nodes):
        raise NotImplementedError("source must be a string and must be the id of a vertex in graph")
    if not isinstance(target, str) or target not in list(graph.nodes):
        raise NotImplementedError("target must be a string and must be the id of a vertex in graph")
    
    sym_lap = graph.sym_lap
    terminal_vertices = graph.terminal_nodes
    source_index = graph.nodes.index(source)
    target_index = graph.nodes.index(target)

    n = sym_lap.rows
    m = len(terminal_vertices)
    Q_n_minus_m = ca.get_sigma_Q_k(sym_lap, n-m)[1]
    denominator = Q_n_minus_m.row(target_index)[target_index]
    numerator = Q_n_minus_m.row(source_index)[target_index]

    return numerator / denominator

    
# METHODS USING HILL-AUGMENTED GRAPHS

def sum_terminal_edge_rho_augmented_graph(graph, target_node, augmentation_node):
    """Iterates over all terminal edges in graph which lead to target node.
    For each terminal edge leading to the target nodes, takes the product of the the weight of the terminal edge
    and the weight of the sum of the spanning forests of the Hill-augmentation of graph to augmentation node 
    rooted at the vertex leading to the target node in the terminal edge.
    Then returns the sum over this product for all terminal edges

    Args:
        graph (LinearFrameworkGraph): graph of interest
        target_node (Any): target node id, must be a terminal node of graph
        augmentation_node (Any): node with which we are Hill-augmenting to

    Returns:
        sympy.core.add.Add: expression calculated
    """
    terminal_edges_to_target_node = []
    for terminal_edge in graph.terminal_edges:
        if terminal_edge[1] == target_node:
            terminal_edges_to_target_node.append(terminal_edge)
        
    augmented_graph = hill_augmented_graph(graph, augmentation_node)

    sum_weights = 0
    for terminal_edge in terminal_edges_to_target_node:
        j = augmented_graph.nodes.index(terminal_edge[0])
        augmented_graph_rho_j = augmented_graph.sym_lap.minor(j, j)
        sum_weights += graph.edge_to_sym[terminal_edge] * augmented_graph_rho_j
    
    return sum_weights


def hill_splitting_probability(graph, source, target):
    """Uses Hill-augmentation to calculate the splitting probability of reaching target from source on graph

    Args:
        graph (LinearFrameworkGraph): graph of interest, must have at least one terminal vertex
        source (Any): source vertex of graph
        target (Any): target vertex of graph

    Returns:
        sympy.core.mul.Mul: symbolic expression of splitting probability
    """
    if not isinstance(graph, LinearFrameworkGraph):
        raise NotImplementedError("graph must be a LinearFrameworkGraph with at least one terminal vertex")
    if len(graph.terminal_nodes) < 1:
        raise NotImplementedError("graph must be a LinearFrameworkGraph with at least one terminal vertex")
    if not isinstance(source, str) or source not in list(graph.nodes):
        raise NotImplementedError("source must be a string and must be the id of a vertex in graph")
    if not isinstance(target, str) or target not in list(graph.nodes):
        raise NotImplementedError("target must be a string and must be the id of a vertex in graph")

    numerator = sum_terminal_edge_rho_augmented_graph(graph, target, source)

    denominator = 0
    for terminal_node in graph.terminal_nodes:
        denominator += sum_terminal_edge_rho_augmented_graph(graph, terminal_node, source)
    
    return numerator / denominator