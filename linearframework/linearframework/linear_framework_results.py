"""
Emre Alca
title: linear_framework_results.py
date: 2024-08-16 14:59:41

holds the functionality for calculating the kth moment of the first passage time and the steady state probability of a given graph.
This is done by separately calculating the numerator and denominator of each of these quantities.
"""

import sympy as sp
import networkx as nx
import numpy as np
from math import factorial
from operator import mul
from functools import reduce

import linearframework.graph_operations as g_ops

from linearframework.linear_framework_graph import LinearFrameworkGraph
import linearframework.ca_recurrence as ca

# def steady_states_from_sym_lap(edge_to_weight, edge_to_sym):
#     """calculates the symbolic formula for the steady states of a graph from its symbolic laplacian using the first-minors MTT.
#     This method is much faster than using Q_k matrices, since it circumvents the need for the calculation of Q_k matrices.

#     Args:
#         edge_to_weight (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): w} where w is some positive number
#         edge_to_sym (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): l} where l is some sympy symbol

#     Returns:
#         list[sympy.core.mul.Mul]: list of the steady states of each vertex in the graph in canonical order
#     """
#     if not isinstance(edge_to_weight, dict):
#         raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
#     for key in edge_to_weight.keys():
#         if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
#             raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
#         if not isinstance(edge_to_weight[key], (float, int)):
#             raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    
#     if not isinstance(edge_to_sym, dict):
#         raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
#     for key in edge_to_sym.keys():
#         if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
#             raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
#         if not isinstance(edge_to_sym[key], sp.core.symbol.Symbol):
#             raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")

#     graph = g_ops.dict_to_graph(edge_to_weight)
#     sym_lap = ca.generate_sym_laplacian(graph, edge_to_sym)

#     n = sym_lap.rows
#     rhos = []
#     for i in range(n):
#         rho_i = sym_lap.minor(i,i)
#         rhos.append(rho_i)
    
#     denominator = sum(rhos)

#     nodes = list(graph.nodes)
#     steady_states = {}
#     for i in range(len(rhos)):
#         steady_states[nodes[i]] = rhos[i] / denominator

#     return steady_states

# def steady_states_from_Q_n_minus_1(edge_to_weight, edge_to_sym):
#     """calculates the symbolic formula for the steady states of a graph from the diagonal elements of Q_(n-1).
#     This method is much slower than the alternative that uses the first-minors MTT.
#     It is mostly used for testing to ensure the two outputs are equal.

#     Args:
#         edge_to_weight (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): w} where w is some positive number
#         edge_to_sym (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): l} where l is some sympy symbol

#     Returns:
#         list[sympy.core.mul.Mul]: list of the steady states of each vertex in the graph in canonical order
#     """
#     if not isinstance(edge_to_weight, dict):
#         raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
#     for key in edge_to_weight.keys():
#         if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
#             raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
#         if not isinstance(edge_to_weight[key], (float, int)):
#             raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    
#     if not isinstance(edge_to_sym, dict):
#         raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
#     for key in edge_to_sym.keys():
#         if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
#             raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
#         if not isinstance(edge_to_sym[key], sp.core.symbol.Symbol):
#             raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")

#     graph = g_ops.dict_to_graph(edge_to_weight)
#     sym_lap = ca.generate_sym_laplacian(graph, edge_to_sym)

#     n = sym_lap.rows
#     Q_n1 = ca.get_sigma_Q_k(sym_lap, n-1)[1]

#     rhos = []
#     for i in range(n):
#         rhos.append(Q_n1.row(i)[i])
    
#     denominator = sum(rhos)

#     nodes = list(graph.nodes)
#     steady_states = {}
#     for i in range(len(rhos)):
#         steady_states[nodes[i]] = rhos[i] / denominator
    
#     return steady_states


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


def filter_by_forbidden_factors(forest_weight_str_list, forbidden_factor_strs):
    """we want the sum of the weights of the spanning forests with a certain set of roots and a certain path. 
    The elements of the Q_k matrix are sums of weights of spanning forests with a certain path, but are
    ambivalent to roots. So we need to filter to only include the sums of the weights of the spanning forests
    that do not contain the outgoing edges of our desired roots.

    Args:
        forest_weight_str_list (list[str]): list of strings of symbolic forest weights
        forbidden_factor_strs (list[str]): list of the symbolic weights of the outgoing edges of roots

    Returns:
        list[str]: list of permitted spanning forest weights
    """
    filtered_forest_weights = []
    for forest_weight_str in forest_weight_str_list:
        edge_weights = forest_weight_str.split('*')
        permitted = True
        for edge_weight in edge_weights:
            if edge_weight in forbidden_factor_strs:
                permitted = False
        if permitted:
            filtered_forest_weights.append(forest_weight_str)
    return filtered_forest_weights


def sum_sym_weights_jq_roots_ij_path(graph, Lap, Q_n_minus_2, roots, i, j):
    """calculates the sum of the weights of the spanning forests rooted at roots, with a path from i to j

    Args:
        graph (nx.DiGraph): the graph of interest
        Lap (sympy.matrices.dense.MutableDenseMatrix): symbolic laplacian of graph
        Q_n_minus_2 (sympy.matrices.dense.MutableDenseMatrix): Q_(n-2) matrix of graph
        roots (list[str]): list of desired roots
        i (str): vertex with a required path from
        j (str): vertex with a required path to

    Returns:
        sympy.core.add.Add: sum of the weights of the spanning forests rooted at roots, with a path from i to j
    """
    nodes = list(graph.nodes)
    i_index = nodes.index(i)
    j_index = nodes.index(j)

    # finding the factors to throw out--the weights of edges outgoing from our roots j, q
    forbidden_factor_str_list = []

    for root in roots:
        root_index = nodes.index(root)

        L_root = -1 * Lap.row(root_index)
        L_root.col_del(root_index)
        L_j_str_list = [str(factor) for factor in list(L_root)]
        forbidden_factor_str_list.extend(L_j_str_list)

    Q_ij = Q_n_minus_2.row(i_index)[j_index]
    expanded_Q_ij = sp.expand(Q_ij)
    expanded_Q_ij_str = str(expanded_Q_ij)
    expanded_Q_ij_str_list = expanded_Q_ij_str.split(' + ')

    filtered_simplified_Q_ij_str_list = filter_by_forbidden_factors(expanded_Q_ij_str_list, forbidden_factor_str_list)

    filtered_simplified_Q_ij_sym_list = forest_weight_string_list_to_forest_weight_sym(filtered_simplified_Q_ij_str_list)

    sum_sym_weights = sum(filtered_simplified_Q_ij_sym_list)
    return sum_sym_weights


def ca_kth_moment_numerator(graph,  sym_lap, Q_n_minus_2, source, target, moment):
    """ calculates the numerator of the k-th moment of a graph using the Q_(n-2) matrix given by the CA recurrence.

    Args:
        graph (networkx.classes.digraph.DiGraph): networkx graph of interent
        Lap (sympy.matrices.dense.MutableDenseMatrix): symbolic laplacian of graph
        Q_n_minus_2 (sympy.matrices.dense.MutableDenseMatrix): Q_(n-2) matrix found using the CA recurrence
        source (str): id of source vertex
        target (str): id of target vertex
        moment (int): moment of interest

    Returns:
        sympy.core.add.Add: symbolic expression for the the numerator of the moment of interest.
    """
    if not isinstance(graph, nx.classes.digraph.DiGraph):
        raise NotImplementedError("graph must be a networkx DiGraph")
    if not isinstance(sym_lap, sp.matrices.dense.MutableDenseMatrix):
        raise NotImplementedError("Lap must be a sympy matrix")
    if not isinstance(Q_n_minus_2, sp.matrices.dense.MutableDenseMatrix):
        raise NotImplementedError("Q_n_minus_2 must be a sympy matrix")
    if not isinstance(source, str) or source not in list(graph.nodes):
        raise NotImplementedError("source must be a string and must be the id of a vertex in graph")
    if not isinstance(target, str) or target not in list(graph.nodes):
        raise NotImplementedError("target must be a string and must be the id of a vertex in graph")
    if not isinstance(moment, int) or moment <= 0:
        raise NotImplementedError("moment must be a natural number")

    I = list(graph.nodes())
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

            sum_weights = sum_sym_weights_jq_roots_ij_path(graph, sym_lap, Q_n_minus_2, [j_n, target], j_n_1, j_n)

            prod_inner_sums *= sum_weights

        sum_prods += prod_inner_sums
    
    return factorial(moment) * sum_prods


def k_moment_fpt_expression(edge_to_weight, edge_to_sym, source, target, moment):
    """calculates the symbolic expression of the moment-th moment of the first passage time from source to target
     of the graph represented by edge_to_weight with the symbols of the edge weights being the values in edge_to_sym

    Args:
        edge_to_weight (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): w} where w is some positive number
        edge_to_sym (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): l} where l is some sympy symbol
        source (str): id of source vertex
        target (str): id of target vertex
        moment (int): moment of interest

    Returns:
        sympy.core.add.Add: symbolic expression of the moment-th moment of the graph over the mean of the graph to the power of moment
    """
    if not isinstance(edge_to_weight, dict):
        raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_weight.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_weight[key], (float, int)):
            raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    
    if not isinstance(edge_to_sym, dict):
        raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_sym.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_sym[key], sp.core.symbol.Symbol):
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")

    graph = g_ops.dict_to_graph(edge_to_weight)
    if not isinstance(source, str) or source not in list(graph.nodes):
        raise NotImplementedError("source must be a string and must be the id of a vertex in graph")
    if not isinstance(target, str) or target not in list(graph.nodes):
        raise NotImplementedError("target must be a string and must be the id of a vertex in graph")
    if not isinstance(moment, int) or moment <= 0:
        raise NotImplementedError("moment must be a natural number")
    
    sym_lap = ca.generate_sym_laplacian(graph, edge_to_sym)
    n = sym_lap.rows
    Q_n_minus_2 = ca.get_sigma_Q_k(sym_lap, n-2)[1]

    numerator = ca_kth_moment_numerator(graph, sym_lap, Q_n_minus_2, source, target, moment)
    q = list(graph.nodes()).index(target)

    denominator = sym_lap.minor(q, q)
    denominator = denominator ** moment

    return numerator / denominator


def splitting_probability(edge_to_sym, terminal_vertices, source, target):
    """Calculates the splitting probability of the graph represented by edge_to_sym from source to target.

    Args:
        edge_to_sym (dict[tuple[str]: sympy.core.symbol.Symbol]): dictionary with the form {('v_1', 'v_2): l_i} 
        terminal_vertices (list[str]): list of the ids of the terminal vertices in the graph represented by edge_to_sym
        source (str): vertex id of source of splitting probability
        target (str): vertex id of target of splitting probability

    Returns:
        sympy.core.mul.Mul: sympy expression of splitting probability
    """
    if not isinstance(edge_to_sym, dict):
        raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_sym.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_sym[key], sp.core.symbol.Symbol):
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
    if not isinstance(terminal_vertices, list):
        raise NotImplementedError("terminal vertices must be a list of the string ids of the terminal vertices in edge_to_sym")
    
    graph = nx.DiGraph()
    graph.add_edges_from(edge_to_sym.keys())
    
    for vertex in terminal_vertices:
        if vertex not in list(graph.nodes()):
            raise NotImplementedError("terminal vertices must be a list of the string ids of the terminal vertices in edge_to_sym")
    if source not in list(graph.nodes):
        raise NotImplementedError("source must a be a vertex of the graph represented by edge_to_sym")
    if target not in list(graph.nodes):
        raise NotImplementedError("target must a be a vertex of the graph represented by edge_to_sym")
    
    sym_lap = ca.generate_sym_laplacian(graph, edge_to_sym)
    n = sym_lap.rows
    m = len(terminal_vertices)
    Q_n_minus_2 = ca.get_sigma_Q_k(sym_lap, n-m)[1]
    denominator = sum_sym_weights_jq_roots_ij_path(graph, sym_lap, Q_n_minus_2, terminal_vertices, target, target)
    numerator = sum_sym_weights_jq_roots_ij_path(graph, sym_lap, Q_n_minus_2, terminal_vertices, source, target)
    return numerator / denominator

    