"""
Emre Alca
title: generalized_aldous_schepp.py
date: 2024-08-17 22:30:10

contains central functions related to the numerical exploration of the generalized aldous-schepp results
"""

import sympy as sp
import networkx as nx
import numpy as np
from math import factorial

import linearframework.graph_operations as g_ops
import linearframework.ca_recurrence as ca
import linearframework.linear_framework_results as lfr

def generalized_randomness_parameter(edge_to_weight, edge_to_sym, source, target, moment):
    """calculates the symbolic expression of the moment-th moment of the graph over the mean of the graph to the power of moment

    Args:
        edge_to_weight (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): w} where w is some positive number
        edge_to_sym (dict[tuple[str]: sp.core.symbol.Symbol]): dict of edges in form {('v_1', 'v_2): l} where l is some sympy symbol
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
    n = len(graph.nodes)
    Q_n_minus_2 = ca.get_sigma_Q_k(sym_lap, n-2)[1]
    
    numerator = lfr.ca_kth_moment_numerator(graph,  sym_lap, Q_n_minus_2, source, target, moment)
    denominator = lfr.ca_kth_moment_numerator(graph,  sym_lap, Q_n_minus_2, source, target, 1) ** moment
    return numerator / denominator


def guzman_alca_equation(n, m):
    """
    calculates the m-th generalized randomness parameter of an erlang process on n vertices

    Args:
        n (int): the number of vertices in the erlang process.
        m (int): the moment in the numerator, and the exponent in the denominator of the generalized randomness parameter.

    Returns:
        float: the m-th generalized randomness parameter of an erlang process on n vertices
    """
    if not isinstance(n, int) or n <= 1:
        raise NotImplementedError("n must be an integer greater than or equal to 2")
    if not isinstance(m, int) or m <= 0:
        raise NotImplementedError("m must be a natural number")
    
    numerator = factorial(n + m - 2)

    denominator = factorial(n - 2)
    denominator *= (n-1)**m
    return numerator / denominator