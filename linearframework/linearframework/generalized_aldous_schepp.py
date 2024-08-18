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

def generalized_randomness_parameter(graph,  Lap, Q_n_minus_2, source, target, moment):
    """calculates the symbolic expression of the moment-th moment of the graph over the mean of the graph to the power of moment

    Args:
        graph (nx.DiGraph): networkx graph of interent
        Lap (sympy.matrices.dense.MutableDenseMatrix): symbolic laplacian of graph
        Q_n_minus_2 (sympy.matrices.dense.MutableDenseMatrix): Q_(n-2) matrix found using the CA recurrence
        source (str): id of source vertex
        target (str): id of target vertex
        moment (int): moment of interest

    Returns:
        sympy.core.add.Add: symbolic expression of the moment-th moment of the graph over the mean of the graph to the power of moment
    """
    if not isinstance(graph, nx.classes.digraph.DiGraph):
        raise NotImplementedError("graph must be a networkx DiGraph")
    if not isinstance(Lap, sp.matrices.dense.MutableDenseMatrix):
        raise NotImplementedError("Lap must be a sympy matrix")
    if not isinstance(Q_n_minus_2, sp.matrices.dense.MutableDenseMatrix):
        raise NotImplementedError("Q_n_minus_2 must be a sympy matrix")
    if not isinstance(source, str) or source not in list(graph.nodes):
        raise NotImplementedError("source must be a string and must be the id of a vertex in graph")
    if not isinstance(target, str) or target not in list(graph.nodes):
        raise NotImplementedError("target must be a string and must be the id of a vertex in graph")
    if not isinstance(moment, int) or moment <= 0:
        raise NotImplementedError("moment must be a natural number")
    
    numerator = lfr.ca_kth_moment_numerator(graph,  Lap, Q_n_minus_2, source, target, moment)
    denominator = lfr.ca_kth_moment_numerator(graph,  Lap, Q_n_minus_2, source, target, 1) ** moment
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