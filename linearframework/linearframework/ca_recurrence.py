"""
Emre Alca
title: ca_recurrence.py
date: 2024-08-08 12:31:08
"""

import linearframework.graph_operations as g_ops
import networkx as nx
import sympy as sp
import numpy as np

def generate_sym_laplacian(graph, edge_to_sym):
    """Given a graph, makes sympy symbols for each edge weight, 
    and returns the Laplacian matrix of the graph with symbolic edge weights.
    This function initializes those symbols.

    Args:
        graph (nx.DiGraph): graph of interest
        edge_to_sym (dict[tuple[str]: sympy.core.symbol.Symbol]): dictionary of edges to symbols

    Returns:
        sympy.matrix: sympy Laplacian matrix
    """
    if not isinstance(graph, nx.classes.digraph.DiGraph):
        raise NotImplementedError
    
    if not isinstance(edge_to_sym, dict):
        raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_sym.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_sym[key], sp.core.symbol.Symbol):
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
    
    nodes = list(graph.nodes())
    sym_lap = []
    for i in range(len(nodes)):
        sym_lap.append([])
        for j in range(len(nodes)):
            sym_lap[i].append(0)
    
    for edge in list(graph.edges()):
        i = nodes.index(edge[0])
        j = nodes.index(edge[1])

        sym_lap[i][j] = -edge_to_sym[edge]
        sym_lap[i][i] = sym_lap[i][i] + (- sym_lap[i][j])

    return sp.Matrix(sym_lap)


def sigma_kpo(L, Q_k, k):
    """given some symbolic laplacian L, some matrix Q_k previously calculated by the CA recurrence, and the k,
    calculates sigma of k + 1

    Args:
        L (sympy.matrices.dense.MutableDenseMatrix): symbolic Laplacian matrix of some graph
        Q_k (sympy.matrices.dense.MutableDenseMatrix): matrix of forest weights with k edges
        k (int): the number of edges

    Returns:
        sympy.core.add.Add: sigma of k edges in the graph in question
    """
    if False:
        return 1
    else:
        trace_LQ = (L * Q_k).trace()
        return trace_LQ / (k + 1)


def Q_kpo(L, Q_k, s_kpo):
    """given a Laplacian, a Q_k matrix, sigma_k+1, returns Q_k+1.

    Args:
        L (sympy.matrices.dense.MutableDenseMatrix): symbolic Laplacian of a graph
        Q_k (sympy.matrices.dense.MutableDenseMatrix): matrix of forest weights with k edges
        s_kpo (sympy.core.add.Add): sigma of k+1

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: matrix of forest weights with k+1 edges
    """
    eye = sp.eye(L.shape[0])
    minus_L_Q_k = (-1 * L) * Q_k
    s_kpo_eye = s_kpo * eye
    return minus_L_Q_k + s_kpo_eye


def get_sigma_Q_k(sym_Lap, k):
    """Given the symbolic laplacian of a graph, 
    and the number of edges of the spanning forests of interest,
    uses the Chabotarev-Agaev recurrence to find the Q_k matrix.
    The Q_k is a matrix of elements where each element Q_(k, ij)
    is the sum of the weights of spanning forests with k edges
    and a path from vertex with canonical basis index i to j.

    Args:
        Lap (sympy.matrices.dense.MutableDenseMatrix): symbolic matrix of a graph of interest
        k (int): number of edges in the spanning forests of interest

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: symbolic Q_k matrix
    """
    if not isinstance(sym_Lap, sp.matrices.dense.MutableDenseMatrix):
        raise NotImplementedError("sym_Lap must be a sympy matrix")
    if not isinstance(k, int):
        raise NotImplementedError("k must be an int")
    eye = sp.eye(sym_Lap.shape[0])
    Q = eye
    n = 0
    sigma = 1
    while n < k:
        sigma = sigma_kpo(sym_Lap, Q, n)
        Q = Q_kpo(sym_Lap, Q, sigma)
        n += 1
    return sigma, Q