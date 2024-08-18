"""
Emre Alca
title: ca_recurrence.py
date: 2024-08-08 12:31:08
"""

import linearframework.graph_operations as g_ops
import networkx as nx
import sympy as sp
import numpy as np

def graph_to_sym_laplacian(graph):
    """Given a graph, makes sympy symbols for each edge weight, 
    and returns the Laplacian matrix of the graph with symbolic edge weights.
    This function initializes those symbols.

    Args:
        graph (nx.DiGraph): graph of interest

    Returns:
        sympy.matrix: sympy Laplacian matrix
    """
    if not isinstance(graph, nx.classes.digraph.DiGraph):
        raise NotImplementedError
    
    nodes = list(graph.nodes())
    sym_lap = []
    for i in range(len(nodes)):
        sym_lap.append([])
        for j in range(len(nodes)):
            sym_lap[i].append(0)
    
    # graph_dict = graph_to_dict(graph)
    for edge in list(graph.edges()):
        i = nodes.index(edge[0])
        j = nodes.index(edge[1])

        sym_lap[i][j] = -sp.symbols(f'{edge[0]}_{edge[1]}')
        sym_lap[i][i] = sym_lap[i][i] + (- sym_lap[i][j])

    return sp.Matrix(sym_lap)


def trace(M):
    """Calculates the trace of a sympy matrix

    Args:
        M (sympy.matrices.dense.MutableDenseMatrix): a sympy matrix

    Returns:
        sympy.core.add.Add: trace of M
    """
    if not isinstance(M, sp.matrices.dense.MutableDenseMatrix):
        raise NotImplementedError("M must be a sympy matrix")
    
    tr = 0
    for i in range(M.shape[0]):
        row = M.row(i)
        tr += row[i]
    return tr


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
    trace_LQ = trace(L * Q_k)
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
    sigma = sigma_kpo(sym_Lap, Q, n)
    while n < k:
        sigma = sigma_kpo(sym_Lap, Q, n)
        Q = Q_kpo(sym_Lap, Q, sigma)
        n += 1
    return sigma, Q