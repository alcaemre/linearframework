"""
Emre Alca
title: test_ca_recurrence.py
date: 2024-08-08 13:20:07

includes tests for 
"""

import pytest
import sympy as sp
import networkx as nx
import numpy as np

from linearframework.linear_framework_graph import LinearFrameworkGraph
import linearframework.ca_recurrence as ca

p_2_butterfly_dict = {
    ('1', 'p_1'): 1.1774154985086274,
    ('p_1', '1'): 0.007327531200248026,
    ('1', 'p_2'): 504.4082462406216,
    ('p_2', '1'): 491.9224950758322,
    ('p_1', 'p_2'): 0.00047413513077016573,
    ('p_2', 'p_1'): 0.07429998260135516,
    ('1', 'p_bar_1'): 1.1774154985086274,
    ('p_bar_1', '1'): 0.07327531200248026,
    ('1', 'p_bar_2'): 504.4082462406216,
    ('p_bar_2', '1'): 4919.224950758322,
    ('p_bar_1', 'p_bar_2'): 0.00047413513077016573,
    ('p_bar_2', 'p_bar_1'): 0.07429998260135516
    }
p_2_butterfly = LinearFrameworkGraph(list(p_2_butterfly_dict.keys()))


erlang_weight = 10 ** (6 * np.random.rand() - 3)
e_5_dict = {
    ('1', '2'): erlang_weight,
    ('2', '3'): erlang_weight,
    ('3', '4'): erlang_weight,
    ('4', '5'): erlang_weight
}
e_5 = LinearFrameworkGraph(list(e_5_dict.keys()))

k3_dict = {
    ('1', '2'): 1,
    ('1', '3'): 2,
    ('2', '1'): 3,
    ('2', '3'): 4,
    ('3', '1'): 5,
    ('3', '2'): 6
}
k3 = LinearFrameworkGraph(list(k3_dict.keys()))


def check_each_row_sum_Q_is_sigma(Q, sigma):
    for i in range(Q.rows):
        assert sp.expand(sum(Q.row(i)) - sigma) == 0


def asserts_q_kpo_s_kpo_by_graph(graph):
    """given a graph, goes through all possible Q_k matrices and their associated s_k values.
    We verify that these values are correct by checking that
    (i) for any k, the sum of each row of Q_k is constant and is equal to sigma_k
    (ii) the determinant of the i,i-th minor of the laplacian is the sum of the weights of the spanning forests rooted at i, and thus should be equal to Q_(k-1),ii (or any element in row i)

    Args:
        graph (nx.classes.digraph.DiGraph): graph of interest
    """
    n = len(graph.nodes)
    k = 0
    sym_lap = graph.sym_lap
    eye = sp.eye(sym_lap.shape[0])
    Q = eye
    sigma = 1
    check_each_row_sum_Q_is_sigma(Q, sigma)
    while k < (n - 1):
        sigma = ca.sigma_kpo(sym_lap, Q, k)
        Q = ca.Q_kpo(sym_lap, Q, sigma)
        check_each_row_sum_Q_is_sigma(Q, sigma)

        k += 1
    
    for i in range(n):
        assert sp.expand(sym_lap.minor(i, i) - Q.row(i)[i]) == 0


def test_Q_kpo_and_sigma_kpo_asserts(): 
    """
    we will test these methods in two ways:
    (i) for any k, the sum of each row of Q_k is constant and is equal to sigma_k
    (ii) the determinant of the i,i-th minor of the laplacian is the sum of the weights of the spanning forests rooted at i, and thus should be equal to Q_(k-1),ii (or any element in row i)
    We will test this on 3 graphs:
    - k3 graph (complete graph with 3 vertices)
    - p4 butterfly graph (butterfly with 4 proximal vertices)
    - erlang process on 5 vertices
    """
    graphs = [k3, p_2_butterfly, e_5]

    for graph in graphs:
        asserts_q_kpo_s_kpo_by_graph(graph)

def test_get_sigma_Q_k_raises():

    with pytest.raises(NotImplementedError):
        ca.get_sigma_Q_k('oops', 3)
    

    with pytest.raises(NotImplementedError):
        k3_sym_lap = k3.sym_lap
        ca.get_sigma_Q_k(k3_sym_lap, 'oops')



