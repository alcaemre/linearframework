"""
Emre Alca
title: test_linear_framework_results.py
date: 2024-08-16 20:37:59

This document tests the functions in linear_framework_results.py
"""

import pytest
import networkx as nx
import sympy as sp
import numpy as np

import linearframework.graph_operations as g_ops

from linearframework.linear_framework_graph import LinearFrameworkGraph
import linearframework.ca_recurrence as ca
import linearframework.linear_framework_results as lfr

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
# p_2_butterfly = g_ops.dict_to_graph(p_2_butterfly_dict)

erlang_weight = 10 ** (6 * np.random.rand() - 3)
e_5_dict = {
    ('1', '2'): erlang_weight,
    ('2', '3'): erlang_weight,
    ('3', '4'): erlang_weight,
    ('4', '5'): erlang_weight
}
# e_5 = g_ops.dict_to_graph(e_5_dict)

k3_dict = {
    ('1', '2'): 1,
    ('1', '3'): 2,
    ('2', '1'): 3,
    ('2', '3'): 4,
    ('3', '1'): 5,
    ('3', '2'): 6
}

graph_dicts = [k3_dict, e_5_dict, p_2_butterfly_dict]


def test_steady_state_calculators_asserts():
    for graph_dict in graph_dicts:

        graph = LinearFrameworkGraph(list(graph_dict.keys()))
        steady_states_from_lap = lfr.steady_states_from_sym_lap(graph)
        steady_states_from_Q_k = lfr.steady_states_from_Q_n_minus_1(graph)

        for key in steady_states_from_lap.keys():
            assert sp.expand(steady_states_from_lap[key] - steady_states_from_Q_k[key]) == 0


def test_steady_state_calculators_raises():
 
    k3_2t_edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
        ('2', '4'),
        ('3', '5')
    ]
    k3_2t = LinearFrameworkGraph(k3_2t_edges)

    with pytest.raises(NotImplementedError):
        lfr.steady_states_from_sym_lap(k3_2t)
    with pytest.raises(NotImplementedError):
        lfr.steady_states_from_sym_lap('oops')

    with pytest.raises(NotImplementedError):
        lfr.steady_states_from_Q_n_minus_1(k3_2t)
    with pytest.raises(NotImplementedError):
        lfr.steady_states_from_Q_n_minus_1('oops')


s = [
    '1_p_1*p_2_1*p_bar_1_1',
    '1_p_1*p_2_1*p_bar_1_p_bar_2',
    '1_p_1*p_2_1*p_bar_2_1', 
    '1_p_1*p_2_1*p_bar_2_p_bar_1', 
    '1_p_1*p_2_p_1*p_bar_1_1',
    '1_p_1*p_2_p_1*p_bar_1_p_bar_2',
    '1_p_1*p_2_p_1*p_bar_2_1', 
    '1_p_1*p_2_p_1*p_bar_2_p_bar_1', 
    '1_p_1*p_bar_1_1*p_bar_2_1', 
    '1_p_1*p_bar_1_1*p_bar_2_p_bar_1', 
    '1_p_1*p_bar_1_p_bar_2*p_bar_2_1', 
    '1_p_2*p_2_p_1*p_bar_1_1',
    '1_p_2*p_2_p_1*p_bar_1_p_bar_2',
    '1_p_2*p_2_p_1*p_bar_2_1', 
    '1_p_2*p_2_p_1*p_bar_2_p_bar_1' 
    ]

s_filtered = [
    '1_p_1*p_2_1*p_bar_1_1',
    '1_p_1*p_2_1*p_bar_1_p_bar_2',
    '1_p_1*p_2_p_1*p_bar_1_1',
    '1_p_1*p_2_p_1*p_bar_1_p_bar_2',
    '1_p_2*p_2_p_1*p_bar_1_1',
    '1_p_2*p_2_p_1*p_bar_1_p_bar_2'
    ]

def test_filter_by_forbidden_factors_asserts():
    assert lfr.filter_by_forbidden_factors(s, ['p_1_1', 'p_1_p_2', '0', '0', 'p_bar_2_1', '0', '0', 'p_bar_2_p_bar_1']) == s_filtered

def test_get_j_vecs_from_indices():
    expected_j_vecs = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 2],
        [0, 2, 0],
        [0, 2, 1],
        [0, 2, 2],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 2],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 0],
        [1, 2, 1],
        [1, 2, 2],
        [2, 0, 0],
        [2, 0, 1],
        [2, 0, 2],
        [2, 1, 0],
        [2, 1, 1],
        [2, 1, 2],
        [2, 2, 0],
        [2, 2, 1],
        [2, 2, 2]
        ]
    assert lfr.get_j_vecs_from_indices([0, 1, 2], 3) == expected_j_vecs


def test_ca_kth_moment_numerator_asserts():

    k3 = LinearFrameworkGraph(list(k3_dict.keys()))

    assert str(sp.simplify(lfr._ca_kth_moment_numerator(k3, '1', '3', 1))) == 'l_1 + l_3 + l_4'
    assert str(sp.simplify(lfr._ca_kth_moment_numerator(k3, '1', '3', 2))) == '2*l_1*l_3 + 2*l_1*(l_1 + l_2) + 2*l_1*(l_3 + l_4) + 2*(l_3 + l_4)**2'
    assert str(sp.simplify(lfr._ca_kth_moment_numerator(k3, '1', '3', 3))) == '6*l_1**2*l_3 + 6*l_1*l_3*(l_1 + l_2) + 12*l_1*l_3*(l_3 + l_4) + 6*l_1*(l_1 + l_2)**2 + 6*l_1*(l_1 + l_2)*(l_3 + l_4) + 6*l_1*(l_3 + l_4)**2 + 6*(l_3 + l_4)**3'


def test_ca_kth_moment_numerator_raises():

    k3 = LinearFrameworkGraph(list(k3_dict.keys()))

    k3_2t_edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
        ('2', '4'),
        ('3', '5')
    ]
    k3_2t = LinearFrameworkGraph(k3_2t_edges)

    with pytest.raises(NotImplementedError):
        lfr._ca_kth_moment_numerator('oops', '1', '3', 1)
    with pytest.raises(NotImplementedError):
        lfr._ca_kth_moment_numerator(k3_2t, '1', '3', 1)
    with pytest.raises(NotImplementedError):
        lfr._ca_kth_moment_numerator(k3, 'oops', '3', 1)
    with pytest.raises(NotImplementedError):
        lfr._ca_kth_moment_numerator(k3, '1', 'oops', 1)
    with pytest.raises(NotImplementedError):
        lfr._ca_kth_moment_numerator(k3, '1', '3', 'oops')


def test_k_moment_fpt_expression_asserts():
    # k3_edge_to_sym = g_ops.edge_to_sym_from_edge_to_weight(k3_dict)

    k3 = LinearFrameworkGraph(list(k3_dict.keys()))

    assert str(sp.simplify(lfr.k_moment_fpt_expression(k3, '1', '3', 1))) == "(l_1 + l_3 + l_4)/(l_1*l_4 + l_2*l_3 + l_2*l_4)"
    assert str(sp.simplify(lfr.k_moment_fpt_expression(k3, '1', '3', 2))) == '2*(l_1*l_3 + l_1*(l_1 + l_2) + l_1*(l_3 + l_4) + (l_3 + l_4)**2)/(l_1*l_4 + l_2*l_3 + l_2*l_4)**2'
    assert str(sp.simplify(lfr.k_moment_fpt_expression(k3, '1', '3', 3))) == '6*(l_1**2*l_3 + l_1*l_3*(l_1 + l_2) + 2*l_1*l_3*(l_3 + l_4) + l_1*(l_1 + l_2)**2 + l_1*(l_1 + l_2)*(l_3 + l_4) + l_1*(l_3 + l_4)**2 + (l_3 + l_4)**3)/(l_1*l_4 + l_2*l_3 + l_2*l_4)**3'


def test_k_moment_fpt_expression_raises():
    k3 = LinearFrameworkGraph(list(k3_dict.keys()))

    k3_2t_edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
        ('2', '4'),
        ('3', '5')
    ]
    k3_2t = LinearFrameworkGraph(k3_2t_edges)

    with pytest.raises(NotImplementedError):
        lfr.k_moment_fpt_expression('oops', '1', '3', 1)
    with pytest.raises(NotImplementedError):
        lfr.k_moment_fpt_expression(k3_2t, '1', '3', 1)
    with pytest.raises(NotImplementedError):
        lfr.k_moment_fpt_expression(k3, 'oops', '3', 1)
    with pytest.raises(NotImplementedError):
        lfr.k_moment_fpt_expression(k3, '1', 'oops', 1)
    with pytest.raises(NotImplementedError):
        lfr.k_moment_fpt_expression(k3, '1', '3', 'oops')

def test_splitting_probability_asserts():
    edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
        ('2', '4'),
        ('3', '5')
    ]
    graph = LinearFrameworkGraph(edges)
    assert str(lfr.splitting_probability(graph, '1', '5')) == '(l_1*l_4*l_8 + l_2*l_3*l_8 + l_2*l_4*l_8 + l_2*l_7*l_8)/(l_1*l_4*l_8 + l_1*l_5*l_7 + l_1*l_6*l_7 + l_1*l_7*l_8 + l_2*l_3*l_8 + l_2*l_4*l_8 + l_2*l_6*l_7 + l_2*l_7*l_8)'


def test_splitting_probability_raises():
    k3 = LinearFrameworkGraph(list(k3_dict.keys()))

    k3_2t_edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
        ('2', '4'),
        ('3', '5')
    ]
    k3_2t = LinearFrameworkGraph(k3_2t_edges)
    with pytest.raises(NotImplementedError):
        lfr.splitting_probability('oops', '1', '5')
    with pytest.raises(NotImplementedError):
        lfr.splitting_probability(k3, '1', '5')
    with pytest.raises(NotImplementedError):
        lfr.splitting_probability(k3_2t, 1, '5')
    with pytest.raises(NotImplementedError):
        lfr.splitting_probability(k3_2t, '1', 5)
