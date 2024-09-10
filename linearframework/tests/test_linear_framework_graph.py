"""
Emre Alca
title: test_linear_framework_graph.py
date: 2024-09-07 17:14:38

tests functionality of linear_framework_graph.py

that is the ability to create objects of type LinearFrameworkGraph
"""
from linearframework.linear_framework_graph import LinearFrameworkGraph

import pytest
import networkx as nx
import sympy as sp

def test_init_asserts():
    k3_edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
    ]
    k3 = LinearFrameworkGraph(k3_edges)

    assert k3.nodes == ['1', '2', '3']
    assert k3.edges == k3_edges
    assert k3.terminal_nodes == []
    assert str(k3.edge_to_sym) == "{('1', '2'): l_1, ('1', '3'): l_2, ('2', '1'): l_3, ('2', '3'): l_4, ('3', '1'): l_5, ('3', '2'): l_6}"
    assert str(k3.sym_lap) == 'Matrix([[l_1 + l_2, -l_1, -l_2], [-l_3, l_3 + l_4, -l_4], [-l_5, -l_6, l_5 + l_6]])'
    assert isinstance(k3.nx_graph, nx.classes.digraph.DiGraph)

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

    assert k3_2t.nodes == ['1', '2', '3', '4', '5']
    assert k3_2t.edges == k3_2t_edges
    assert k3_2t.terminal_nodes == ['4', '5']
    assert str(k3_2t.edge_to_sym) == "{('1', '2'): l_1, ('1', '3'): l_2, ('2', '1'): l_3, ('2', '3'): l_4, ('3', '1'): l_5, ('3', '2'): l_6, ('2', '4'): l_7, ('3', '5'): l_8}"
    assert str(k3_2t.sym_lap) == 'Matrix([[l_1 + l_2, -l_1, -l_2, 0, 0], [-l_3, l_3 + l_4 + l_7, -l_4, -l_7, 0], [-l_5, -l_6, l_5 + l_6 + l_8, 0, -l_8], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])'

def test_init_raises():
    with pytest.raises(NotImplementedError):
        LinearFrameworkGraph('oops')
    with pytest.raises(NotImplementedError):
        LinearFrameworkGraph(['oops'])
    with pytest.raises(NotImplementedError):
        LinearFrameworkGraph([('1', '2', '3')])

def test_generate_random_edge_to_weight_asserts():
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
    assert k3_2t.generate_random_edge_to_weight(seed=1) == {('1', '2'): 0.3177840006884067, ('1', '3'): 20.986835607646604, ('2', '1'): 0.001001581395585897, ('2', '3'): 0.06516215458215692, ('3', '1'): 0.0075951323286823896, ('3', '2'): 0.0035812246787002297, ('2', '4'): 0.013108749615263331, ('3', '5'): 0.11840345146135145}


def test_generate_random_edge_to_weight():
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
        k3_2t.generate_random_edge_to_weight('oops')


def test_make_sym_weight_asserts():
    expected_sym_to_weight = {
        sp.Symbol('l_1'): 1,
        sp.Symbol('l_2'): 2,
        sp.Symbol('l_3'): 3,
        sp.Symbol('l_4'): 4,
        sp.Symbol('l_5'): 5,
        sp.Symbol('l_6'): 6
        }
    
    k3_dict = {
        ('1', '2'): 1,
        ('1', '3'): 2,
        ('2', '1'): 3,
        ('2', '3'): 4,
        ('3', '1'): 5,
        ('3', '2'): 6,
    }
    k3 = LinearFrameworkGraph(list(k3_dict.keys()))
    
    assert k3.make_sym_to_weight(k3_dict) == expected_sym_to_weight
    assert isinstance(k3.make_sym_to_weight(), dict)
    assert len(k3.make_sym_to_weight().keys()) == len(k3_dict.keys())


def test_make_sym_weight_raises():
    k3_edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
    ]
    k3 = LinearFrameworkGraph(k3_edges)

    with pytest.raises(NotImplementedError):
        k3.make_sym_to_weight('oops')