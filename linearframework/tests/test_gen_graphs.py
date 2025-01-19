"""
Emre Alca
title: test_gen_graphs.py
date: 2024-08-17 19:41:51

This page holds tests for the generation of butterfly graphs.
"""

import numpy as np
import linearframework.gen_graphs as gg
import pytest
import sympy as sp


def test_gen_core_butterfly_dict_asserts():
    expected_core_dict = {
        ('1', 'p_1'): 1.1774154985086274,
        ('p_1', '1'): 491.9224950758322,
        ('1', 'p_2'): 504.4082462406216,
        ('p_2', '1'): 0.07429998260135516,
        ('p_1', 'p_2'): 262407330.6797874,
        ('p_2', 'p_1'): 92.51572636262478,
        ('1', 'p_3'): 0.007327531200248026,
        ('p_3', '1'): 0.3467038735465146,
        ('p_2', 'p_3'): 8.879741738791062e-07,
        ('p_3', 'p_2'): 0.2852297479905831,
        ('1', 'p_bar_1'): 1.1774154985086274,
        ('p_bar_1', '1'): 4919.224950758322,
        ('1', 'p_bar_2'): 504.4082462406216,
        ('p_bar_2', '1'): 0.7429998260135516,
        ('p_bar_1', 'p_bar_2'): 262407330.6797874,
        ('p_bar_2', 'p_bar_1'): 92.51572636262478,
        ('1', 'p_bar_3'): 0.007327531200248026,
        ('p_bar_3', '1'): 3.467038735465146,
        ('p_bar_2', 'p_bar_3'): 8.879741738791062e-07,
        ('p_bar_3', 'p_bar_2'): 0.2852297479905831
        }
    generated_core_dict = gg.gen_core_butterfly_dict(10, 3, random_seed=1)

    assert generated_core_dict == expected_core_dict


def test_gen_core_butterfly_dict_raises():

    with pytest.raises(NotImplementedError):
        gg.gen_core_butterfly_dict('oops', 3, random_seed=1)

    with pytest.raises(NotImplementedError):
        gg.gen_core_butterfly_dict(10, 'oops', random_seed=1)

    with pytest.raises(NotImplementedError):
        gg.gen_core_butterfly_dict(10, 3, equilibrium='oops', random_seed=1)

    with pytest.raises(NotImplementedError):
        gg.gen_core_butterfly_dict(10, 3, random_seed='oops')


def test_gen_erlang_asserts():
    expected_erlang_dict = {('1', '2'): 1, ('2', '3'): 1, ('3', '4'): 1, ('4', '5'): 1, ('5', '6'): 1}

    assert gg.gen_erlang_process_dict(6, rate=1) == expected_erlang_dict


def test_gen_erlang_raises():
    with pytest.raises(NotImplementedError):
        gg.gen_erlang_process_dict('oops')
    
    with pytest.raises(NotImplementedError):
        gg.gen_erlang_process_dict(3, 'oops')


def test_gen_core_butterfly_edges_raises():
   with pytest.raises(NotImplementedError):
        gg.gen_core_butterfly_edges(1.5)


def test_gen_core_butterfly_edges_asserts():
    edges = [('1', 'p_1'),
            ('p_1', '1'),
            ('1', 'p_2'),
            ('p_2', '1'),
            ('p_1', 'p_2'),
            ('p_2', 'p_1'),
            ('1', 'p_bar_1'),
            ('p_bar_1', '1'),
            ('1', 'p_bar_2'),
            ('p_bar_2', '1'),
            ('p_bar_1', 'p_bar_2'),
            ('p_bar_2', 'p_bar_1')]
    
    assert gg.gen_core_butterfly_edges(2) == edges


def test_gen_core_butterfly_edge_to_sym_raises():
    with pytest.raises(NotImplementedError):
        gg.gen_core_butterfly_edge_to_sym(1.5)


def test_gen_core_butterfly_edge_to_sym_asserts():
    edge_to_sym_str = "{('1', 'p_1'): l_1, ('p_1', '1'): l_2, ('1', 'p_2'): l_3, ('p_2', '1'): l_4, ('p_1', 'p_2'): l_5, ('p_2', 'p_1'): l_6, ('1', 'p_bar_1'): l_1, ('p_bar_1', '1'): alpha*l_2, ('1', 'p_bar_2'): l_3, ('p_bar_2', '1'): alpha*l_4, ('p_bar_1', 'p_bar_2'): l_5, ('p_bar_2', 'p_bar_1'): l_6}"

    assert str(gg.gen_core_butterfly_edge_to_sym(2) == edge_to_sym_str)
