"""
Emre Alca
title: test_linear_framework_graph.py
date: 2024-09-07 17:14:38

tests functionality of linear_framework_graph.py

that is the ability to create objects of type LinearFrameworkGraph
"""
from linearframework.linear_framework_graph import LinearFrameworkGraph, hill_augmented_graph, terminalize
from linearframework.gen_graphs import gen_core_butterfly_dict

import pytest
import numpy as np

## TESTING THE INITIALIZATION OF GRAPHS ---------------------------------------------------------

k3_edge_to_weight = {
        (0, 1): 1,
        (1, 0): 2,
        (0, 2): 3,
        (2, 0): 4,
        (1, 2): 5,
        (2, 1): 6,
    }
k3 = LinearFrameworkGraph(k3_edge_to_weight)

k3_2t_edge_to_weight = {
        (0, 1): 1,
        (1, 0): 2,
        (0, 2): 3,
        (2, 0): 4,
        (1, 2): 5,
        (2, 1): 6,
        (1, 3): 7,
        (2, 4): 8,
    }
k3_2t = LinearFrameworkGraph(k3_2t_edge_to_weight)

random_k3_butterfly_edge_to_weight = gen_core_butterfly_dict(10, 3)
butterfly_k3 = LinearFrameworkGraph(random_k3_butterfly_edge_to_weight)

random_terminal_k3_butterfly_edge_to_weight = gen_core_butterfly_dict(10, 3, tails=True)
terminal_butterfly_k3 = LinearFrameworkGraph(random_terminal_k3_butterfly_edge_to_weight)

def test_init_raises():

    with pytest.raises(NotImplementedError):
        LinearFrameworkGraph('oops')
    
    with pytest.raises(NotImplementedError):
        LinearFrameworkGraph({(1, 2, 3):1})

def test_init_assert():

    # k3 asserts ----------------------------------------
    assert k3.edges == list(k3_edge_to_weight.keys())
    assert k3.edge_to_weight == k3_edge_to_weight
    assert k3.terminal_edges == []
    assert k3.terminal_nodes == []
    assert k3.nodes == [0, 1, 2]

    expected_k3_lap = np.array([
        [ -4.,   2.,   4.],
        [  1.,  -7.,   6.],
        [  3.,   5., -10.]
        ])
    assert k3.Lap.all() == expected_k3_lap.all()

    # k3 2t asserts ----------------------------------------
    assert k3_2t.edges == list(k3_2t_edge_to_weight.keys())
    assert k3_2t.edge_to_weight == k3_2t_edge_to_weight
    assert k3_2t.terminal_edges == [(1, 3), (2, 4)]
    assert k3_2t.terminal_nodes == [3, 4]
    assert k3_2t.nodes == [0, 1, 2, 3, 4]

    expected_k3_2t_lap = np.array([
        [ -4.,   2.,   4.,   0.,   0.],
        [  1., -14.,   6.,   0.,   0.],
        [  3.,   5., -18.,   0.,   0.],
        [  0.,   7.,   0.,   0.,   0.],
        [  0.,   0.,   8.,   0.,   0.]
        ])
    assert k3_2t.Lap.all() == expected_k3_2t_lap.all()

    # non-terminal butterfly asserts ----------------------------------------
    assert butterfly_k3.edges == list(random_k3_butterfly_edge_to_weight.keys())
    assert butterfly_k3.edge_to_weight == random_k3_butterfly_edge_to_weight
    assert butterfly_k3.terminal_edges == []
    assert butterfly_k3.terminal_nodes == []

    for i in range(len(butterfly_k3.Lap)):
        assert sum(butterfly_k3.Lap[:, i]) < 10**(-10)
    
    for edge in butterfly_k3.edges:
        source = butterfly_k3.nodes.index(edge[0])
        target = butterfly_k3.nodes.index(edge[1])

        assert butterfly_k3.Lap[target, source] == random_k3_butterfly_edge_to_weight[edge]

    # terminal butterfly asserts ----------------------------------------
    assert terminal_butterfly_k3.edges == list(random_terminal_k3_butterfly_edge_to_weight.keys())
    assert terminal_butterfly_k3.edge_to_weight == random_terminal_k3_butterfly_edge_to_weight
    assert terminal_butterfly_k3.terminal_edges == [('p_3', 'e'), ('p_bar_3', 'e_bar')]
    assert terminal_butterfly_k3.terminal_nodes == ['e', 'e_bar']

    for i in range(len(terminal_butterfly_k3.Lap)):
        assert sum(terminal_butterfly_k3.Lap[:, i]) < 10**(-10)
    
    for edge in terminal_butterfly_k3.edges:
        source = terminal_butterfly_k3.nodes.index(edge[0])
        target = terminal_butterfly_k3.nodes.index(edge[1])

        assert terminal_butterfly_k3.Lap[target, source] == random_terminal_k3_butterfly_edge_to_weight[edge]
    

## TESTING HILL AUGMENTATION ---------------------------------------------------------

def test_hill_augmented_graph_raises():
    with pytest.raises(NotImplementedError):
        hill_augmented_graph('oops', 1)
    
    with pytest.raises(NotImplementedError):
        hill_augmented_graph(k3_2t, 7)


def test_hill_augmented_graph_asserts():
    k3_2t_a0 = hill_augmented_graph(k3_2t, 0)

    expected_k3_2t_a0_lap = np.array([
        [ -4.,   9.,  12.],
        [  1., -14.,   6.],
        [  3.,   5., -18.]])
    assert k3_2t_a0.Lap.all() == expected_k3_2t_a0_lap.all()
    assert k3_2t_a0.terminal_edges == []
    assert k3_2t_a0.terminal_nodes == []


    hill_butterfly_1 = hill_augmented_graph(terminal_butterfly_k3, '1')
    for edge in hill_butterfly_1.edges:
        if edge == ('p_3', '1'):
            assert hill_butterfly_1.edge_to_weight[edge] == terminal_butterfly_k3.edge_to_weight[edge] + terminal_butterfly_k3.edge_to_weight[('p_3', 'e')]
        elif edge == ('p_bar_3', '1'):
            assert hill_butterfly_1.edge_to_weight[edge] == terminal_butterfly_k3.edge_to_weight[edge] + terminal_butterfly_k3.edge_to_weight[('p_bar_3', 'e_bar')]
        else:
            assert hill_butterfly_1.edge_to_weight[edge] == terminal_butterfly_k3.edge_to_weight[edge]
    
    assert hill_butterfly_1.terminal_edges == []
    assert hill_butterfly_1.terminal_nodes == []


## TESTING TERMINALIZE ---------------------------------------
def test_terminalize_raises():
    with pytest.raises(NotImplementedError):
        terminalize('oops', 1)
    
    with pytest.raises(NotImplementedError):
        terminalize(k3, 5)

def test_terminalize_asserts():
    terminalized_k3 = terminalize(k3, 0)

    assert 0 in terminalized_k3.terminal_nodes

    for edge in k3.edges:
        if not 0 == edge[0]:
            assert terminalized_k3.edge_to_weight[edge] == k3.edge_to_weight[edge]