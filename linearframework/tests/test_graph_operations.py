"""
testing functions in linearframework.graph_operations.py
- graph to dict
- dict to graph
"""

import linearframework.graph_operations as g_ops
import linearframework.linear_framework_results as lfr
import networkx as nx
import sympy as sp
import pytest

g_dict = {
        ('1', '2'): 1,
        ('1', '3'): 2,
        ('2', '1'): 3,
        ('2', '3'): 4,
        ('3', '1'): 5,
        ('3', '2'): 6
    }


def test_dict_to_graph_basic():
    """
    tests that the weight of a graph made from a dictionary contains the
    edges in the dictionary and that they all have the correct associated weight
    """

    G = g_ops.dict_to_graph(g_dict)

    for edge in G.edges():
        assert g_dict[edge] == G[edge[0]][edge[1]]["weight"]

def test_dict_to_graph_raises():
    """
    tests the raising of NotImplementedErrors in dict_to_graph()
    """
    with pytest.raises(NotImplementedError):
        g_ops.dict_to_graph('oops')
    
    with pytest.raises(NotImplementedError):
        g_ops.dict_to_graph({'oops': 1})
    
    with pytest.raises(NotImplementedError):
        g_ops.dict_to_graph({(1, '2'): 1})

    with pytest.raises(NotImplementedError):
        g_ops.dict_to_graph({('1', 2): 1})
    
    with pytest.raises(NotImplementedError):
        g_ops.dict_to_graph({('1', '2'): '1'})


def test_graph_to_dict_basic():
    """
    tests that the output of graph_to_dict is an equivalent dictionary to the input to dict_to_graph used to generate a graph.
    """

    G = g_ops.dict_to_graph(g_dict)

    dict_from_graph = g_ops.graph_to_dict(G)
    assert g_dict == dict_from_graph

def test_graph_to_dict_raises():
    """
    tests that using a non-nx.digraph object raises an error in graph_to_dict
    """
    with pytest.raises(NotImplementedError):
        g_ops.graph_to_dict('oops')


def test_edge_to_sym_from_edge_to_weight_asserts():
    expected_dict = {
        ('1', '2'): sp.Symbol('l_1'),
        ('1', '3'): sp.Symbol('l_2'),
        ('2', '1'): sp.Symbol('l_3'),
        ('2', '3'): sp.Symbol('l_4'),
        ('3', '1'): sp.Symbol('l_5'),
        ('3', '2'): sp.Symbol('l_6')}
    
    assert g_ops.edge_to_sym_from_edge_to_weight(g_dict) == expected_dict


def test_make_sym_weight_asserts():
    expected_sym_to_weight = {
        sp.Symbol('l_1'): 1,
        sp.Symbol('l_2'): 2,
        sp.Symbol('l_3'): 3,
        sp.Symbol('l_4'): 4,
        sp.Symbol('l_5'): 5,
        sp.Symbol('l_6'): 6
        }
    
    assert g_ops.make_sym_to_weight(g_dict, g_ops.edge_to_sym_from_edge_to_weight(g_dict)) == expected_sym_to_weight


def test_edges_to_random_weight_dict_asserts():
    expected_edges = {
        ('1', '2'): 0.3177840006884067,
        ('1', '3'): 20.986835607646604,
        ('2', '1'): 0.001001581395585897,
        ('2', '3'): 0.06516215458215692,
        ('3', '1'): 0.0075951323286823896,
        ('3', '2'): 0.0035812246787002297
        }
    assert g_ops.edges_to_random_weight_dict(g_dict.keys(), 1) == expected_edges 


def test_evaluate_at_many_points_asserts():
    expected_datapoints = [
        0.0134273488624399,
        6.84370374583779,
        519.118523890523,
        0.153337855411112,
        0.0190287725832753,
        7.11499491296838e-5,
        2571.01922043714,
        0.000875244740095057,
        3.94000911637864e-5,
        1.90611208775257,
        0.00580635627153307,
        11731.8409705777,
        0.00287442187191628,
        0.00130472186757315,
        0.00107187666346953,
        91.8456916746851,
        1.08531464884090,
        290.702919331544,
        1.86894294083199,
        0.153471675388933,
        3.44815433120454e-5,
        678.879341413862,
        3.17141435717037,
        1.26678667215469e-5,
        0.0201029200292843,
        10149.2245518911,
        1.10268475298061,
        0.000334074480508424,
        31.0169427895468,
        0.00181890399978189,
        157.420110883583,
        6.36399699901444e-6,
        0.000132120351168552,
        10.4581890311949,
        0.000872591666815257,
        863.677820778044,
        0.131939387925270,
        0.209241730139303,
        9.65854245375006e-5,
        0.000565925272884216,
        344154.436817815,
        559713.869946258,
        8.75158137735806e-6,
        1.98375813210118,
        154.125800211550,
        218608.338347087,
        0.00292005643235665,
        4.07385811125134e-6,
        0.000926436657856979,
        665.883307327600]

    k3_edge_to_weight = {
        ('1', '2'): 1,
        ('1', '3'): 2,
        ('2', '1'): 3,
        ('2', '3'): 4,
        ('3', '1'): 5,
        ('3', '2'): 6
    }

    k3_edge_to_sym = g_ops.edge_to_sym_from_edge_to_weight(k3_edge_to_weight)
    k3_second_moment_formula = lfr.k_moment_fpt_expression(k3_edge_to_weight, k3_edge_to_sym, '1', '3', 2)
    k3_second_moment_datapoints = g_ops.evaluate_at_many_points(k3_edge_to_weight, k3_edge_to_sym, k3_second_moment_formula, 50)
    for i in range(len(expected_datapoints)):
        assert (k3_second_moment_datapoints[i] - expected_datapoints[i]) < (k3_second_moment_datapoints[i] * 0.05)


def test_evaluate_at_many_points_raises():
    k3_edge_to_weight = {
        ('1', '2'): 1,
        ('1', '3'): 2,
        ('2', '1'): 3,
        ('2', '3'): 4,
        ('3', '1'): 5,
        ('3', '2'): 6
    }

    k3_edge_to_sym = g_ops.edge_to_sym_from_edge_to_weight(k3_edge_to_weight)
    k3_edge_to_sym = g_ops.edge_to_sym_from_edge_to_weight(k3_edge_to_weight)
    k3_second_moment_formula = lfr.k_moment_fpt_expression(k3_edge_to_weight, k3_edge_to_sym, '1', '3', 2)

    with pytest.raises(NotImplementedError):
        g_ops.evaluate_at_many_points('oops', k3_edge_to_sym, k3_second_moment_formula, 10)
    with pytest.raises(NotImplementedError):
        g_ops.evaluate_at_many_points(k3_edge_to_weight, 'oops', k3_second_moment_formula, 10)
    with pytest.raises(NotImplementedError):
        g_ops.evaluate_at_many_points(k3_edge_to_weight, k3_edge_to_sym, 'oops', 10)
    with pytest.raises(NotImplementedError):
        g_ops.evaluate_at_many_points(k3_edge_to_weight, k3_edge_to_sym, k3_second_moment_formula, 'oops')


def test_edges_to_edge_to_sym_asserts():
    k3_edges = [
        ('1', '2'),
        ('1', '3'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
        ('2', '4'),
        ('3', '5'),
    ]
    str(g_ops.edge_to_sym_from_edges(k3_edges)) == "{('1', '2'): l_1, ('1', '3'): l_2, ('2', '1'): l_3, ('2', '3'): l_4, ('3', '1'): l_5, ('3', '2'): l_6, ('2', '4'): l_7, ('3', '5'): l_8}"


def test_edges_to_edge_to_sym_raises():
    with pytest.raises(NotImplementedError):
        g_ops.edge_to_sym_from_edges('oops')