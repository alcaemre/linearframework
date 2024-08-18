"""
testing functions in linearframework.graph_operations.py
- graph to dict
- dict to graph
"""

import linearframework.graph_operations as g_ops
import networkx as nx
import pytest

def test_dict_to_graph_basic():
    """
    tests that the weight of a graph made from a dictionary contains the
    edges in the dictionary and that they all have the correct associated weight
    """
    g_dict = {
        ('1', '2'): 1,
        ('1', '3'): 2,
        ('2', '1'): 3,
        ('2', '3'): 4,
        ('3', '1'): 5,
        ('3', '2'): 6
    }

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
    g_dict = {
        ('1', '2'): 1,
        ('1', '3'): 2,
        ('2', '1'): 3,
        ('2', '3'): 4,
        ('3', '1'): 5,
        ('3', '2'): 6
    }

    G = g_ops.dict_to_graph(g_dict)

    dict_from_graph = g_ops.graph_to_dict(G)
    assert g_dict == dict_from_graph

def test_graph_to_dict_raises():
    """
    tests that using a non-nx.digraph object raises an error in graph_to_dict
    """
    with pytest.raises(NotImplementedError):
        g_ops.graph_to_dict('oops')
