"""
this file contains basic functions on graphs
- graph to dict
- dict to graph
"""

import networkx as nx
import sympy as sp
import numpy as np
import tqdm

def graph_to_dict(graph):
    """given a networkx graph, creates a dictionary associating each edge to its weight

    Args:
        graph (nx.DiGraph): graph of interest

    Returns:
        dict[tuple[str]:int or float]: dictionary of edges to edge weights
    """
    if not isinstance(graph, nx.classes.digraph.DiGraph):
        raise NotImplementedError
    
    graph_dict = {}
    for edge in graph.edges():
        graph_dict[(str(edge[0]), str(edge[1]))] = graph[edge[0]][edge[1]]['weight']
    return graph_dict


def dict_to_graph(edge_to_weight):
    """given a dictionary of 2-tuples of vertex id's pointing to float weights,
    generates an nx.digraph with the edges in the keys and the weights in the values.

    Args:
        edge_to_weight (dict[tuple[str], float or int]): dictionary with 2-tuples of strings as keys and floats or ints as values.

    Returns:
        nx.DiGraph: graph with the edges in edge_to_weight.keys() and the associated weights in edge_to_weight.values()
    """
    if not isinstance(edge_to_weight, dict):
        raise NotImplementedError("graph_dict must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_weight.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("graph_dict must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_weight[key], (float, int)):
            raise NotImplementedError("graph_dict must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    

    graph = nx.DiGraph()

    for edge in edge_to_weight.keys():
        graph.add_edge(edge[0], edge[1], weight = edge_to_weight[edge])
    return graph


def edge_to_sym_from_edge_to_weight(edge_to_weight):
    """takes a dictionary of edges (tuples of 2 vertex id's, in the form ('v_1', 'v_2')) pointing to the weights of the edges they represent
    and returns a dictionary of the same edges pointing to new sympy symbols.
    These symbols are in the format l_i where i is the order of the initialization of the symbols.

    Args:
        edge_to_weight (dict[tuple[str]: float]): a dictionary of edges to weights

    Returns:
        dict[tuple[str]: sympy.core.symbol.Symbol]: dictionary of edges to symbols (representing weights)
    """
    edges = list(edge_to_weight.keys())
    edge_to_sym = {}
    for i in range(len(edges)):
        edge_to_sym[edges[i]] = sp.symbols(f'l_{i + 1}')
    return edge_to_sym


def make_sym_to_weight(edge_to_weight, edge_to_sym):
    """takes an edge-to-weight dict and an edge-to-sym dict and makes a dictionary where the symbol of an edge points to the weight pointed to by the same edge.

    Args:
        edge_to_weight (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): w} where w is some positive number
        edge_to_sym (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): l} where l is some sympy symbol

    Returns:
        dict[sp.core.symbol.Symbol: float]: symbol_to_float_dict
    """
    if not isinstance(edge_to_weight, dict):
        raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_weight.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_weight[key], (float, int)):
            raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    
    if not isinstance(edge_to_sym, dict):
        raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_sym.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_sym[key], sp.core.symbol.Symbol):
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")

    sym_to_weight = {}
    for edge in edge_to_weight.keys():
        sym_to_weight[edge_to_sym[edge]] = edge_to_weight[edge]
    return sym_to_weight


def edges_to_random_weight_dict(edges, seed=None):
    """given a list (or other iterable) of edges in the form ('v_1', 'v_2'),
    makes a dictionary with the edges as keys pointing at random weights
    sampled from the range [10**(-3), 10**6]

    Args:
        edges (iterable[tuple[str]]): iterable containing the edges
        seed (int, float): seed of random process

    Returns:
        dict[tuple[str]: float]: _description_
    """
    np.random.seed(seed)
    edge_to_weight = {}
    for edge in edges:
        edge_to_weight[(str(edge[0]), str(edge[1]))] = 10 ** ((6 * np.random.rand()) - 3)
    return edge_to_weight


def evaluate_at_many_points(edge_to_weight, edge_to_sym, expression, num_samples):
    """evaluates the sympy expression for some linear framework result

    Args:
        edge_to_weight (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): w} where w is some positive number
        edge_to_sym (dict[tuple[str]: float]): dict of edges in form {('v_1', 'v_2): l} where l is some sympy symbol
        expression (sympy.core.mul.Mul): sympy expression
        num_samples (int): number of samples desired

    Returns:
        list[float]: list of sampled datapoints
    """
    if not isinstance(edge_to_weight, dict):
        raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_weight.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_weight[key], (float, int)):
            raise NotImplementedError("edge_to_weight must be a dictionary in the form {('v_1, 'v_2'): w} where w is a positive number and 'v_1' and 'v_2' are the ids of vertices.")
    
    if not isinstance(edge_to_sym, dict):
        raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
    for key in edge_to_sym.keys():
        if not isinstance(key, tuple) or not isinstance(key[0], str) or not isinstance(key[1], str) or len(key) != 2:
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")
        if not isinstance(edge_to_sym[key], sp.core.symbol.Symbol):
            raise NotImplementedError("edge_to_sym must be a dictionary of edges to sympy symbols in the form {('v_1, 'v_2'): l_i} where l_i is a sympy symbol and 'v_1' and 'v_2' are the ids of vertices.")

    if not isinstance(expression, sp.core.mul.Mul):
        raise NotImplementedError("expression must be a sympy expression")
    if not isinstance(num_samples, int):
        raise NotImplementedError("num_samples must be an int")

    datapoints = []
    for i in tqdm.tqdm(range(num_samples)):
        new_edge_to_weight = edges_to_random_weight_dict(edge_to_weight.keys(), seed=i)
        new_sym_to_weight = make_sym_to_weight(new_edge_to_weight, edge_to_sym)
        new_datapoint = expression.subs(new_sym_to_weight)
        datapoints.append(new_datapoint)
    return datapoints