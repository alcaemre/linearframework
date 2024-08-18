"""
this file contains basic functions on graphs
- graph to dict
- dict to graph
"""

import networkx as nx

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
        graph_dict[(edge[0], edge[1])] = graph[edge[0]][edge[1]]['weight']
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
        raise NotImplementedError("edge_to_weight must be a dictionary")
    for key in edge_to_weight.keys():
        if not isinstance(key, tuple):
            raise NotImplementedError("keys of edge_to_weight must be a tuple of string vertex ids")
        if not isinstance(key[0], str) or not isinstance(key[1], str):
            raise NotImplementedError("keys of edge_to_weight must be a tuple of string vertex ids")
        if not isinstance(edge_to_weight[key], (int,float)):
            raise NotImplementedError("values of edge_to_weight must be of type float or int")

    graph = nx.DiGraph()

    for edge in edge_to_weight.keys():
        graph.add_edge(edge[0], edge[1], weight = edge_to_weight[edge])
    return graph