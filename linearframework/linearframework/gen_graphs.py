import numpy as np
import networkx as nx

def gen_erlang_process_dict(number_of_states, rate=None):
    """generates the edge-to-weight dictionary of an erlang process. 
    The uniform rate can be pre-set, or generated randomly

    Args:
        number_of_states (int): number of states
        rate (float, optional): uniform rate. Defaults to None.

    Returns:
        dict[tuple[str]: float]: edge-to-weight dictionary of an erlang process
    """
    if not isinstance(number_of_states, (float, int)) or number_of_states < 2:
        raise NotImplementedError("number_of_states must be an int and number_of_states > 1")
    if not isinstance(rate, (float, int)) and rate != None:
        raise NotImplementedError("rate must be a float or an int or None")

    if rate == None:
        rate = 10 ** (6 * np.random.rand() - 3)
    
    erlang_dict = {}
    for i in range(1, (number_of_states)):
        erlang_dict[(f'{i}', f'{i + 1}')] = rate

    return erlang_dict


def gen_core_butterfly_dict(alpha, num_prox_vertices, equilibrium=True, tails=False, random_seed=None):
    """generates a dictionary of edges and weights for a core butterfly graph

    Args:
        alpha (int or float): discrimination factor
        p (int): number of proximal vertices in each wing
        equilibrium (bool, optional): True if the desired graph is at equilibrium, false if not. Defaults to True.
        random_seed (int or float, optional): seed for random generation of numbers. Defaults to None.

    Returns:
        dict[tuple[str], float]: edge to weight dict of a butterfly graph with p proximal vertices in each wing
    """
    if not isinstance(alpha, (float, int)):
        raise NotImplementedError("alpha must be a float or an int")
    if not isinstance(num_prox_vertices, int):
        raise NotImplementedError("butterfly graphs can only have an integer number of proximal vertices")
    if not isinstance(equilibrium, bool):
        raise NotImplementedError("equilibrium must be a bool")
    if not isinstance(random_seed, (float, int)) and random_seed is not None:
        raise NotImplementedError("random_seed must be a float or an int")

    rng = np.random.default_rng(random_seed)

    on_rates = 10**(6* rng.random(num_prox_vertices) - 3)
    off_rates = 10**(6* rng.random(num_prox_vertices) - 3)
    m_r = 10**(6* rng.random(num_prox_vertices-1) - 3)
    exit_rate = 10**(6* rng.random() - 3)

    if equilibrium:
        m = (off_rates[: -1] * on_rates[1 :] * m_r) / (on_rates[: -1] * off_rates[1:])
    else:
        m = 10**(6* rng.random(num_prox_vertices-1) - 3)

    butterfly_dict = {}
    for wing in ['', '_bar']:

        if wing == '_bar':
            off_rates = alpha * off_rates # setting proofreading asymmetry

        for i in range(len(on_rates)):

            butterfly_dict[('1', f'p{wing}_{i+1}')] = on_rates[i] # ith on rate
            butterfly_dict[(f'p{wing}_{i+1}', '1')] = off_rates[i] # ith off rate

            if i > 0:
                butterfly_dict[(f'p{wing}_{i}', f'p{wing}_{i+1}')] = m[i-1] # m for edge between p_{i-1} and p_{i}
                butterfly_dict[(f'p{wing}_{i+1}', f'p{wing}_{i}')] = m_r[i-1] # m_r for edge between 
            
        if tails:
            butterfly_dict[(f'p{wing}_{len(on_rates)}', f'e{wing}')] = exit_rate

    return butterfly_dict


def gen_core_butterfly_edges(p):
    """generates the list of edges in a butterfly graph ith p proofreading steps

    Args:
        p (int): number of proofreading steps

    Returns:
        list[tuple[str]]: list of edges in a butterfly graph with p proofreading steps
    """
    if not isinstance(p, int):
        raise NotImplementedError("butterfly graphs can only have an integer number of proximal vertices")

    edges = []

    for wing in ['', '_bar']:
        for i in range(p):
            edges.append(('1', f'p{wing}_{i+1}'))
            edges.append((f'p{wing}_{i+1}', '1'))

            if i > 0:
                edges.append((f'p{wing}_{i}', f'p{wing}_{i+1}'))
                edges.append((f'p{wing}_{i+1}', f'p{wing}_{i}'))

    return edges