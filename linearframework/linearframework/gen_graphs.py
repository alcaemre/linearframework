import numpy as np
import networkx as nx
# from spanning_forests import Spanning_Forest

def gen_erlang_process_dict(number_of_states, rate=None):
    """generates the edge-to-weight dictionary of an erlang process. 
    The uniform rate can be pre-set, or generated randomly

    Args:
        number_of_states (int): number of states
        rate (float, optional): uniform rate. Defaults to None.

    Returns:
        dict[tuple[str]: float]: edge-to-weight dictionary of an erlang process
    """
    if not isinstance(number_of_states, (float, int)):
        raise NotImplementedError("number_of_states must be a float or an int")
    if not isinstance(rate, (float, int)) and rate != None:
        raise NotImplementedError("rate must be a float or an int or None")

    if rate == None:
        rate = 10 ** (6 * np.random.rand() - 3)
    
    erlang_dict = {}
    for i in range(1, (number_of_states + 1)):
        erlang_dict[(f'{i}', f'{i + 1}')] = rate

    return erlang_dict

def core_butterfly(alpha: float, on_rates: np.ndarray, off_rates:np.ndarray, m_r: np.ndarray, m=None):
    """ Generates a core butterfly graph. 
        Butterfly graphs are symmetrical with the exception of the incorrect off-rates, which are the correct off-rates (off_rates object) multiplied by alpha.
        This function generates a 'core' butterfly graph with k=len(on_rates)=len(off_rates) proximal vertices and no distal vertices
        and returns the PyGraph object of that graph.
        To make an equilibrium graph, leave m as None. 

    Args:
        alpha (float): the factor the correct off-rates are multiplied by to set the incorrect off-rates
        on_rates (np.ndarray with length k): the on-rates of the correct and incorrect 'wings' of the graph
        off_rates (np.ndarray with length k): _description_
        m_r (np.ndarray with length k-1): rates up the wing of a graph, between proximal vertices
        equilibrium (bool, optional): _description_. Defaults to True.
        m (np.ndarray with length k-1): when nonequilibrium the set of rates down the wing of the graph, between proximal vertices. 
                                        Defaults to None to be set according to the cycle condition to make an equilibrium graph.

    Returns:
        pygraph.PreciseDigraph: the core graph generated.
    """

    g = nx.DiGraph()

    if m is None:
        m = (off_rates[: -1] * on_rates[1 :] * m_r) / (on_rates[: -1] * off_rates[1:])
    
    for wing in ['', '_bar']:

        if wing == '_bar':
            off_rates = alpha * off_rates # setting proofreading asymmetry

        for i in range(len(on_rates)):

            g.add_edge('1', f'p{wing}_{i+1}', weight=on_rates[i]) # ith on rate
            g.add_edge(f'p{wing}_{i+1}', '1', weight=off_rates[i]) # ith off rate

            if i > 0:
                g.add_edge(f'p{wing}_{i}', f'p{wing}_{i+1}', weight=m[i-1]) # m for edge between p_{i-1} and p_{i}
                g.add_edge(f'p{wing}_{i+1}', f'p{wing}_{i}', weight=m_r[i-1]) # m_r for edge between 
    return g


def gen_core_graph(alpha, k, equilibrium = True, random_seed=None):
    """Generates a core butterfly graph with k proximal vertices, with random transitions rates with values between 10^-3 and 10^3.
    When equilibrium is True, the graph is set to maintain the cycle condition, otherwise the rates are totally random.

    Args:
        alpha (float): factor by which the proximal off-rates are asymmetric
        k (int): number of proximal edges
        equilibrium (bool, optional): generates an equilibrium graph when True, and a nonequilibrium graph when False. Defaults to True.

    Returns:
        pygraph.PreciseDigraph: the generated pygraph object.
    """

    rng = np.random.default_rng(random_seed)

    on_rates = 10**(6* rng.random(k) - 3)
    off_rates = 10**(6* rng.random(k) - 3)
    m_r = 10**(6* rng.random(k-1) - 3)

    if equilibrium:
        g = core_butterfly(alpha, on_rates, off_rates, m_r)

    if not equilibrium:
        m = 10**(6* rng.random(k-1) - 3)

        g = core_butterfly(alpha, on_rates, off_rates, m_r, m)
    return g


def gen_core_graph_int_rates(alpha, k, equilibrium = True, random_seed=None):
    """Generates a core butterfly graph with k proximal vertices, with random transitions rates with values between 10^-3 and 10^3.
    When equilibrium is True, the graph is set to maintain the cycle condition, otherwise the rates are totally random.

    Args:
        alpha (float): factor by which the proximal off-rates are asymmetric
        k (int): number of proximal edges
        equilibrium (bool, optional): generates an equilibrium graph when True, and a nonequilibrium graph when False. Defaults to True.

    Returns:
        pygraph.PreciseDigraph: the generated pygraph object.
    """
    max_rate = 10000

    np.random.seed(random_seed)

    on_rates = np.random.randint(1, max_rate, k).astype('float64')
    off_rates = np.random.randint(1, max_rate, k).astype('float64')
    m_r = np.random.randint(1, max_rate, k-1).astype('float64')

    if equilibrium:
        g = core_butterfly(alpha, on_rates, off_rates, m_r)

    if not equilibrium:
        m = np.random.randint(1, max_rate, k)

        g = core_butterfly(alpha, on_rates, off_rates, m_r, m)
    return g


def gen_core_butterfly_dict(alpha, p, equilibrium = True, random_seed=None):
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
    if not isinstance(p, int):
        raise NotImplementedError("butterfly graphs can only have an integer number of proximal vertices")
    if not isinstance(equilibrium, bool):
        raise NotImplementedError("equilibrium must be a bool")
    if not isinstance(random_seed, (float, int)) and random_seed is not None:
        raise NotImplementedError("random_seed must be a float or an int")

    rng = np.random.default_rng(random_seed)

    on_rates = 10**(6* rng.random(p) - 3)
    off_rates = 10**(6* rng.random(p) - 3)
    m_r = 10**(6* rng.random(p-1) - 3)

    if equilibrium:
        m = (off_rates[: -1] * on_rates[1 :] * m_r) / (on_rates[: -1] * off_rates[1:])
    else:
        m = 10**(6* rng.random(p-1) - 3)

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

    return butterfly_dict