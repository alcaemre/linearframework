"""
Emre Alca
title: test_generalized_aldous_schepp.py
date: 2024-08-17 23:35:17
"""

import sympy as sp
import networkx as nx
import numpy as np
from math import factorial
import pytest

from linearframework.linear_framework_graph import LinearFrameworkGraph
import linearframework.gen_graphs as gens
import linearframework.linear_framework_results as lfr
import linearframework.generalized_aldous_schepp as gas

k3_dict = {
    ('1', '2'): 1,
    ('1', '3'): 2,
    ('2', '1'): 3,
    ('2', '3'): 4,
    ('3', '1'): 5,
    ('3', '2'): 6
}
k3 = LinearFrameworkGraph(list(k3_dict.keys()))

def test_aldous_schepp_results_assert():
    for n in range(2, 6):
        for m in range(1, 6):
            erlang_dict = gens.gen_erlang_process_dict(n, rate=10)
            erlang = LinearFrameworkGraph(list(erlang_dict.keys()))

            sym_generalized_randomness_param = gas.generalized_randomness_parameter(erlang, '1', f'{n}', m)

            free_symbols = sym_generalized_randomness_param.free_symbols

            e_r = sp.symbols('e_r')

            symbol_dict = {}
            for free_symbol in free_symbols:
                symbol_dict[free_symbol] = e_r

            generalized_randomness_param = float(sym_generalized_randomness_param.subs(symbol_dict))

            ga_result = gas.guzman_alca_equation(n, m)

            assert generalized_randomness_param == ga_result


def test_generalized_randomness_parameter_raises():

    with pytest.raises(NotImplementedError):
        gas.generalized_randomness_parameter('oops', '1', '3', 1)
        
    with pytest.raises(NotImplementedError):
        gas.generalized_randomness_parameter(k3, 'oops', '3', 1)
    with pytest.raises(NotImplementedError):
        gas.generalized_randomness_parameter(k3,'1', 'oops', 1)
    with pytest.raises(NotImplementedError):
        gas.generalized_randomness_parameter(k3, '1', '3', 'oops')


def test_guzman_alca_equation_raises():
    with pytest.raises(NotImplementedError):
        gas.guzman_alca_equation('oops', 5)
    with pytest.raises(NotImplementedError):
        gas.guzman_alca_equation(5, 'oops')