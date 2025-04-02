"""
Emre Alca
title: kinetic-proofreading-numerics.py
date: 2025-03-14 16:33:19

This file is for numerical experiments on the transient hopfield barrier for terminal core butterfly graphs.
That is, looking at the error ratios for the splitting probability and conditional mean first passage times, as well as their product.
"""

import numpy as np
import json
import tqdm
import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import linearframework.gen_graphs as gen
from linearframework.linear_framework_graph import LinearFrameworkGraph, hill_augmented_graph




def det_matrix_minor(matrix, i, j):
    """calculates the determinant of the i-j first minor of matrix

    Args:
        matrix (numpy.ndarray): _description_
        i (int): 0-axis index
        j (int): 1-axis index

    Returns:
        numpy.float64: determinant of thei-j first minor of matrix
    """
    minor = np.delete(np.delete(matrix,i,axis=0), j, axis=1)
    return np.linalg.det(minor)


def calc_quantities(terminal_butterfly_edge_to_weight, k):
    """given the edge_to_weight dictionary for a terminal butterfly graph, 
    and it's number of proximal vertices
    calculate and return the error ratio in its:
    - splitting probability
    - conditional mean first passage time
    - splitting probability times conditional mean first passage time 

    Args:
        terminal_butterfly_edge_to_weight (dict): edge_to_weight dict for the graph
        k (int): number of proximal vertices

    Returns:
        epsilon_sp, epsilon_cmfpt, trade_off: error ratio in splitting probability, conditional mean first passage time, and their product
    """

    terminal_butterfly = LinearFrameworkGraph(terminal_butterfly_edge_to_weight)
    hill_butterfly_1 = hill_augmented_graph(terminal_butterfly, '1')

    index_pk = hill_butterfly_1.nodes.index(f'p_{k}')
    correct_sp = det_matrix_minor(hill_butterfly_1.Lap, index_pk, index_pk)

    index_pbk = hill_butterfly_1.nodes.index(f'p_bar_{k}')
    incorrect_sp = det_matrix_minor(hill_butterfly_1.Lap, index_pbk, index_pbk)

    epsilon_sp =  incorrect_sp / correct_sp

    correct_to = 0
    incorrect_to = 0

    for j in hill_butterfly_1.nodes:
        index_j = hill_butterfly_1.nodes.index(f'p_bar_{k}')
        hill_butterfly_j = hill_augmented_graph(terminal_butterfly, j)

        jndex_pk = hill_butterfly_j.nodes.index(f'p_{k}')
        correct_to += det_matrix_minor(hill_butterfly_1.Lap, index_j, index_j) * det_matrix_minor(hill_butterfly_j.Lap, jndex_pk, jndex_pk)

        jndex_pbk = hill_butterfly_j.nodes.index(f'p_bar_{k}')
        incorrect_to += det_matrix_minor(hill_butterfly_1.Lap, index_j, index_j) * det_matrix_minor(hill_butterfly_j.Lap, jndex_pbk, jndex_pbk)
    
    trade_off = incorrect_to / correct_to

    epsilon_cmfpt = (1 / epsilon_sp) * trade_off
    
    return epsilon_sp, epsilon_cmfpt, trade_off

def proofreading_monte_carlo(k, alphas, num_samples, equilibrium=False):
    """runs a complete swath of numerical experiments.

    Args:
        k (int): number of terminal vertices
        alphas (numpy.ndarray): np.arange of discrimination factor (alpha) values
        num_samples (int): the number of random graphs to test with at a given alpha
        equilibrium (bool): whether the system is at (True) or away from (False) equilibrium. Defaults to False

    Returns:
        epsilon_sp_by_alpha, epsilon_cmfpt_by_alpha, trade_off_by_alpha: 
            dictionaries holding the sampled results of the error ratio of splitting probability,
            conditional mean first passage time, and their product. Each dictionary has alpha values
            as keys pointing to lists of datapoints with the length num_samples.
    """

    epsilon_sp_by_alpha = {}
    epsilon_cmfpt_by_alpha = {}
    trade_off_by_alpha = {}

    for alpha in tqdm.tqdm(alphas):
        epsilon_sp_by_alpha[alpha] = []
        epsilon_cmfpt_by_alpha[alpha] = []
        trade_off_by_alpha[alpha] = []

        for i in range(num_samples):
            butterfly_t_eq_etw = gen.gen_core_butterfly_dict(alpha, k, equilibrium=equilibrium, tails=True)
            epsilon_sp, epsilon_cmfpt, trade_off =  calc_quantities(butterfly_t_eq_etw, k)

            epsilon_sp_by_alpha[alpha].append(epsilon_sp)
            epsilon_cmfpt_by_alpha[alpha].append(epsilon_cmfpt)
            trade_off_by_alpha[alpha].append(trade_off)

    
    return epsilon_sp_by_alpha, epsilon_cmfpt_by_alpha, trade_off_by_alpha


def plot_transient_results(eq_datapoints, neq_datapoints, alphas, num_samples, k, save=False):
    """plots the transient results of both an equilibrium and nonequilibrium run of proofreading_monte_carlo

    Args:
        eq_datapoints (dict(tuple(dict))): dict containing tuple containing epsilon_sp_by_alpha, epsilon_cmfpt_by_alpha, trade_off_by_alpha at equilibrium
        neq_datapoints (dict(tuple(dict))): dict containing tuple containing epsilon_sp_by_alpha, epsilon_cmfpt_by_alpha, trade_off_by_alpha away from equilibrium
        alphas (numpy.ndarray): numpy array of alpha values
        num_samples (int): number of samples per alpha value
        k (int): number of proximal vertices.
        save (bool): whether or not to save the file. defaults to false.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Numerics for transient proofreading on on k={k} terminal core butterfly graph')

    for alpha in alphas:
        # plotting epsilon_sp datapoints
        ax0.scatter(alpha * np.ones(num_samples), neq_datapoints[k][0][alpha], color='tab:blue', s=5)
        ax0.scatter(alpha * np.ones(num_samples), eq_datapoints[k][0][alpha], color='tab:orange', s=5)

        # plotting epsilon_cmfpt datapoints
        ax1.scatter(alpha * np.ones(num_samples), 1/ np.array(neq_datapoints[k][1][alpha]), color='tab:blue', s=5)
        ax1.scatter(alpha * np.ones(num_samples), 1 / np.array(eq_datapoints[k][1][alpha]), color='tab:orange', s=5)

        #plotting trade_off datapoints
        ax2.scatter(alpha * np.ones(num_samples), neq_datapoints[k][2][alpha], color='tab:blue', s=5)
        ax2.scatter(alpha * np.ones(num_samples), eq_datapoints[k][2][alpha], color='tab:orange', s=5)

    k = int(k)
    # plotting epsilon_sp bounds
    ax0.plot(alphas, 1/alphas, color='black', label='1/alpha', linewidth=2)
    ax0.plot(alphas, 1/(alphas ** k), color='tab:red', label=f'1/(alpha^{k})', linewidth=2)
    ax0.set_xlabel('alpha (dimensionless)')
    ax0.set_ylabel("error ratio (dimensionless)")
    ax0.set_title("splitting probability")

    # plotting epsilon_cmfpt bounds
    ax1.plot(alphas, 1/(alphas), color='black', label='alpha', linewidth=2)
    ax1.plot(alphas, 1/(alphas ** k), color='tab:red', label=f'alpha^{k}', linewidth=2)

    ax1.set_xlabel('alpha (dimensionless)')
    ax1.set_ylabel("error ratio (dimensionless)")


    ax1.set_title(f'1 / conditional mean first passage time')
    # ax1.legend(loc='upper right', handles=handles)

    # plotting trade_off bounds and labels
    ax2.plot(alphas, 1/alphas, color='black', label='1/alpha', linewidth=2)
    ax2.plot(alphas, 1/(alphas ** k), color='tab:red', label=f'1/(alpha^{k})', linewidth=2)
    ax2.set_xlabel('alpha (dimensionless)')
    ax2.set_ylabel("error ratio (dimensionless)")


    equilibrium_patch = mpatches.Patch(color='tab:orange', label='equilibrium')
    non_equilibrium_patch = mpatches.Patch(color='tab:blue', label='nonequilibrium')
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([equilibrium_patch, non_equilibrium_patch])
    ax2.set_title(f'splitting probability * conditional mean first passage time')
    ax2.legend(loc='upper right', handles=handles)
    if save:
        filepath = f'epsilon_sp_k{k}_butterfly_transient_results_{datetime.datetime.now()}.png'
        
        plt.savefig(filepath, format='png')
        print(f'plot saved as {filepath}')

        plt.clf()
        
    else:
        plt.show()

if __name__ == "__main__":
    ks = [5, 7, 9]

    num_samples = 100000
    num_alphas = 150
    max_alpha = 10
    alphas = np.arange(1, max_alpha, (max_alpha - 1)/num_alphas)

    eq_datapoints = {}
    neq_datapoints = {}

    for k in ks:
        print(f'generating epsilon_sp, epsilon_cmfpt, trade_off samples for k={k} terminal butterfly graph . . .')
        
        print('equilibrium')
        eq_epsilon_sp_by_alpha, eq_epsilon_cmfpt_by_alpha, eq_trade_off_by_alpha = proofreading_monte_carlo(k, alphas, True, num_samples)

        eq_datapoints[k] = (eq_epsilon_sp_by_alpha, eq_epsilon_cmfpt_by_alpha, eq_trade_off_by_alpha)

        print('nonequilibrium')
        neq_epsilon_sp_by_alpha, neq_epsilon_cmfpt_by_alpha, neq_trade_off_by_alpha = proofreading_monte_carlo(k, alphas, False, num_samples)

        print('saving data file ... ')

        neq_datapoints[k] = (neq_epsilon_sp_by_alpha, neq_epsilon_cmfpt_by_alpha, neq_trade_off_by_alpha)
        
        out_file = open(f"epsion_cmfpt-k{k}-datapoints-{datetime.datetime.now()}.json", "w")  

        json.dump((neq_datapoints[k], eq_datapoints[k]), out_file, indent = 6)  
            
        out_file.close()  

        print('plotting results . . . ')
        plot_transient_results(eq_datapoints, neq_datapoints, alphas, num_samples, k, save=True)

        print('done!')

