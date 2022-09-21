#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 07:58:02 2022

@author: user

"""

import numpy as np
import cirq
import openfermion as of

from matplotlib import pyplot as plt
from sympy import *

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from circuit.create_circuit import create_circuit 
from logistics.helpers import calculate_hamiltonian_logistics, write_results, save_figures
from optimisers.run_optimisers import run_lcbcmaes, run_rosalin
from logistics.constants import N_QUBITS, A_MATRIX, b_VECTOR, c_VECTOR, RHO, TOTAL_SHOTS_DICT, MIN_SHOTS_ROS, N_REPETITIONS, DEPTHS


# I need to get the cost functions and average them.
# Also, I need to keep the best parameters


path_parent = Path(__file__).parents[2]
path_parent_txt = path_parent / 'results' / 'logistics' / 'txt_results'
path_parent_figures  = path_parent / 'results' / 'logistics' / 'figures'

path_ros_shots = path_parent_txt / 'rosalin_shots.txt'
path_ros_noisy_res = path_parent_txt / 'rosalin_results_noisy.txt'
path_ros_noiseless_res = path_parent_txt / 'rosalin_results_noiseless.txt'
path_ros_best_params = path_parent_txt / 'best_params_ros.txt'

path_lcb_shots = path_parent_txt / 'lcb_shots.txt'
path_lcb_noisy_res = path_parent_txt / 'lcb_results_noisy.txt'
path_lcb_noiseless_res = path_parent_txt / 'lcb_results_noiseless.txt'
path_lcb_best_params = path_parent_txt / 'best_params_lcb.txt'

# TODO: does this work? See how this works in matplotlib
path_plot = path_parent_figures / 'plot_results.png'
path_bar_graph = path_parent_figures / 'bar_graph_results.png'



def get_logistics_results():
    write_results(N_REPETITIONS, DEPTHS, TOTAL_SHOTS_DICT, path_ros_shots, path_ros_noisy_res, 
                  path_ros_noiseless_res, path_ros_best_params, path_lcb_shots, path_lcb_noisy_res,
                  path_lcb_noiseless_res, path_lcb_best_params)
    


def create_plots_and_graphs():
    save_figures(path_plot, path_bar_graph, path_ros_shots, path_ros_noisy_res, path_ros_noiseless_res, 
                 path_lcb_shots, path_lcb_noisy_res, path_lcb_noiseless_res)
    
    
#get_logistics_results()
create_plots_and_graphs()








