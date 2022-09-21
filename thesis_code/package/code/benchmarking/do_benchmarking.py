#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:38:01 2022

@author: user
"""

from pathlib import Path
# sys.path stores all the locations where python searches for packages
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from benchmarking.helpers_benchmarking.write_results import write_results 
from benchmarking.helpers_benchmarking.save_figures import save_figures
from benchmarking.constants import N_REPETITIONS, TOTAL_SHOTS, N_QUBITS, MIN_SHOTS_ROS, RH, HH, IH, GH, DICT_SMALLEST_EIG, DICT_BIGGEST_COEFF


path_parent = Path(__file__).parents[2]
path_parent = path_parent / 'results' / 'benchmarking' / 'txt_results'

path_ros_shots = path_parent / 'rosalin_shots.txt'
path_ros_noisy_res = path_parent / 'rosalin_results_noisy.txt'
path_ros_noiseless_res = path_parent / 'rosalin_results_noiseless.txt'
path_lcb_shots = path_parent / 'lcb_shots.txt'
path_lcb_noisy_res = path_parent / 'lcb_results_noisy.txt'
path_lcb_noiseless_res = path_parent / 'lcb_results_noiseless.txt'


def get_benchmarking_results():
    hamiltonians = [RH, HH, IH, GH]
    
    write_results(N_REPETITIONS, hamiltonians, path_ros_shots, path_ros_noisy_res, path_ros_noiseless_res, path_lcb_shots, path_lcb_noisy_res, path_lcb_noiseless_res)

def create_plots():
    save_figures(path_ros_shots, path_ros_noisy_res, path_ros_noiseless_res, path_lcb_shots, path_lcb_noisy_res, path_lcb_noiseless_res)
    
    
# get_benchmarking_results()
create_plots()
             
                




