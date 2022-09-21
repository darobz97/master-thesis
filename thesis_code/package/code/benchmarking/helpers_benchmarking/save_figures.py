#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:05:08 2022

@author: user
"""

from pathlib import Path
# sys.path stores all the locations where python searches for packages
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from benchmarking.helpers_benchmarking.format_results import get_data_benchmarking
from benchmarking.helpers_benchmarking.plotting import create_plot
from benchmarking.constants import TOTAL_SHOTS, N_QUBITS, MIN_SHOTS_ROS, RH, HH, IH, GH, DICT_SMALLEST_EIG, DICT_BIGGEST_COEFF


def save_figures(path_shots_ros, path_ros_noisy_res, path_ros_noiseless_res, path_shots_lcb, path_lcb_noisy_res, path_lcb_noiseless_res):
    # Information for each of the 4 different plots
    info_plotting = [('noisy', 'relative'), ('analytic', 'relative'), ('noisy', 'absolute'), ('analytic', 'absolute')]
    results = get_data_benchmarking(path_shots_ros, path_ros_noisy_res, path_ros_noiseless_res, path_shots_lcb, 
                                    path_lcb_noisy_res, path_lcb_noiseless_res, TOTAL_SHOTS)
    
    # Plot n.1: noisy relative
    # Plot n.2: noiseless relative
    # Plot n.3: noisy absolute
    # Plot n.4: noiseless absolute
    for i in range(4):
        shots_ros = results[f'shots_ros_{info_plotting[i][0]}']
        shots_lcb = results[f'shots_lcb_{info_plotting[i][0]}']
        results_ros = results[f'results_ros_{info_plotting[i][0]}']
        results_lcb = results[f'results_lcb_{info_plotting[i][0]}']
        stdev_ros = results[f'stdev_ros_{info_plotting[i][0]}']
        stdev_lcb = results[f'stdev_lcb_{info_plotting[i][0]}']
        create_plot(shots_lcb, results_lcb, stdev_lcb, shots_ros, results_ros, stdev_ros, info_plotting[i][1], info_plotting[i][0], DICT_SMALLEST_EIG, DICT_BIGGEST_COEFF)
        
        
        
