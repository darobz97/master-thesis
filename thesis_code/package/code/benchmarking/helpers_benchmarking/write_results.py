#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:04:58 2022

@author: user
"""

from pathlib import Path
# sys.path stores all the locations where python searches for packages
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from benchmarking.constants import TOTAL_SHOTS, N_QUBITS, MIN_SHOTS_ROS, RH, HH, IH, GH, DICT_SMALLEST_EIG, DICT_BIGGEST_COEFF, SHOTS_MEASUREMENT_ROS
from benchmarking.helpers_benchmarking.helper_functions import format_results
from optimisers.run_optimisers import run_lcbcmaes, run_rosalin



'''
This function runs both Rosalin and LCB and writes the shots and the cost results to the text files. There are 50 iterations per hamiltonian, and the 
hamiltonians are separated in each file with \n.
hamiltonians is a list of the 4 qubit operators (hamiltonians = [rh, hh, ih, gh])
'''
def write_results(n_repetitions, hamiltonians, path_ros_shots, path_ros_noisy_res, path_ros_noiseless_res, path_lcb_shots, path_lcb_noisy_res, path_lcb_noiseless_res):
    '''Write in different files the results for the runs of the optimisation algorithms.

    Parameters
    ----------
    num_repetitions : int
        The number of times the algorithms will be run. Each of the runs use a different seed. By aggregating several runs, the results represent better the
        performance of the algorithms.
    hamiltonians : List[QubitOperator]
        The hamiltonians that are being minimised by the optimisation algorithms.
    path_ros_shots : str
        path to a txt file where, for each of the runs of Rosalin, the shots of the different iterations are written down. The different runs are separated 
        with a new line (\n) and within a run, the shots are separated with a space ( ).
    path_ros_noisy_res : str
        Path to a txt file where, for each of the runs of Rosalin, the noisy cost function of the different iterations are written down. The different runs 
        are separated with a new line (\n) and within a run, the shots are separated with a space ( ).
    path_ros_noiseless_res : str
        Path to a txt file where, for each of the runs of Rosalin, the noiseless cost function of the different iterations are written down. The different 
        runs are separated with a new line (\n) and within a run, the shots are separated with a space ( ).
    path_lcb_shots : str
        Path to a txt file where, for each of the runs of Rosalin, the shots of the different iterations are written down. The different runs are separated 
        with a new line (\n) and within a run, the shots are separated with a space ( ).
    path_lcb_noisy_res : str
        Path to a txt file where, for each of the runs of LCB-CMAES, the noisy cost function of the different iterations are written down. The different 
        runs are separated with a new line (\n) and within a run, the shots are separated with a space ( ).
    path_lcb_noiseless_res : str
        Path to a txt file where, for each of the runs of LCB-CMAES, the noiseless cost function of the different iterations are written down. The different 
        runs are separated with a new line (\n) and within a run, the shots are separated with a space ( ).

    Returns
    -------
    None (the results are written down in the 6 files specified in the input, and nothing is returned by the function)
    '''
    
    
    # hamiltonians = [RH, HH, IH, GH]
    for i in range(len(hamiltonians)):
        hamiltonian = hamiltonians[i]

        
        for j in range(n_repetitions):
            
            seed = i*50+j
            
            shots_rosalin, costs_n_rosalin, costs_a_rosalin, hist_params = run_rosalin(N_QUBITS, hamiltonian, TOTAL_SHOTS, MIN_SHOTS_ROS, SHOTS_MEASUREMENT_ROS, seed)
            shots_lcb, costs_n_lcb, costs_a_lcb, hist_params, hist_expect = run_lcbcmaes(N_QUBITS, hamiltonian, TOTAL_SHOTS, seed)
            
            # Formatting the result. Originally it is a list, but it is formatted to be used more easily
            # The transformation is  '[a, b, c]' -> 'a b c'
            shots_rosalin_formatted = format_results(shots_rosalin)
            costs_n_rosalin_formatted = format_results(costs_n_rosalin)
            costs_a_rosalin_formatted = format_results(costs_a_rosalin)
            shots_lcb_formatted = format_results(shots_lcb)
            costs_n_lcb_formatted = format_results(costs_n_lcb)
            costs_a_lcb_formatted = format_results(costs_a_lcb)
            
            
            # Writing the Rosalin results
            with open(path_ros_shots, 'a') as f:
                f.write(shots_rosalin_formatted + '\n')
                if j == 49 and i != (len(hamiltonians)-1):
                    f.write('\n')
                
            with open(path_ros_noisy_res, 'a') as f:
                f.write(costs_n_rosalin_formatted + '\n')
                if j == 49 and i != (len(hamiltonians)-1):
                    f.write('\n')
            
            with open(path_ros_noiseless_res, 'a') as f:
                f.write(costs_a_rosalin_formatted + '\n')
                if j == 49 and i != (len(hamiltonians)-1):
                    f.write('\n')
                    
            # Writing the LCB results
            with open(path_lcb_shots, 'a') as f:
                f.write(shots_lcb_formatted + '\n')
                if j == 49 and i != (len(hamiltonians)-1):
                    f.write('\n')
                
            with open(path_lcb_noisy_res, 'a') as f:
                f.write(costs_n_lcb_formatted + '\n')
                if j == 49 and i != (len(hamiltonians)-1):
                    f.write('\n')
            
            with open(path_lcb_noiseless_res, 'a') as f:
                f.write(costs_a_lcb_formatted + '\n')
                if j == 49 and i != (len(hamiltonians)-1):
                    f.write('\n')