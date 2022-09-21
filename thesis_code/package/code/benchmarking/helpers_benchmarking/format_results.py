#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:05:24 2022

@author: user
"""


import math
import numpy as np

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from benchmarking.helpers_benchmarking.helper_functions import calculate_mean_rosalin




def get_shots_and_results_rosalin(path_shots, path_results, percentage, total_shots):
    '''This function reads the files where the information of the Rosalin runs is stored (both the shots used and the costs values), 
    formats the results and returns them in a way that can be plotted.

    Parameters
    ----------
    path_shots : str
        Path to the file containing the shots of all the runs of Rosalin. 
    path_results : str
        Path to the file containing the results of all the runs of Rosalin. Note that these results correspond to noisy or noiseless, 
        depending on which file is passed.
    percentage : float
        Percentage that is used to calculate the average of the Rosalin. This paremeters is passed to the function calculate_mean_rosalin 
        definedvin helper_functions.py.
    total_shots : int
        Total shots used for a single run of Rosalin.


    Returns
    -------
    mean_shots_hamiltonians : List[List[int]]
        List of lists. Each of the inside lists contains the shots used in each of the iterations of Rosalin, for a certain run. Overall,
        each of the inside list represents a run of Rosalin.
        
    mean_results_hamiltonian : List[List[float]]
        List of lists. Each of the inside lists contains the results obtained in each of the iterations of Rosalin, for a certain run. 
        Overall, each of the inside list represents a run of Rosalin.
    '''
    

    with open(path_shots) as tf:
        shots_per_h = tf.read().split('\n\n')
        
    with open(path_results) as tf:
        results_per_h = tf.read().split('\n\n')
    
    # This list has 4 elements, one per hamiltonian
    shots = [[] for _ in range(len(shots_per_h))]
    results = [[] for _ in range(len(shots_per_h))]
        
    for i in range(len(shots_per_h)):
        results_hamiltonian = results_per_h[i].split('\n')
        shots_hamiltonian = shots_per_h[i].split('\n')
        for j in range(len(results_hamiltonian)):
            # Somethimes the line is empty ('')
            if (results_hamiltonian[j] != '') and (shots_hamiltonian[j] != ''):
                # Get the shots and results of an individual run
                results_str = results_hamiltonian[j].split(' ')
                shots_str = shots_hamiltonian[j].split(' ')
                results[i].append([float(x) for x in results_str])
                shots[i].append([math.floor(float(x)) for x in shots_str])
                
    mean_shots_hamiltonians = []
    mean_results_hamiltonian = []   
    stdev_results_hamiltonian = []        
    for i in range(len(shots)):
        avg_shots, avg_results, stdev = calculate_mean_rosalin(shots[i], results[i], percentage, total_shots)
        mean_shots_hamiltonians.append(avg_shots)
        mean_results_hamiltonian.append(avg_results)
        stdev_results_hamiltonian.append(stdev)
        
    return mean_shots_hamiltonians, mean_results_hamiltonian, stdev_results_hamiltonian




def get_shots_and_results_lcb(path_shots, path_results):
    '''We can't Withdraw Money without it being in the bank

    Parameters
    ----------
    path_shots : str
        Path to the file containing the shots of all the runs of LCB-CMAES. 
    path_results : str
        Path to the file containing the results of all the runs of LCB-CMAES. Note that these results correspond to noisy or noiseless, 
        depending on which file is passed.


    Returns
    -------
    shots : List[List[int]]
        List of lists. Each of the inside lists contains the shots used in each of the iterations of LCB-CMAES, for a certain run. Overall,
        each of the inside list represents a run of LCB-CMAES.
        
    final_results : List[List[float]]
        List of lists. Each of the inside lists contains the results obtained in each of the iterations of LCB-CMAES, for a certain run. 
        Overall, each of the inside list represents a run of LCB-CMAES.
    '''
    
    
    
    with open(path_shots) as tf:
        shots_per_h = tf.read().split('\n\n')
        
    with open(path_results) as tf:
        results_per_h = tf.read().split('\n\n')
        
    shots_str = shots_per_h[0].split('\n')[0].split(' ')
    shots = [[int(x) for x in shots_str]]*len(shots_per_h)
    
    results = [[] for _ in range(len(shots_per_h))]
    
    for i in range(len(shots_per_h)):
        results_hamiltonian = results_per_h[i].split('\n')
        for j in range(len(results_hamiltonian)):
            # Somethimes the line is empty ('')
            if (results_hamiltonian[j] != ''):
                # Get the shots and results of an individual run
                results_str = results_hamiltonian[j].split(' ')
                # print('The length of the results is', len(results_str))
                results[i].append([float(x) for x in results_str])
                
                
    final_results = []
    stdev_results = []
    num_hamiltonians = len(results)
    num_evals = len(results[0][0])
    num_runs = len(results[0])
    # For each of the hamiltonians            
    for i in range(num_hamiltonians):
        results_mean = []
        results_for_stdev = []
        # Iterate over the function evaluations that are paired to the shots in the plot.
        for j in range(num_evals):
            sum_results = 0
            values_for_stdev = []
            # Iterate along the 50 runs of LCB
            for k in range(num_runs):
                sum_results += results[i][k][j]
                values_for_stdev.append(results[i][k][j])
            results_mean.append(sum_results/num_runs)
            
            #results_for_stdev = results[i][:][j]
            stdev = np.std(values_for_stdev)
            '''print('results_for_stdev are',  results_for_stdev)
            print('the resulting stdev of lcb is', stdev)'''
            results_for_stdev.append(stdev)
            
        final_results.append(results_mean)
        stdev_results.append(results_for_stdev)
    return shots, final_results, stdev_results



# TODO put the number of shots when calling this function
def get_data_benchmarking(path_ros_shots, path_ros_noisy_res, path_ros_noiseless_res, path_lcb_shots, path_lcb_noisy_res, path_lcb_noiseless_res, total_shots, percentage=0.1):    
    '''We can't Withdraw Money without it being in the bank

    Parameters
    ----------
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
    total_shots : int
        Total number of shots for each of the runs of the optimisation algorithms.
    percentage : float
        This parameter indicates how to divide the total number of shots to calculate the average in each of the intervals.


    Returns
    -------
    results : Dict
        This dictionary contains the shots and results (noisy and noiseless) for both Rosalin and LCB-CMAES. This results are then used to plot all the 
        graphs for the benchmarking.
    '''
    
    
    
    
    shots_ros_noisy, results_ros_noisy, stdev_ros_noisy = get_shots_and_results_rosalin(path_ros_shots, path_ros_noisy_res, percentage, total_shots)
    shots_ros_analytic, results_ros_analytic, stdev_ros_analytic = get_shots_and_results_rosalin(path_ros_shots, path_ros_noiseless_res, percentage, total_shots)
    shots_lcb_noisy, results_lcb_noisy, stdev_lcb_noisy = get_shots_and_results_lcb(path_lcb_shots, path_lcb_noisy_res)
    shots_lcb_analytic, results_lcb_analytic, stdev_lcb_analytic = get_shots_and_results_lcb(path_lcb_shots, path_lcb_noiseless_res)
    
    results = {'shots_ros_noisy': shots_ros_noisy, 'results_ros_noisy': results_ros_noisy, 'shots_ros_analytic': shots_ros_analytic,
               'results_ros_analytic': results_ros_analytic, 'stdev_ros_noisy': stdev_ros_noisy, 'stdev_ros_analytic': stdev_ros_analytic, 
               'shots_lcb_noisy': shots_lcb_noisy, 'results_lcb_noisy': results_lcb_noisy, 'shots_lcb_analytic': shots_lcb_analytic, 
               'results_lcb_analytic': results_lcb_analytic, 'stdev_lcb_noisy': stdev_lcb_noisy, 'stdev_lcb_analytic': stdev_lcb_analytic
        }
    
    return results  