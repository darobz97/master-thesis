#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 06:58:29 2022

@author: user
"""

import math
import openfermion as of
from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path
# sys.path stores all the locations where python searches for packages
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))



from hamiltonians.hamiltonian_construction import heisenberg_hamiltonian, ising_hamiltonian, ghz_hamiltonian
from benchmarking.constants import TOTAL_SHOTS, N_QUBITS, MIN_SHOTS_ROS, RH, HH, IH, GH, DICT_SMALLEST_EIG, DICT_BIGGEST_COEFF



def construct_hamiltonians(n_qubits):
    '''This function builds four QubitOperators, one for each of the hamitonians.

    Parameters
    ----------
    n_qubits : int
        The number of qubits define the size of the system for which the hamiltonians will be built. 

    Returns
    -------
    QubitOperator x 4
        Each of the operators is one of the hamiltonians of interest
    '''
    
    
    # The smallest eignevalue is -2.6665536436268886
    # 0.549085680004131 [] + 3.182267213548876 [X1] - 2.953513659621575 [Z1 Y2] + 0.46207319902157984 [Z1 Z3]
    # The seed to create this random_hamiltonian is 2
    rh = of.QubitOperator('', 0.549085680004131) + of.QubitOperator('X1', 3.182267213548876) + of.QubitOperator('Z1 Z3', 0.46207319902157984)
    # rh = random_hamiltonian(n_qubits, 5, 2, [-5, 5], rng)
    
    # The smallest eigenvalue is -12.807647106714825
    # 1.0 [X0] - 2.0 [Z0 Z1] - 2.0 [Z0 Z3] + 1.0 [X1] - 2.0 [Z1 Z2] + 1.0 [X2] - 2.0 [Z2 Z3] + 1.0 [X3]
    hh = heisenberg_hamiltonian(n_qubits, 1, -3, 2, periodic=True)
    
    # 1.0 [X0 X1] + 1.0 [X0 X3] - 3.0 [Y0 Y1] - 3.0 [Y0 Y3] + 2.0 [Z0 Z1] + 2.0 [Z0 Z3] + 1.0 [X1 X2] - 3.0 [Y1 Y2] + 2.0 [Z1 Z2] + 1.0 [X2 X3] - 3.0 [Y2 Y3] + 2.0 [Z2 Z3]
    # The smallest eigenvalue is -8.543116820279423
    ih = ising_hamiltonian(n_qubits, 1, -2, periodic=True)
    
    # The smallest eigenvalue is -8.0
    # 2.0 [Z0 Z1] + 2.0 [Z0 Z3] + 2.0 [Z1 Z2] + 2.0 [Z2 Z3]
    gh = ghz_hamiltonian(n_qubits, 2, periodic=True)

    return rh, hh, ih, gh



def calculate_mean_rosalin(shots_runs, results_runs, percentage, total_shots):
    '''This function calculates the average result of all the runs of Rosalin. Because Rosalin adapts the number of shots on each iteration of the algorithm,
    one run will calculate the cost function at certain number of shots, while other run will calculate the cost function at other number of shots.
    This is a problem when aggregating the results of 50 runs, as an average cannot be done directly because of the adaptive shots.
    For this reason, this function defines a percentage and splits the total number of shots into a number of intervals given by the percentage. For instance,
    if the percentage is 5%, then there will be 100%/5% = 20 intervals. For each of these intervals, all the results in the 50 runs that fall in the interval, 
    an average result will be calculated. 
    This technique creates a meaningful approximation of the 50 runs.

    Parameters
    ----------
    shots_runs : list[list[int]]
        A list of lists. Each of the inside lists represents a run of Rosalin and indicates on which number of shots the cost function was calculated. 
    results_runs : list[list[float]]
        A list of lists. Each of the inside lists represents a run of Rosalin and indicates on the results of the cost functions on each of the evaluations. 
    percentage : float
        This parameter indicates how to divide the total number of shots to calculate the average in each of the intervals.
    total_shots : int
        Total number of shots used for the optimisation.

    Returns
    -------
    positions: List[int]
        list of integers showing the number of shots corresponding to each one of the intervals. If the interval is [10000, 15000], the shots chosen for 
        representation will be the middle of the interval, so 12500. This is used to plot the shots in a plot.
        the middle of the interval,
    means: List[float]
        List of floats showing the average cost function for each of the intervals. These results together with the positions are used to plot the 
        benchmarking of Rosalin.
    '''
        
        
    # I assume that both lists have the same number of components and each sublist also has the same amount of elements
    if len(shots_runs) != len(results_runs):
        raise ValueError('Lenght of both lists has to be the same')

    for i in range(len(shots_runs)):
        if len(shots_runs[i]) != len(results_runs[i]):
            raise ValueError(f'Lenght of sublist {i} is not the same on both lists')
            
    dict_results = {}
    list_tuples_shots = []
    
    # This part fills the list of tuples (shots) and the dictionary of results. This is done to be able to sort the number of shots at the same time as the results associated with the shots.
    for i in range(len(shots_runs)):
        for j in range(len(shots_runs[i])):
            identifier = f'{i}_{j}'
            list_tuples_shots.append((identifier, shots_runs[i][j]))
            dict_results[identifier] = results_runs[i][j]
    
    # This part works on sorting both the shots list and the results list.
    sorted_shots = sorted(list_tuples_shots, key=lambda x:x[1]) 
    sorted_results = []           
    for i in range(len(sorted_shots)):
        identifier = sorted_shots[i][0]
        sorted_results.append(dict_results[identifier])
        
    # I take out the identifier as we don't need it anymore (the results are already sorted)
    sorted_shots = [x[1] for x in sorted_shots]    
    
    # Now I take the values that are above the limit
    pos_above_max = None
    for i in range(len(sorted_shots)):
        if sorted_shots[i] > total_shots:
            pos_above_max = i
    if isinstance(pos_above_max, int):
        # Some shots are above the limit, so it's necessary to eliminate those.
        sorted_shots = sorted_shots[:pos_above_max]
        sorted_results = sorted_results[:pos_above_max]
        
        
    shots_per_interval = total_shots*percentage
    print('Shots per interval are', shots_per_interval)
    
    num_intervals = 1/percentage
    
    # This cont takes into account the position in the 'shots' list as it iterates through the whole list.
    # It is used to find the positions that mark the x*n% of the shots, with n=1,2,3...
    # The shots are divided in N equal intervals, e.g. 1000 total shots and separation at 20% will give [20,40,60,80,100], five intervals.
    cont = 0
    positions = []
    means = []
    stdevs = []
    
    # Here I am putting the num_iterations to the closest int, num_iterations is the number of intervals that we are building
    for i in range(int(num_intervals)):
        # cont_start_int is the position where the shots start in a new interval. At the start it is 0 for the first interval.
        cont_start_interval = cont
        # num_shots shows the when it goes from one interval to another
        max_shots_interval = min(shots_per_interval*(i+1), total_shots)
        # The sum of costs in the interval
        sum_res_interval = 0
        # List containing the elements of the interval, to then calculate the standard deviation
        stdev_list = []
        # Getting the second element, the number
        min_shots_interval = sorted_shots[cont_start_interval]
        current_shots = min_shots_interval
        while current_shots < max_shots_interval:
            sum_res_interval += sorted_results[cont]
            stdev_list.append(sorted_results[cont])
            cont += 1
            current_shots = sorted_shots[cont]

        if current_shots != min_shots_interval:
            # There are some results in this interval
            mean = sum_res_interval/(cont-cont_start_interval)
            stdev = np.std(stdev_list)
            positions.append(int(shots_per_interval*(i+1/2)))
            means.append(mean)
            stdevs.append(stdev)
        current_shots = sorted_shots[cont]
        
    return (positions, means, stdevs) 



def relative_measurement(smallest_eig, biggest_coeff, cost):
    '''This function is used to calculate the relative measure given the result of a cost function. The relative value
        is then used to plot the results.

        Parameters
        ----------
        smallest_eig : float
            The smallest eigenvalue of the hamiltonian that is being minimised
        biggest_coeff : float
            The biggest coefficient of the hamiltonian, when expressed as a sum of Pauli strings
        cost : float
            Value of the cost function (the cost function wants to be minimised).


        Returns
        -------
        result: value of the relative measurement. The cost function is converted to this new value 
        which is then plotted.
        '''
        
    
    fraction = (smallest_eig - cost)/(smallest_eig - biggest_coeff)
    result = abs(fraction)
    
    return(result)


def format_results(list_results):
    str_results = str(list_results)

    str_results = str_results.replace(',', '')
    str_results = str_results[1:-1]
    
    return str_results
    


        
        
        
        
