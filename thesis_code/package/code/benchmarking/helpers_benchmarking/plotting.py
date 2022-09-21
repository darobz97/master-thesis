#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:04:37 2022

@author: user
"""

from matplotlib import pyplot as plt

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from benchmarking.helpers_benchmarking.helper_functions import relative_measurement

def create_plot(shots_hamiltonians_lcb, costs_hamiltonians_lcb, stdev_hamiltonians_lcb, shots_hamiltonians_ros, costs_hamiltonians_ros, stdev_hamiltonians_ros, type_measure, simulation_mode, smallest_eig_dict, biggest_coeff_dict):
    '''Create a plot with four subplots. Each subplot shows how Rosalin and LCB-CMAES optimize one of the hamiltonians.

    Parameters
    ----------
    shots_hamiltonians_lcb : str
        Total Money in the account
    costs_hamiltonians_lcb : str
        Total Money in the account
    stdev_hamiltonians_lcb : str
        Total Money in the account
    shots_hamiltonians_ros : str
        Total Money in the account
    costs_hamiltonians_ros : str
        Total Money in the account
    stdev_hamiltonians_ros: str
        Total Money in the account
    type_measure : str
        This paremeter indicates what type of measurement of the cost function will be used. 
        If 'absolute', the cost value will be optimized towards zero.
        If 'relative', the cost value will be normalized using a relative measurement.
    simulation_mode : str
        This parameters indicates what type of results are obtained from the simulation of the quantum computer.
        If 'noisy', the cost values have been approximated using a number of shots.
        If 'noiseless', the cost values are the exact ones.
    smallest_eig_dict : dict
        Dictionary containing the smallest eigenvalues of the hamiltonians. These eigenvalues represent the ground
        state and are used to translate the cost function in the absolute measure. 
    biggest_coeff_dict : dict
        Dictionary containing the biggest coefficient in each of the hamiltonians. These coefficients are used to 
        calculate the relative measure.


    Returns
    -------
    None (the plot is saved but the function does not return any additional information)
    '''
    
    hamiltonians_full_name = ['Random', 'Heisenberg', 'Ising', 'Ghz']
    hamiltonians_short_name = ['rh', 'hh', 'ih', 'gh']
    normalized_costs_lcb = []
    normalized_costs_ros = []
    for (costs_lcb, costs_ros, hamiltonian) in zip(costs_hamiltonians_lcb, costs_hamiltonians_ros, hamiltonians_short_name):
        normalized_cost_lcb = None
        normalized_cost_ros = None
        if type_measure == 'absolute':
            normalized_cost_lcb = [x-smallest_eig_dict[hamiltonian] for x in costs_lcb]
            normalized_cost_ros = [x-smallest_eig_dict[hamiltonian] for x in costs_ros]
        elif type_measure == 'relative':
            normalized_cost_lcb = [relative_measurement(smallest_eig_dict[hamiltonian], biggest_coeff_dict[hamiltonian], cost) for cost in costs_lcb]
            normalized_cost_ros = [relative_measurement(smallest_eig_dict[hamiltonian], biggest_coeff_dict[hamiltonian], cost) for cost in costs_ros]

        normalized_costs_lcb.append(normalized_cost_lcb)
        normalized_costs_ros.append(normalized_cost_ros)
        
    fig, axs = plt.subplots(2,2)
    coordinates = [(0,0), (0,1), (1,0), (1,1)]
    
    for i in range(len(coordinates)):
        coordinate = coordinates[i]
        hamiltonian = hamiltonians_full_name[i]
        shots_lcb = shots_hamiltonians_lcb[i]
        shots_ros = shots_hamiltonians_ros[i]
        costs_lcb = normalized_costs_lcb[i]
        costs_ros = normalized_costs_ros[i]
        stdev_lcb = stdev_hamiltonians_lcb[i]
        stdev_ros = stdev_hamiltonians_ros[i]
        
        costs_lcb_plus_stdev = [x + y for (x,y) in zip(costs_lcb, stdev_lcb)]
        costs_lcb_minus_stdev = [x - y for (x,y) in zip(costs_lcb, stdev_lcb)]
        costs_ros_plus_stdev = [x + y for (x,y) in zip(costs_ros, stdev_ros)]
        costs_ros_minus_stdev = [x - y for (x,y) in zip(costs_ros, stdev_ros)]
        
        # simulation_mode is either noisy or analytic
        axs[coordinate].set_title(f'{hamiltonian}')
        axs[coordinate].plot(shots_lcb, costs_lcb, 'b', label=f'{simulation_mode} cost LCB CMA-ES {type_measure} for {hamiltonian}')
        axs[coordinate].plot(shots_ros, costs_ros, 'g', label=f'{simulation_mode} cost Rosalin {type_measure} for {hamiltonian}')
        
        # Plotting the stdev graphs
        axs[coordinate].plot(shots_lcb, costs_lcb_plus_stdev, 'b', alpha=0.35)
        axs[coordinate].plot(shots_lcb, costs_lcb_minus_stdev, 'b', alpha=0.35)
        axs[coordinate].plot(shots_ros, costs_ros_plus_stdev, 'g', alpha=0.35)
        axs[coordinate].plot(shots_ros, costs_ros_minus_stdev, 'g', alpha=0.35)
        
        # Fill in between the stdev graphs
        axs[coordinate].fill_between(shots_lcb, costs_lcb_plus_stdev, costs_lcb_minus_stdev, facecolor='b', alpha = 0.25)
        axs[coordinate].fill_between(shots_ros, costs_ros_plus_stdev, costs_ros_minus_stdev, facecolor='g', alpha = 0.25)

        if coordinate == (0,0) or coordinate == (0,1):
            axs[coordinate].xaxis.set_visible(False)
            

    # fig.subplots_adjust(bottom=-0.2)
    
    path_parent = Path(__file__).parents[3]
    path_save_fig = path_parent / 'results' / 'benchmarking' / 'figures' / f'plot_{type_measure}_{simulation_mode}'
    
    plt.savefig(path_save_fig, dpi = 300)
    
    

