#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 07:06:49 2022

@author: user
"""

import openfermion as of

from pathlib import Path
# sys.path stores all the locations where python searches for packages
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from hamiltonians.hamiltonian_construction import random_hamiltonian, heisenberg_hamiltonian, ising_hamiltonian, ghz_hamiltonian


# total shots in an optimization run, used for both Rosalin and LCB-CMAES
TOTAL_SHOTS = 100000
# number of qubits used in the benchmarking problems
N_QUBITS = 4
# minimum number of shots used by Rosalin to estimate a component of the gradient
MIN_SHOTS_ROS = 10
# shots used to estimate the expected value in Rosalin once the new parameters are found
SHOTS_MEASUREMENT_ROS = 500
N_REPETITIONS = 50
# the four hamiltonians that are tested in the benchmarking. In order they are:
# random hamiltonian, heisenberg hamiltonian, ising hamiltonian, GHZ hamiltonian
RH = of.QubitOperator('', 0.549085680004131) + of.QubitOperator('X1', 3.182267213548876) + of.QubitOperator('Z1 Z3', 0.46207319902157984)
HH = heisenberg_hamiltonian(N_QUBITS, 1, -3, 2, periodic=True)
IH = ising_hamiltonian(N_QUBITS, 1, -2, periodic=True)
GH = ghz_hamiltonian(N_QUBITS, 2, periodic=True)
# dictionary containing the smallest eigenvalues of each of the four hamiltonians
# these values are used to normalize the cost functions so that all the plots go towards zero
DICT_SMALLEST_EIG = {'rh': -2.6665536436268886, 'hh': -12.807647106714825,
                     'ih': -8.543116820279423, 'gh': -8.0}
# dictionary containing the biggest coefficients in each of the four hamiltonians
# these values are used in the relative measure of the hamiltonians
DICT_BIGGEST_COEFF = {'rh':3.182267213548876, 'hh':1.0, 
                      'ih':2.0, 'gh':2.0}
