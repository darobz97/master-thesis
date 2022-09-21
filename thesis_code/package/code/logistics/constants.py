#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:48:38 2022

@author: user
"""

import numpy as np

TOTAL_SHOTS_DICT = {1: 1000000,
               3: 1000000,
               5: 1000000}
DEPTHS = [1, 3, 5]
# Minimum shots to estimate a component of the gradient on a given iteration. Used 
# for Rosalin.
MIN_SHOTS_ROS = 10
# Shots used to estimate the cost function when a new candidate is obtained at the 
# end of an iteration. Used for Rosalin.
SHOTS_MEASUREMENT_ROS = 5000
# Number of qubits used in the logistics problem. This number of qubits is related
# to the size of the problem.
N_QUBITS = 12
# The A matrix is used to describe the constraints of the logistics problem. The 
# constraints are of the form Ax = b
A_MATRIX = np.array([np.array([0,0,1,0,1,1,0,0,0,0,0,0]),
     np.array([0,0,0,1,0,0,1,1,0,0,0,0]),
     np.array([0,0,0,0,0,0,0,0,1,0,1,0]),
     np.array([0,0,0,0,0,0,0,0,0,1,0,1]),
     np.array([0,0,0,0,0,0,0,0,0,0,1,1]),
     np.array([1,1,0,0,0,0,0,0,0,0,0,0]),
     np.array([1,0,0,1,0,0,0,0,0,0,0,0]),
     np.array([0,1,1,0,0,0,0,0,0,0,0,0]),
     np.array([0,0,0,0,1,0,1,0,0,1,0,0]),
     np.array([0,0,0,0,0,1,0,1,1,0,0,0])])
# The b vector is the coefficients of the constraints (Ax = b)
b_VECTOR = np.array([1,1,1,1,1,1,1,1,1,1])
# The c vector is the cost vector, used to calculate the cost function in the 
# logistics problem.
c_VECTOR = np.array([2,3,4,4,5,6,4,3,2,2,2,5])
# The constant rho is used as a penalty term when a candidate solution does not 
# meet all the constraints.
RHO = sum([abs(c_i) for c_i in c_VECTOR]) + 1
MIN_EIGENVALUE = -309
# Number of repretitions to get the average result for a more precise graph.
N_REPETITIONS = 5