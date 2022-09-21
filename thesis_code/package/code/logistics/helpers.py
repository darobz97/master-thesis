#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:48:51 2022

@author: user
"""

import numpy as np
import cirq
import openfermion as of
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
# sys.path stores all the locations where python searches for packages
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from optimisers.run_optimisers import run_lcbcmaes, run_rosalin
from logistics.constants import N_QUBITS, MIN_SHOTS_ROS, SHOTS_MEASUREMENT_ROS, A_MATRIX, b_VECTOR, c_VECTOR, RHO, DEPTHS, TOTAL_SHOTS_DICT, MIN_EIGENVALUE
from circuit.create_circuit import create_circuit 



def build_M_matrix(A, b, c, rho):
    '''Build the M matrix that defines the QUBO problem (x^TMx).

    Parameters
    ----------
    A : int
        The number of qubits that will be used in the quantum circuit
    b : QubitOperator
        The hamiltonian that is being minimised by the LCB-CMAES.
    c : int
        Total number of shots used in the optimisation run
    rho : int
        Seed that is used to generate the random numbers. This is fixed to be able to replicate the results
    
    
    Returns
    -------
    M : List[List[int]] 
        Matrix used in the QUBO formulation (x^TMx)
    '''
    
    
    M = rho*np.dot(np.transpose(A), A) + rho*np.diag(-2*np.dot(np.transpose(A), b)) + np.diag(c)
    return M


'''M = build_M_matrix(A_MATRIX, b_VECTOR, c_VECTOR, RHO)
eigenvalues = np.linalg.eigh(M)
print(eigenvalues[0])'''
# min_eig = min(eigenvalues)




def calculate_hamiltonian_logistics(A, b, c, rho, type_hamiltonian):
    M = build_M_matrix(A, b, c, rho)
        
    n_qubits = A.shape[1]
    qubits = cirq.LineQubit.range(n_qubits)
    
    result = None

    if type_hamiltonian == 'pauli_sum':
        pauli_sum = cirq.PauliSum()
        
        # rows
        for i in range(M.shape[0]):
            # columns
            for j in range(M.shape[1]):
                '''coeff = M[i][j]
                pauli_string = None
                if i == j:
                    pauli_string = cirq.PauliString(cirq.Z(qubits[i]))
                else:
                    pauli_string = cirq.PauliString(cirq.Z(qubits[i]))*cirq.PauliString(cirq.Z(qubits[j]))'''
                
                # x = (1-z)/2 = -z/2
                # xixj = 1/4 - z1/4 -z2/4 + z1z2/4
                
                coeff = M[i][j]
                pauli_string = None
                if i == j:
                    # pauli_string = cirq.PauliString(cirq.Z(qubits[i]))
                    pauli_sum += -0.5*float(coeff)*cirq.PauliString(cirq.Z(qubits[i]))
                else:
                    # pauli_string = cirq.PauliString(cirq.Z(qubits[i]))*cirq.PauliString(cirq.Z(qubits[j]))
                    pauli_sum += -0.25*float(coeff)*cirq.PauliString(cirq.Z(qubits[i])) - 0.25*float(coeff)*cirq.PauliString(cirq.Z(qubits[j])) + 0.25*float(coeff)*cirq.PauliString(cirq.Z(qubits[i]))*cirq.PauliString(cirq.Z(qubits[j]))
                # pauli_sum += float(coeff)*pauli_string
        result = pauli_sum
        
    elif type_hamiltonian == 'qubit_operator':
        qubit_operator = of.QubitOperator()
        
        # rows
        for i in range(M.shape[0]):
            # columns
            for j in range(M.shape[1]):
                '''coeff = M[i][j]
                if i == j:
                    part_of_operator = of.QubitOperator(f'Z{i}', float(coeff))
                else:
                    part_of_operator = of.QubitOperator(f'Z{i} Z{j}', float(coeff))
                qubit_operator += part_of_operator'''
                
                coeff = M[i][j]
                if i == j:
                    # part_of_operator = of.QubitOperator(f'Z{i}', -0.5*float(coeff))
                    qubit_operator += of.QubitOperator(f'Z{i}', -0.5*float(coeff))
                else:
                    # part_of_operator = of.QubitOperator(f'Z{i} Z{j}', 0.25*float(coeff))
                    qubit_operator += of.QubitOperator(f'Z{i} Z{j}', 0.25*float(coeff))
                    qubit_operator += of.QubitOperator(f'Z{i}', -0.25*float(coeff))
                    qubit_operator += of.QubitOperator(f'Z{j}', -0.25*float(coeff))
        result = qubit_operator
        
    return result

'''hamiltonian = calculate_hamiltonian_logistics(A_MATRIX, b_VECTOR, c_VECTOR, RHO, 'qubit_operator')
print(hamiltonian)
eigenvalues = of.linalg.eigenspectrum(hamiltonian, N_QUBITS)
print(eigenvalues)'''



def format_results(list_results, last_repetition_bool, last_depth_bool):
    str_results = str(list_results)

    str_results = str_results.replace(',', '')
    str_results = str_results[1:-1]
    
    if not last_repetition_bool:
        str_results += '\n'
    elif last_repetition_bool and not last_depth_bool:
        str_results += '\n\n'
    
    return str_results

#print(build_M_matrix(A_MATRIX, b_VECTOR, c_VECTOR, RHO))
#print(calculate_hamiltonian_logistics(A_MATRIX, b_VECTOR, c_VECTOR, RHO, 'pauli_sum'))





def write_results(n_repetitions, depths, total_shots_dict, path_ros_shots, 
                  path_ros_noisy_res, path_ros_noiseless_res, path_ros_best_params, 
                  path_lcb_shots, path_lcb_noisy_res, path_lcb_noiseless_res, 
                  path_lcb_best_params):
    
    hamiltonian_cirq = calculate_hamiltonian_logistics(A_MATRIX, b_VECTOR, c_VECTOR, RHO, 'pauli_sum')
    hamiltonian_of = calculate_hamiltonian_logistics(A_MATRIX, b_VECTOR, c_VECTOR, RHO, 'qubit_operator')
    
    for i in range(len(depths)):
        depth = depths[i]
        total_shots = total_shots_dict[depth]
        
        last_depth_bool = i == len(depths)-1
        
        for j in range(n_repetitions):
            
            # TODO: This is the normal seed, now I am changing it because I only want to do depths 3 and 5
            # seed = i*j + j + 1
            
            seed = i*j + j + 1 + 5
            
            last_repetition_bool = j == n_repetitions-1
        
            shots_ros, cost_ros_noisy, cost_ros_noiseless, hist_params_ros = run_rosalin(N_QUBITS, hamiltonian_cirq, total_shots, MIN_SHOTS_ROS, SHOTS_MEASUREMENT_ROS, seed, depth=depth)
            best_params_ros = hist_params_ros[-1]
            
            with open(path_ros_best_params, 'a') as f:
                f.write(format_results(best_params_ros, last_repetition_bool, last_depth_bool))
            with open(path_ros_shots, 'a') as f:
                f.write(format_results(shots_ros,  last_repetition_bool, last_depth_bool))
            with open(path_ros_noisy_res, 'a') as f:
                f.write(format_results(cost_ros_noisy,  last_repetition_bool, last_depth_bool))
            with open(path_ros_noiseless_res, 'a') as f:
                f.write(format_results(cost_ros_noiseless,  last_repetition_bool, last_depth_bool))
            
            
            shots_lcb, cost_lcb_noisy, cost_lcb_noiseless, best_params_lcb = run_lcbcmaes(N_QUBITS, hamiltonian_of, total_shots, seed, depth=depth)
                
            with open(path_lcb_best_params, 'a') as f:
                f.write(format_results(best_params_lcb,  last_repetition_bool, last_depth_bool))
            with open(path_lcb_shots, 'a') as f:
                f.write(format_results(shots_lcb,  last_repetition_bool, last_depth_bool))
            with open(path_lcb_noisy_res, 'a') as f:
                f.write(format_results(cost_lcb_noisy,  last_repetition_bool, last_depth_bool))
            with open(path_lcb_noiseless_res, 'a') as f:
                f.write(format_results(cost_lcb_noiseless,  last_repetition_bool, last_depth_bool))
            
                
            
def get_results_of_the_best(depth):
    # list_best = [0.03471061, 0.27526467, -0.39330261, 0.46541619, 0.36858352, -0.13440317, -1.98190109, -0.61484277, -0.59035496, -0.32439535, 0.43709481, -0.24181164, 0.40136804, -0.15787582, -0.19123458, 0.00821595, 0.00896696, -0.34739111, 0.51158291, 0.10430826, -0.12779502, 0.37978508, 1.33202933, 0.70159861, -0.26723383, 0.30535669, -0.71792886, 1.16732369, 0.89349499, -0.59106019, 0.58452741, 0.66033307, 0.18486374, -0.63642506, 0.38902372, 0.18523158, -0.46334234, -0.61427195, -0.1950989, 0.46228024, -0.45265697, -0.32241143, 1.33990163, 1.8187656, 0.72093333, -0.45702329, 0.18141948, -0.63925643, 0.20369304, 0.69378274, 0.25841226, -0.16770694, 0.40575285, -0.54553801, -0.58569503, 0.45240172, -0.25095585, 0.48910142, 0.13301601, 0.52019561, -0.05974707, 0.22138607, 0.46742136, 0.97569353, -0.18557372, -0.12821656, 0.51789109, 0.4981309, 0.75954664, -0.44751773, 0.40594528, 0.4468324, 0.17387933, 1.24168445, -1.30328486, 0.049341, 0.04687361, -0.3324543, -0.2710045, -0.14350763, -0.69923168, -0.24164791, 0.0738232, 0.83027184, 0.48973613, -0.06535788, 0.32787735, 0.04950284, 0.2780325, -0.52603056, -1.00084402, -0.07086744, -0.47952871, 0.42574298, -0.48810077, 0.09907773, 0.82998669, 0.74473151, -0.10796768, -0.26477237, -0.37631598, 0.38113243, 0.24043534, -0.06719809, 0.29129995, -0.99962015, 1.96238762, 1.64247524, -0.24445045, -0.68172272, 0.57618106, 0.059088]
    list_best = [-0.73854475, -0.07499446, 0.201951, 0.12908964, 1.11156263, 0.15904222,
                -1.28228861, -1.64481963, 0.6988627, -0.29676246, -1.37024889, -0.02683531,
                0.01437324, 1.18365471, -0.29493101, -0.13935557, 0.22777663, 1.59073212,
                0.2403008, 0.32110053, 0.10845397, 0.7714213, 0.46333973, 0.27394743,
                -0.91199575, 0.8652514, -1.0477047, 0.59829899, 0.43908691, 0.022884,
                0.4118659, 0.94123281, 0.57442223, -0.28554989, -0.00276327, -1.1496719,
                1.33443206, 0.09535932, -0.42146407, 0.22992409, -0.61289834, 0.35076183,
                -1.86328218, -1.36885916, -0.65333705, -0.51553241, -0.20047771, -1.51596342,
                0.28871729, 0.47499114, -0.41551661, -0.46789507, -0.84677944, -0.11248755,
                0.29426776, 0.02929861, -0.246894, 0.60648471, 0.38090945, 0.59281761,
                -0.13452666, 0.15755749, 0.88021805, -0.61422092, 1.4242716, -0.35024069,
                0.55136975, -0.03953332, -0.22083576, 0.56625118, -0.61842887, 0.16316734,
                0.05994517, -0.14748214, -0.42505043, 0.01178902, -1.71965272, 0.63617848,
                1.20452769, -0.99206535, 0.77949804, -0.43440733, 0.17740187, -0.85319996,
                -1.65250348, 0.28942425, -0.98247896, 1.04429554, 0.80276536, -0.44117252,
                -0.5753383, 0.31718091, -0.47230663, 0.39576446, -0.37892195, -0.407887,
                0.18301308, -0.37645291, -0.07722269, -0.97619706, -0.00939124, 0.25099049,
                -0.17140609, 0.19757936, -0.44333298, 0.39994749, 0.59085716, 0.17875786,
                -0.94256852, -0.70200411, -0.25754668, -0.98660081]
    
    
    qubits = cirq.LineQubit.range(N_QUBITS)
    
    circuit, parameters_dict = create_circuit(N_QUBITS, depth)

    simulator = cirq.Simulator(dtype=np.complex128)
    for val, key in zip(list_best, parameters_dict.keys()):
        parameters_dict[key] = val
    
    circuit = cirq.resolve_parameters(circuit,
            cirq.ParamResolver(parameters_dict))
    
    result = simulator.simulate(circuit, qubit_order=qubits)
    psi = result.final_state_vector
    
    expectations = []
    for pos in range(N_QUBITS):
        print(f'Calculating pauli {pos}')
        pauli_string = cirq.PauliString(cirq.Z(qubits[pos]))
        expectation = pauli_string.expectation_from_state_vector(state_vector=psi, qubit_map={q:number for number,q in enumerate(qubits)})
        print(f'The expectation value of Z_{pos} is {expectation}')
        expectations.append(expectation)
        
    print(expectations)
    
#get_results_of_the_best(1)
           
            
def solve_1(seed, depth):
    hamiltonian_of = calculate_hamiltonian_logistics(A_MATRIX, b_VECTOR, c_VECTOR, RHO, 'qubit_operator')
    shots_lcb, cost_lcb_noisy, cost_lcb_noiseless, best_params_lcb = run_lcbcmaes(N_QUBITS, hamiltonian_of, 400000, seed, depth=depth)
    print('The best params are', best_params_lcb)

#solve_1(2,1)               

    

def sample_circuit(depth_circuit, best_params_path):
    with open(best_params_path, "r") as tf:
        params = tf.read().split(' ')
        
    # Adding the depth here
    circuit, parameters_dict = create_circuit(N_QUBITS, depth_circuit)
        
    simulator = cirq.Simulator()
    
    # Maybe this is not very elegant, but I am doing this in line (so not various solutions in parallel like in the LCB)
    for val, key in zip(params, parameters_dict.keys()):
        parameters_dict[key] = val
    
    # self.circuit continues to be the circuit with unresolved params, and we are resolving them for the circuit varaible here
    circuit = cirq.resolve_parameters(circuit,
            cirq.ParamResolver(parameters_dict))
    
    qubits = cirq.LineQubit.range(N_QUBITS)
    circuit.append(cirq.measure(*qubits, key='result'))
    print('Sample the circuit:')
    samples = simulator.run(circuit, repetitions=1000)
    print(str(samples))



def get_lcb_1_5():
    with open('results/results_lcb_1_5.txt', 'r') as file:
        file_split = file.read().split('\n\n\n')
        print(len(file_split))
    
    results_lcb_1_str = file_split[0].split('\n\n')
    results_lcb_5_str = file_split[1].split('\n\n')
    
    results_lcb_1 = []
    results_lcb_5 = []
    
    for iteration in results_lcb_1_str:
        iteration_parts = iteration.split('\n')
        number_shots = iteration_parts[0].split(' ')[-1]
        number_shots = float(number_shots)
        noiseless_results = iteration_parts[1].split(' ')[-1]
        noiseless_results = float(noiseless_results)
        results_lcb_1.append((number_shots, noiseless_results))
        
    for iteration in results_lcb_5_str:
        iteration_parts = iteration.split('\n')
        number_shots = iteration_parts[0].split(' ')[-1]
        number_shots = float(number_shots)
        noiseless_results = iteration_parts[1].split(' ')[-1]
        noiseless_results = float(noiseless_results)
        results_lcb_5.append((number_shots, noiseless_results))
    print(results_lcb_1)
    print('\n\n\n')
    print(results_lcb_5)
        
'''def create_plot():
    lcb_1_unzip = list(zip(*lcb_1))
    lcb_5_unzip = list(zip(*lcb_5))
    lcb_10_unzip = list(zip(*lcb_10))
    
    ros_1_unzip = list(zip(*rosalin_1))
    ros_5_unzip = list(zip(*rosalin_5))
    ros_10_unzip = list(zip(*rosalin_10))
    
    plt.plot(lcb_1_unzip[0], lcb_1_unzip[1], "g", label = "LCB-1")
    plt.plot(lcb_5_unzip[0], lcb_5_unzip[1], "b", label = "LCB-5")
    plt.plot(lcb_10_unzip[0], lcb_10_unzip[1], "r", label = "LCB-10")
    plt.plot(ros_1_unzip[0], ros_1_unzip[1], "y", label = "Ros-1")
    plt.plot(ros_5_unzip[0], ros_5_unzip[1], "m", label = "Ros-5")
    plt.plot(ros_10_unzip[0], ros_10_unzip[1], "c", label = "Ros-10")
    
    plt.xlim([0, 1000000])
    
    plt.xlabel("Number of shots")
    plt.ylabel("Cost function")
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    
    plt.show()'''
    
    
def colour_value(row):    
    """
    

    Parameters
    ----------
    row : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    
    red = 'background-color: red;'
    green = 'background-color: green;'
    yellow = 'background-color: yellow;'
    
    results = []
    first_element = row[0]
    
    for element in row:
        if abs(element - first_element) < 0.5:
            results.append(green)
        elif abs(element - first_element) < 1:
            results.append(yellow)
        elif abs(element - first_element) < 2:
            results.append(red)    
            

    '''# must return one string per cell in this row
    if row['num_children'] > row['num_pets']:
        return [highlight, default]
    elif row['num_pets'] > row['num_children']:
        return [default, highlight]
    else:
        return [default, default]'''
    return results


def colour_table(results):
    '''
    

    Returns
    -------
    None.

    '''
    '''results = np.array([[1, 0.841, 0.331, -0.076, 0.999, 0.873, 0.184],
                       [-1, 0.203, 0.066, -0.039, -0.992, -0.776, 0.513],
                       [1, 0.207, 0.271, 0.002, 0.997, 0.772, 0.494],
                       [-1, -0.015, 0.030, -0.003, -0.955, -0.762, 0.550],
                       [-1, 0.250, 0.170, 0.034, 0.988, -0.808, -0.378],
                       [-1, 0.003, 0.314, 0.021, -0.976, 0.821, 0.466],
                       [-1, -0.163, 0.147, -0.012, 0.989, 0.853, -0.005],
                       [1, 0.153, -0.075, -0.002, 0.990, 0.740, -0.132],
                       [-1, 0.266, 0.249, -0.005, -0.368, -0.752, 0.261],
                       [1, 0.173, 0.196, 0.009, -0.965, 0.831, 0.484],
                       [1, 0.001, 0.236, 0.006, 0.971, 0.823, 0.751],
                       [-1, -0.163, 0.100, -0.011, -0.860, -0.782, -0.102]])'''
    
    columns = ['Optimum', 'Ros-1', 'Ros-5', 'Ros-10', 'LCB-1', 'LCB-5', 'LCB-10']
    rows = ['$Z_1$', '$Z_2$', '$Z_3$', '$Z_4$', '$Z_5$', '$Z_6$', '$Z_7$', '$Z_8$', '$Z_9$', '$Z_10$', '$Z_11$', '$Z_12$']
    df = pd.DataFrame(results, rows, columns)
    
    df.round(decimals = 2)
    
    table = df.style.apply(colour_value, axis=1)
    
    # html = df.to_html()
        
    print(table.to_html())
    
    

def get_results_simulation(path_best_params_rosalin, path_best_params_lcb, depths):
    """
    

    Returns
    -------
    None.

    """
    qubits = cirq.LineQubit.range(N_QUBITS)
    
    # Separating the 3 different sizes of the circuit
    with open(path_best_params_rosalin, "r") as tf:
        params_ros = tf.read().split('\n\n')
    with open(path_best_params_lcb, "r") as tf:
        params_lcb = tf.read().split('\n\n')
        
        
    final_expectations_ros = []
    final_expectations_lcb = []
    
    algorithms = ['ros', 'lcb']
    
    # Two algorithms
    for algorithm in algorithms:
        # Three depths
        for i in range(len(depths)):
            depth = depths[i]
            params_per_depth = None
            
            if algorithm == 'ros':
                params_per_depth = params_ros[i]
            elif algorithm == 'lcb':
                params_per_depth = params_lcb[i]
            
            expectations_per_depth = []
            
            params_split = params_per_depth.split('\n')
            # Take out the empty strings
            params_split = [i for i in params_split if i]
                        
            # Each one of the 5 repetitions
            for params in params_split:
                
                params = params.split(' ')
                params = [float(param) for param in params]
                circuit, parameters_dict = create_circuit(N_QUBITS, depth)
            
                simulator = cirq.Simulator(dtype=np.complex128)
                for val, key in zip(params, parameters_dict.keys()):
                    parameters_dict[key] = val
                
                circuit = cirq.resolve_parameters(circuit,
                        cirq.ParamResolver(parameters_dict))
                
                result = simulator.simulate(circuit, qubit_order=qubits)
                psi = result.final_state_vector
                
                expectations = []
                for pos in range(N_QUBITS):
                    print(f'Calculating pauli {pos}')
                    pauli_string = cirq.PauliString(cirq.Z(qubits[pos]))
                    expectation = pauli_string.expectation_from_state_vector(state_vector=psi, qubit_map={q:number for number,q in enumerate(qubits)})
                    print(f'The expectation value of Z_{pos} is {expectation}')
                    expectations.append(expectation)
                    
                expectations_per_depth.append(expectations)
                
                '''print('printing results of qubits')
                circuit.append(cirq.measure(*qubits, key='result'))

                print('Sample the circuit:')
                samples = simulator.run(circuit, repetitions=1000)
                print(str(samples))'''
                
            # Make the 5 expectations per depth only 1
            all_expectations = list(zip(*expectations_per_depth))
            avg_expectations = [sum(x)/len(x) for x in all_expectations]
            if algorithm == 'ros':
                final_expectations_ros.append(avg_expectations)
            elif algorithm == 'lcb':
                final_expectations_lcb.append(avg_expectations)
                
    return (final_expectations_ros, final_expectations_lcb)


path_parent = Path(__file__).parents[2]
path_parent = path_parent / 'results' / 'logistics' / 'txt_results'
path_ros_best_params = path_parent / 'best_params_ros.txt'
path_lcb_best_params = path_parent / 'best_params_lcb.txt'

#results = get_results_simulation(path_ros_best_params, path_lcb_best_params, DEPTHS)
#print(results)


def get_correct_decisions():
    optimum_solution = [1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1]
    path_parent = Path(__file__).parents[2]
    path_parent = path_parent / 'results' / 'logistics' / 'txt_results'
    path_ros_decisions = path_parent / 'decisions_rosalin.txt'
    path_lcb_decisions = path_parent / 'decisions_lcb.txt'

    with open(path_ros_decisions, "r") as tf:
        decisions_ros = tf.read().split('\n\n')
    with open(path_lcb_decisions, "r") as tf:
        decisions_lcb = tf.read().split('\n\n')
       
    final_results_ros_mean = []
    final_results_ros_stdev = []
    for decisions in decisions_ros:
        decisions_per_depth = decisions.split('\n')
        decisions_per_depth = [i for i in decisions_per_depth if i]
        values_per_depth_ros = []
        
        for decisions_one_repetition in decisions_per_depth:
            decisions_list = decisions_one_repetition.split(' ')
            decisions_list = [float(decision) for decision in decisions_list]
            
            value = 0
            for x,y in zip(optimum_solution, decisions_list):
                if abs(x+y) < 0.5:
                    value += 1
                elif abs(x+y) < 1:
                    value += 0.5
            print('value in ros is', value)
            values_per_depth_ros.append(value)
        final_results_ros_mean.append(np.mean(values_per_depth_ros))
        final_results_ros_stdev.append(np.std(values_per_depth_ros))
        #print(sum(values_per_depth)/len(values_per_depth))
                
    final_results_lcb_mean = []
    final_results_lcb_stdev = []
    for decisions in decisions_lcb:
        decisions_per_depth = decisions.split('\n')
        decisions_per_depth = [i for i in decisions_per_depth if i]
        values_per_depth_lcb = []

        for decisions_one_repetition in decisions_per_depth:
            decisions_list = decisions_one_repetition.split(' ')
            decisions_list = [float(decision) for decision in decisions_list]
            
            value = 0
            for x,y in zip(optimum_solution, decisions_list):
                if abs(x+y) < 0.5:
                    value += 1
                elif abs(x+y) < 1:
                    value += 0.5
                '''if abs(x+y) < 1:
                    value += 1'''

            print('value in lcb is', value)
            values_per_depth_lcb.append(value)
        final_results_lcb_mean.append(np.mean(values_per_depth_lcb))
        final_results_lcb_stdev.append(np.std(values_per_depth_lcb))
        #print(sum(values_per_depth)/len(values_per_depth))
    return(final_results_ros_mean, final_results_ros_stdev, final_results_lcb_mean, final_results_ros_stdev)
    

# path_final_plot, path_final_graph, path_ros_shots, path_ros_noiseless_res, path_ros_noisy_res, path_lcb_shots, path_lcb_noiseless_res, path_lcb_noisy_res
def save_figures(path_final_plot, path_final_graph, path_ros_shots, path_ros_noiseless_res, path_ros_noisy_res, path_lcb_shots, path_lcb_noiseless_res, path_lcb_noisy_res):
    with open(path_ros_shots) as tf:
        shots_ros = tf.read().split('\n\n')
        shots_ros_formatted = format_txt(shots_ros)
    with open(path_ros_noiseless_res) as tf:
        results_ros_noiseless = tf.read().split('\n\n')
        results_ros_noiseless_formatted = format_txt(results_ros_noiseless)
    with open(path_ros_noisy_res) as tf:
        results_ros_noisy = tf.read().split('\n\n')
        results_ros_noisy_formatted = format_txt(results_ros_noisy)
    with open(path_lcb_shots) as tf:
        shots_lcb = tf.read().split('\n\n')
        shots_lcb_formatted = format_txt(shots_lcb)
    with open(path_lcb_noiseless_res) as tf:
        results_lcb_noiseless = tf.read().split('\n\n')
        results_lcb_noiseless_formatted = format_txt(results_lcb_noiseless)
    with open(path_lcb_noisy_res) as tf:
        results_lcb_noisy = tf.read().split('\n\n')
        results_lcb_noisy_formatted = format_txt(results_lcb_noisy)
        
    # means_lcb_noiseless is a list with three elements for each depth. For each depth, there are 5 lists with the results
    means_lcb_noiseless, stdevs_lcb_noiseless = mean_and_stdev_lcb(results_lcb_noiseless_formatted)
    means_lcb_noisy, stdevs_lcb_noisy = mean_and_stdev_lcb(results_lcb_noisy_formatted)
    shots_lcb = shots_lcb_formatted[0][0]
    
    means_lcb_plus_stdev_noiseless = [[mean + stdev for (mean, stdev) in zip(means_per_depth, stdevs_per_depth)] for (means_per_depth, stdevs_per_depth) in zip(means_lcb_noiseless, stdevs_lcb_noiseless)]
    means_lcb_minus_stdev_noiseless = [[mean - stdev for (mean, stdev) in zip(means_per_depth, stdevs_per_depth)] for (means_per_depth, stdevs_per_depth) in zip(means_lcb_noiseless, stdevs_lcb_noiseless)]
    
    
    percentage = 0.1
    total_shots = 1000000
    means_ros_noiseless_depths = []
    means_ros_noisy_depths = []
    shots_ros_intervals_depths = []
    means_ros_plus_stdev_noiseless = []
    means_ros_minus_stdev_noiseless = []
    for i in range(len(shots_ros_formatted)):
        shots_ros_noiseless, means_ros_noiseless, stdev_ros_noiseless = calculate_mean_and_stdev_rosalin(shots_ros_formatted[i], results_ros_noiseless_formatted[i], percentage, total_shots)
        shots_ros_noisy, means_ros_noisy, stdev_ros_noisy = calculate_mean_and_stdev_rosalin(shots_ros_formatted[i], results_ros_noiseless_formatted[i], percentage, total_shots)
        # This works well suuu
        extra_shots, extra_results_noisy, extra_stdev_noisy = calculate_mean_stdev_extra(shots_ros_formatted[i], results_ros_noisy_formatted[i], 1000000)
        extra_shots, extra_results_noiseless, extra_stdev_noiseless = calculate_mean_stdev_extra(shots_ros_formatted[i], results_ros_noiseless_formatted[i], 1000000)
        
        shots_ros_noiseless.append(extra_shots)
        means_ros_noiseless.append(extra_results_noiseless)
        means_ros_noisy.append(extra_results_noisy)
        means_ros_noiseless = [mean - MIN_EIGENVALUE for mean in means_ros_noiseless]
        means_ros_noisy = [mean - MIN_EIGENVALUE for mean in means_ros_noisy]
        stdev_ros_noiseless.append(extra_stdev_noiseless)
        stdev_ros_noisy.append(extra_stdev_noisy)
        
        shots_ros_intervals_depths.append(shots_ros_noiseless)
        means_ros_noiseless_depths.append(means_ros_noiseless)
        means_ros_noisy_depths.append(means_ros_noisy)
        
        mean_minus_stdev_noisy = [mean - stdev for (mean, stdev) in zip(means_ros_noisy, stdev_ros_noisy)]
        mean_plus_stdev_noisy = [mean + stdev for (mean, stdev) in zip(means_ros_noisy, stdev_ros_noisy)]
        mean_minus_stdev_noiseless = [mean - stdev for (mean, stdev) in zip(means_ros_noiseless, stdev_ros_noiseless)]
        mean_plus_stdev_noiseless = [mean + stdev for (mean, stdev) in zip(means_ros_noiseless, stdev_ros_noiseless)]
        
        means_ros_plus_stdev_noiseless.append(mean_plus_stdev_noiseless)
        means_ros_minus_stdev_noiseless.append(mean_minus_stdev_noiseless)
        
    
    plt.plot(shots_ros_intervals_depths[0], means_ros_noiseless_depths[0], 'r', label = 'Ros-1')
    plt.plot(shots_ros_intervals_depths[0], means_ros_plus_stdev_noiseless[0], 'r', alpha=0.35)
    plt.plot(shots_ros_intervals_depths[0], means_ros_minus_stdev_noiseless[0], 'r', alpha=0.35)
    plt.fill_between(shots_ros_intervals_depths[0], means_ros_plus_stdev_noiseless[0], means_ros_minus_stdev_noiseless[0], facecolor='r', alpha = 0.25)
    
    plt.plot(shots_ros_intervals_depths[1], means_ros_noiseless_depths[1], 'b', label = 'Ros-3')
    plt.plot(shots_ros_intervals_depths[1], means_ros_plus_stdev_noiseless[1], 'b', alpha=0.35)
    plt.plot(shots_ros_intervals_depths[1], means_ros_minus_stdev_noiseless[1], 'b', alpha=0.35)
    plt.fill_between(shots_ros_intervals_depths[1], means_ros_plus_stdev_noiseless[1], means_ros_minus_stdev_noiseless[1], facecolor='b', alpha = 0.25)
    
    plt.plot(shots_ros_intervals_depths[2], means_ros_noiseless_depths[2], 'g', label = 'Ros-5')
    plt.plot(shots_ros_intervals_depths[2], means_ros_plus_stdev_noiseless[2], 'g', alpha=0.35)
    plt.plot(shots_ros_intervals_depths[2], means_ros_minus_stdev_noiseless[2], 'g', alpha=0.35)
    plt.fill_between(shots_ros_intervals_depths[2], means_ros_plus_stdev_noiseless[2], means_ros_minus_stdev_noiseless[2], facecolor='g', alpha = 0.25)
    
    plt.plot(shots_lcb, means_lcb_noiseless[0], 'c', label = 'LCB-1')
    plt.plot(shots_lcb, means_lcb_plus_stdev_noiseless[0], 'c', alpha=0.35)
    plt.plot(shots_lcb, means_lcb_minus_stdev_noiseless[0], 'c', alpha=0.35)
    plt.fill_between(shots_lcb, means_lcb_plus_stdev_noiseless[0], means_lcb_minus_stdev_noiseless[0], facecolor='c', alpha = 0.25)
    
    plt.plot(shots_lcb, means_lcb_noiseless[1], 'm', label = 'LCB-3')
    plt.plot(shots_lcb, means_lcb_plus_stdev_noiseless[1], 'm', alpha=0.35)
    plt.plot(shots_lcb, means_lcb_minus_stdev_noiseless[1], 'm', alpha=0.35)
    plt.fill_between(shots_lcb, means_lcb_plus_stdev_noiseless[1], means_lcb_minus_stdev_noiseless[1], facecolor='m', alpha = 0.25)
    
    
    plt.plot(shots_lcb, means_lcb_noiseless[2], 'y', label = 'LCB-5')
    plt.plot(shots_lcb, means_lcb_plus_stdev_noiseless[2], 'y', alpha=0.35)
    plt.plot(shots_lcb, means_lcb_minus_stdev_noiseless[2], 'y', alpha=0.35)
    plt.fill_between(shots_lcb, means_lcb_plus_stdev_noiseless[2], means_lcb_minus_stdev_noiseless[2], facecolor='m', alpha = 0.25)
    
    
    plt.xlim([0, 1000000])
    plt.ylim([0, 400])
    plt.legend(loc = "upper right", ncol = 3)
    plt.savefig(path_final_plot, dpi=500)
    #plt.show()
    
    
    results_ros_mean, results_ros_stdev, results_lcb_mean, results_lcb_stdev = get_correct_decisions()
    all_means = results_ros_mean + results_lcb_mean
    all_stdevs = results_ros_stdev + results_lcb_stdev
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylim([0, 12])
    categories = ['Ros-1', 'Ros-3', 'Ros-5', 'LCB-1', 'LCB-3', 'LCB-5']
    #colours = ['r', 'b', 'g', 'c', 'm', 'y']
    colours = ['#accbff', '#92bbff', '#78aaff', '#649eff', '#4188ff', '#4076D9']
    ax.bar(categories, all_means, yerr = all_stdevs, align = 'center', ecolor = 'black', capsize = 5, color = colours)
    #ax.errorbar(categories, all_means, all_stdevs, color='Black', alpha=0.5, capsize = 5)
    fig.savefig(path_final_graph, dpi=500)


    
def calculate_mean_stdev_extra(shots, results, total_shots):
    shots_over_the_limit = []
    results_over_the_limit = []
    for shots_per_run, results_per_run in zip(shots, results):
        if shots_per_run[-1] > total_shots:
            shots_over_the_limit.append(shots_per_run[-1])
            results_over_the_limit. append(results_per_run[-1])
    return (np.mean(shots_over_the_limit),
            np.mean(results_over_the_limit),
            np.std(results_over_the_limit))
    
    
    
    
        
def format_txt(txt_data):
    result = [txt_per_depth.split('\n') for txt_per_depth in txt_data]
    result = [[txt_single_run.split(' ') for txt_single_run in txt_per_depth if txt_single_run] for txt_per_depth in result]
    # Changing strings to floats
    try:
        result = [[[float(individual_string) for individual_string in txt_single_run] for txt_single_run in txt_per_depth] for txt_per_depth in result]
    except:
        print('result at this stage are', result)
    return result

def mean_and_stdev_lcb(results):
    final_results = []
    final_stdevs = []
    num_depths = len(results)
    num_runs = len(results[0])
    num_evals = len(results[0][0])
    # For each of the hamiltonians            
    for i in range(num_depths):
        results_mean = []
        results_stdev = []
        # Iterate over the function evaluations that are paired to the shots in the plot.
        for j in range(num_evals):
            values_for_mean = []
            values_for_stdev = []
            # Iterate along the 50 runs of LCB
            for k in range(num_runs):
                values_for_mean.append(results[i][k][j])
                values_for_stdev.append(results[i][k][j])
                # Here I am minusing the minimum eigenvalue to get an absolute metric
            results_mean.append(np.mean(values_for_mean) - MIN_EIGENVALUE)
            results_stdev.append(np.std(values_for_stdev))
        final_results.append(results_mean)
        final_stdevs.append(results_stdev)
    return final_results, final_stdevs
        
            
        
        
def calculate_mean_and_stdev_rosalin(shots_runs, results_runs, percentage, total_shots):
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
    
    #print('shots_runs are', shots_runs)
        
        
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
    #print('sorted_shots are', sorted_shots)
    
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
    #print('Shots per interval are', shots_per_interval)
    
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

        # Checking if there are some results in this interval
        if current_shots != min_shots_interval:
            # The smallest eigenvalue is minused to the mean to make the measure absolute
            mean = sum_res_interval/(cont-cont_start_interval)
            positions.append(int(shots_per_interval*(i+1/2)))
            stdev = np.std(stdev_list)
            
            means.append(mean)
            stdevs.append(stdev)

        current_shots = sorted_shots[cont]
        
    return (positions, means, stdevs) 
        




        


        
         
            
            
    
        
