#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:30:35 2022

@author: user
"""

import cirq
import openfermion
import numpy as np

from scipy.stats import multinomial

from pathlib import Path
# sys.path stores all the locations where python searches for packages
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from optimisers.rosalin import Rosalin
from optimisers.lcb_cmaes import LCB_CMAES
from optimisers.helpers import exact_expectation_circuit, sampled_expectation_circuit, noisy_expect_val, generate_circuit_from_pauli_string, _generate_one_pauli, _generate_two_pauli
from circuit.create_circuit import create_circuit



# Stop this for now
# optimizer.run(kcma=kcma)
#save_dir = (os.getcwd() + '/../../data/experiment_sampling/H4line/uccsd/oneplusonetocma/dataset08_12/')

# Esto no esta en el codigo que yo tengo ahora
# _hist_shots_per_iteration = optimizer._hist_shots_per_iteration

# Start callback function and variables
# Estas tampoco las uso
hist_params = []
hist_expectations = []

def run_lcbcmaes(n_qubits, hamiltonian, total_shots, seed, depth=1):
    '''Run LCB-CMAES for a given hamiltonian.

    Parameters
    ----------
    n_qubits : int
        The number of qubits that will be used in the quantum circuit
    hamiltonian : QubitOperator
        The hamiltonian that is being minimised by the LCB-CMAES.
    total_shots : int
        Total number of shots used in the optimisation run
    seed : int
        Seed that is used to generate the random numbers. This is fixed to be able to replicate the results
    depth : int
        Depth of the circuit. The main building block of the circuit is a set of parametrised gates that 
        act on all the qubits. This building block can be replicated and put one after the other. The 
        parameter 'depth' indicates how many of these building blocks for the whole circuit. 
    
    
    Returns
    -------
    optimizer.used_shots_per_iteration : List[int] # TODO: check this one too
        List containing the shots that are used for each of the iterations.
    optimizer.best_expectations_noisy : List[float]
        List containing the noisy expectations for all the iterations.
    optimizer.best_expectations_noiseless : List[float]
        List containing the noiseless expectations for all the iterations.
    hist_params : List[List[float]] 
        List containing the the parameters  for all the iterations. Given one iteration, the parameters 
        used in that iteration are stored as a List[float]
    hist_expectations : List[]  # TODO: I don't know what this does exactly
    '''
    
    # Setting the parameters
    # TODO: put this well, the correct value. I put 5 as dummy data
    # kcma = 10
    _mu = 5
    popsize = 10
    sigma0 = 0.15
    min_shots = 100
    delta_shots = 200
    it_shots = 3 * 5 * delta_shots + min_shots * popsize
    # total_shots = maxfev * it_shots
    # Quito esto para que sea por num shots y no por num evaluaciones
    # Lo necesito para el options que va como input al LCB
    maxfev = 10
    
    simulator = cirq.Simulator()
    
    rng = np.random.RandomState(seed=seed)
    
    # Creating the circuit
    # I am adding depth of the circuit here too
    circuit, parameters_dict = create_circuit(n_qubits, depth)
    
    # Creating the different components of the Hamiltonian
    hamiltonian_matrix = openfermion.get_sparse_operator(
        hamiltonian)
    ham_matrices = []
    ham_terms = []
    # Get the Pauli operators in coefficient and matrix form.
    for matrix, coeff in hamiltonian.terms.items():
        ham_terms.append(coeff)
        ham_matrices.append(
            openfermion.get_sparse_operator(
                openfermion.QubitOperator(matrix, 1.0),
                n_qubits=n_qubits
            )
        )
        
    options = {
        'bounds': [-np.pi, np.pi],
        'seed' : seed,
        'maxfevals' : maxfev,
        'verbose' : -1,
        'verb_disp' : 1,
        'verb_log' : 1,
        'CMA_diagonal' : False
    }
    
    # fun_args for a certain problem
    fun_args = {
        'circuit' : circuit,
        'parameters_dict' : parameters_dict,
        'hamiltonian' : hamiltonian,
        'hamiltonian_terms' : ham_terms,
        'hamiltonian_matrices' : ham_matrices,
        'simulator' : simulator,
        'rng' : rng,
        'thld' : 1e-3,
        'sampled' : True,
        'hamiltonian_matrix' : hamiltonian_matrix
    }
    
    # Ready to start the algorithm
    optimizer = LCB_CMAES(
        fun=sampled_expectation_circuit,
        x0=(np.zeros(len(parameters_dict))),
        sigma0=sigma0,
        min_nb_shots_candidate=min_shots,
        increment_nb_shots=delta_shots,
        total_nb_shots_iteration=it_shots,
        total_nb_shots=total_shots,
        CMA_options=options,
        fun_args=fun_args,
        fun_noiseless=exact_expectation_circuit,
        fraction_var_reduction=0.1
    )
    # TODO: I am trying there without kcma, this was before optimizer.run(kcma=kcma, var_reduction=True)
    optimizer.run(var_reduction=True)
    
    #return(optimizer.used_shots_per_iteration, optimizer.best_expectations_noisy, optimizer.best_expectations_noiseless, optimizer.final_x)
    return(optimizer.used_shots_per_iteration, optimizer.best_expectations_noisy, optimizer.best_expectations_noiseless, optimizer._best.x)

    
    

def run_rosalin(n_qubits, hamiltonian, total_shots, min_shots, shots_iteration_meas, seed, depth=1):    
    '''Run Rosalin for a given hamiltonian.

    Parameters
    ----------
    n_qubits : int
        The number of qubits that will be used in the quantum circuit
    hamiltonian : QubitOperator
        The hamiltonian that is being minimised by the Rosalin.
    total_shots : int
        Total number of shots used in the optimisation run
    min_shots : int
        Minimum number of shots used to estimate a component of the gradient on a certain iteration.
    shots_iteration_meas : int
        Number of shots to estimate the cost function once a new set of parameters is obtained at the end of
        an iteration.
    seed : int
        Seed that is used to generate the random numbers. This is fixed to be able to replicate the results
    depth : int
        Depth of the circuit. The main building block of the circuit is a set of parametrised gates that 
        act on all the qubits. This building block can be replicated and put one after the other. The 
        parameter 'depth' indicates how many of these building blocks for the whole circuit. 
    
    
    Returns
    -------
    shots_rosalin : List[int] # TODO: check this one too
        List containing the shots that are used for each of the iterations.
    cost_rosalin_noisy : List[float]
        List containing the noisy expectations for all the iterations.
    cost_rosalin_noiseless : List[float]
        List containing the noiseless expectations for all the iterations.
    opt.history_params : List[List[float]] 
        List containing the the parameters  for all the iterations. Given one iteration, the parameters 
        used in that iteration are stored as a List[float]
    '''
    
    
    # Number of parameters for the circuit showed in the iCANS paper
    number_parameters = (n_qubits - 1)*9*depth + n_qubits
    
    np.random.RandomState(seed=seed)
    init_params = np.random.uniform(-np.pi, np.pi, number_parameters)
    # init_params = np.array([random.uniform(0, 2*np.pi) for _ in range(number_parameters)])

    # Call the optimizer class
    opt = Rosalin(n_qubits, init_params, hamiltonian, min_shots, depth)
    
    # To do the estimation of the Hamiltonian with the newly found theta and see how the result of the 
    # algorithm is evolving
    cost_rosalin_noisy = []
    cost_rosalin_noiseless = []
    shots_rosalin = []

    params = init_params

    cont = 1

    while (opt.shots_used < total_shots):
        # New theta after a new iteration
        params = opt.step(params)

        # We calculate how many shots we have done
        shots_rosalin.append(opt.shots_used)

        # Calculate the estimation for the noisy cost with the new params of this iteration
        si = multinomial(n=shots_iteration_meas,p=opt.probabilities)
        shots_per_term = si.rvs()[0]
        estimations = opt.estimate_hamiltonian(params, shots_per_term)
        opt.shots_used += shots_iteration_meas
        
        average = sum(estimations)/len(estimations)        
        cost_rosalin_noisy.append(average)
        
        exact_cost = opt.hamiltonian_exact_result(params)
        cost_rosalin_noiseless.append(exact_cost)
        
        print(f"Step {cont}: analytic cost = {cost_rosalin_noiseless[-1]}, noisy cost = {cost_rosalin_noisy[-1]}, shots_used = {shots_rosalin[-1]}")

        cont += 1
        
    return (shots_rosalin, cost_rosalin_noisy, cost_rosalin_noiseless, opt.history_params)

