#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 06:32:32 2022

@author: user
"""

import numpy as np
import cirq
import openfermion as of

from sympy import Symbol


hist_params = []
hist_expectations = []


def noisy_expect_val(expectation_value: float, nshots: int,
                     rng: np.random.RandomState,
                     thld: float = 1e-6,
                     **kwargs) -> float:
    """
    Compute noisy expectation value.
    Samples the expectation value from a multinomial distribution
    with a given number of repetitions.
    Args:
        expectation_value: noiseless expectation value.
        nshots: number of shots to sample.
        thld: tolerance threshold to assign -1 or 1.
    Returns:
        sampled expectation value.
    """
    expectation_value = expectation_value.real
    if expectation_value < (-1.0 - thld) or expectation_value > (1.0 + thld):
        raise ValueError('Expectation value can not be larger than +1 or -1.')
    if nshots <= 0:
        raise ValueError('Number of shots must be larger than 0.')

    bernoulli_rv = (1 - expectation_value) / 2

    if -thld < bernoulli_rv <= 0.0:
        noisy_prob = 0.0

    elif 1 <= bernoulli_rv < (1 + thld):
        noisy_prob = 1.0

    else:
        noisy_prob = rng.binomial(nshots, bernoulli_rv) / nshots

    noisy_expectation_value = (1 - 2 * noisy_prob)

    return noisy_expectation_value


def exact_expectation_circuit(parameters,
                              circuit,
                              parameters_dict,
                              hamiltonian,
                              simulator,
                              **kwargs) -> float:
    """
    Compute energy expectation value without sampling.

    Args:
        circuit: Circuit object to prepare state.
        parameters: parameterst to update the circuit.
        parameters_dict: Dictionary of parameters.
        hamiltonian: OpenFermion object that can be converted to
        a sparse matrix.

    Returns:
        expectation value of Hamiltonian without sampling.
    """
    if len(parameters) != len(parameters_dict):
        raise ValueError(
            'Number of parameters and dictionary do not match.')
    #if not isinstance(hamiltonian, openfermion.QubitOperator):
    #    raise TypeError('Hamiltonian must be a QubitOperator.')

    global hist_params
    global hist_expectations

    for val, key in zip(parameters, parameters_dict.keys()):
        parameters_dict[key] = val

    simulated_circuit = simulator.simulate(
        cirq.resolve_parameters(circuit,
                                cirq.ParamResolver(parameters_dict)))

    expectation_value = np.real(of.expectation(
        of.get_sparse_operator(hamiltonian),
        simulated_circuit.final_state_vector)
    )

    hist_params.append(parameters)
    hist_expectations.append(expectation_value)
    return expectation_value

def sampled_expectation_circuit(
                            parameters,
                            circuit,
                            parameters_dict,
                            hamiltonian_terms,
                            hamiltonian_matrices,
                            simulator,
                            rng,
                            nshots,
                            thld=1e-6,
                            sampled=False,
                            hamiltonian_matrix=None,
                            **kwargs):
    """
    Compute energy by sampling Pauli operators.

    Args:
        circuit: Circuit object to prepare state.
        parameters: parameterst to update the circuit.
        parameters_dict: Dictionary of parameters.
        hamiltonian_terms: List of coefficients of Hamiltonian.
        hamiltonian_matrices: List of sparse matrices defining
            the Hamiltonian.
        nshots: number of shots to sample.
        thld: threshold for error in sampled probability.

    Returns:
        expectation value of Hamiltonina sampled by the number of shots.
    """
    if len(parameters) != len(parameters_dict):
        raise ValueError(
            'Number of parameters and dictionary do not match.')
    if len(hamiltonian_terms) != len(hamiltonian_matrices):
        raise ValueError(
            'Number of coefficients and matrices do not match.')
    if nshots <= 0:
        raise ValueError('Number of shots must be larger than 0.')
    for val, key in zip(parameters, parameters_dict.keys()):
        parameters_dict[key] = val

    global hist_params
    global hist_expectations

    # The type of this simulated_circuit is StateVectorTrialResult
    # IMPORTANT: for 4 qubits, the length of the final state vector is 16, which makes sense
    simulated_circuit = simulator.simulate(
                    cirq.resolve_parameters(circuit,
                            cirq.ParamResolver(parameters_dict)))
    
    
    if sampled:
        sampled_res = 0
        # Change simulated_circuit.final_state for simulated_circuit.final_state_vector
        for coeff, mat in zip(hamiltonian_terms, hamiltonian_matrices):
            result = of.expectation(
                            mat,
                            simulated_circuit.final_state_vector)  # type: ignore
            sampled_res += coeff * noisy_expect_val(
                            expectation_value=result,
                            nshots=nshots,
                            rng=rng,
                            thld=thld)

        expectation_value = np.real(sampled_res)
    else:
        expectation_value = np.real(
            of.expectation(
                hamiltonian_matrix, simulated_circuit.final_state)
            )
    hist_params.append(parameters)
    hist_expectations.append(expectation_value)

    return expectation_value




def _generate_two_pauli(n_qubits, pauli):
    '''
    Create sum_i,j Sigma_iSigma_{i+1} operator.
    
    Args:
        n_qubits (int): Number of qubits.
        pauli (str): X, Y or Z represeting a Pauli operator. 
    ''' 
    two_pauli = of.QubitOperator()
    for i in range(n_qubits):
        two_pauli += of.QubitOperator(
            f'{pauli}{i} {pauli}{(i+1)%n_qubits}', 1.0) 
    return two_pauli




def _generate_one_pauli(n_qubits, pauli):
    '''
    Create sum_i Sigma_i operators.
    
    Args:
        n_qubits (int): Number of qubits.
        pauli (str): X, Y or Z represeting a Pauli operator. 
    '''
    one_pauli = of.QubitOperator()
    for i in range(n_qubits):
        one_pauli += of.QubitOperator(f'{pauli}{i}', 1.0)
    return one_pauli




def _qubit_operator_term_to_pauli_string(term,qubits):
    """
    Convert term of QubitOperator to a PauliString.
    Args:
        term (dict): QubitOperator term.
        qubits (list): List of qubit names.
    Returns:
        pauli_string (PauliString): cirq PauliString object.
    """
    ind_ops, coeff = term

    return cirq.PauliString(dict((qubits[ind], op) for ind, op in ind_ops),
                            coeff)



def qubit_operator_to_pauli_sum(operator, qubits=None):
    """
    Convert QubitOperator to a sum of PauliString.
    Args:
        operator (QubitOperator): operator to convert.
        qubits (List): Optional list of qubit names.
            If `None` a list of `cirq.LineQubit` of length number of qubits
            in operator is created.
    Returns:
        pauli_sum (PauliSum): cirq PauliSum object.
    Raises:
        TypeError: if qubit_op is not a QubitOpertor.
    """
    if not isinstance(operator, of.QubitOperator):
        raise TypeError('Input must be a QubitOperator.')

    if qubits is None:
        qubits = cirq.LineQubit.range(of.count_qubits(operator))

    pauli_sum = cirq.PauliSum()
    for pauli in operator.terms.items():
        pauli_sum += _qubit_operator_term_to_pauli_string(pauli, qubits)

    return pauli_sum

def generate_circuit_from_pauli_string(
        operator, parameter_name,
        transformation=of.jordan_wigner):
    """
    Create a cirq.Circuit object from the operator.
    This function uses PauliString and PauliStringPhasor objects
    to generate a cirq.Circuit from the operator.
    Makes a circuit with a parametrized gate named after the input
    parameter_name.
    Args:
        operator (QubitOperator): Operator to translate
            to a circuit.
        parameter_name (str): Name to use for the sympy parameter.
        transformation: Optional fermion to qubit transformation.
            It uses Jordan-Wigner by default.
    Yields:
        cirq.Circuit objects from PauliStringPhasor.
    Raises:
        TypeError: if operator is not a FermionOperator.
    Notes:
        If this function is used to generate a concatenation of circuits
        be sure that the parameter name is the same or different.
    """
    if not isinstance(operator, of.QubitOperator):
        raise TypeError('Operator must be a QubitOperator object.')
    if not isinstance(parameter_name, str):
        raise TypeError('Parameter name must be a string.')

    for op, val in operator.terms.items():
        pauli_sum = qubit_operator_to_pauli_sum(
            of.QubitOperator(op, numpy.sign(val)))
        # See if I can do this better
        pauli_string = None
        for item in iter(pauli_sum):
            pauli_string = item
            
        
        yield cirq.Circuit(
                cirq.PauliStringPhasor(pauli_string,
                                       exponent_neg=-1.0*Symbol(
                                           parameter_name)))