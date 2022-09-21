# -*- coding: utf-8 -*-
"""
Rosalin with cirq
"""

import cirq
import openfermion as of
import numpy as np

from scipy.stats import multinomial

from pathlib import Path
import sys
# This path is the code folder where the optimisers are
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from circuit.create_circuit import create_circuit


class Rosalin:

    def __init__(self, num_qubits, params, qubit_operator, min_shots, depth, mu=0.99, b=1e-6, lr=0.07):
        self.params = params
        
        self.qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        
        pauli_sum = None
        if isinstance(qubit_operator, cirq.PauliSum):
            pauli_sum = qubit_operator
        elif isinstance(qubit_operator, of.QubitOperator):
            pauli_sum = of.transforms.qubit_operator_to_pauli_sum(
                qubit_operator,
                self.qubits
            ) 
        
        self.hamiltonian_terms = []
        
        coeffs = []

        for term in pauli_sum:
            # This takes the number to calculate the probabilities
            string_coeff = str(term).split("*")[0]
            real_coeff = None
            if string_coeff[0] in ['X', 'Y', 'Z']:
                # The coeff is 1 so it does not appear in the QubitOperator
                real_coeff = 1
            # I just added this elif
            elif string_coeff[:2] in ['-X', '-Y', '-Z']:
                real_coeff = -1
                term = term/real_coeff
            else:
                # The coeff is a complex number different from 1
                real_coeff = complex(string_coeff).real
                term = term/real_coeff
                
            self.hamiltonian_terms.append(term)
            coeffs.append(real_coeff)
        self.coeffs = coeffs
        self.lipschitz = np.sum(np.abs(coeffs))
            
        self.probabilities = [np.abs(coeff)/sum(np.abs(coeffs)) for coeff in coeffs]

        if lr > 2 / self.lipschitz:
            lr = 2 / (self.lipschitz + 1)

        # hyperparameters
        self.min_shots = min_shots
        self.mu = mu  # running average constant
        self.b = b    # regularization bias
        self.lr = lr  # learning rate

        # keep track of the total number of shots used
        self.shots_used = 0
        # total number of iterations
        self.k = 0
        
        # Number of shots per parameter
        self.s = np.zeros(len(params), dtype=np.float64) + min_shots

        # Running average of the parameter gradients
        self.chi = None
        # Running average of the variance of the parameter gradients
        self.xi = None
        
        self.circuit, self.parameters_dict = create_circuit(num_qubits, depth)
        self.history_params = []

    # This is then used to calculate H+ and H-.
    def estimate_hamiltonian(self, params, shots_per_term):
        """Returns an array containing length ``shots`` single-shot estimates
        of the Hamiltonian. The shots are distributed randomly over
        the terms in the Hamiltonian, as per a Multinomial distribution.

        Since we are performing single-shot estimates, the QNodes must be
        set to 'sample' mode.
        """        
        # circuit = create_circuit(self.qubits, params)
        # I am going to change this here and put the circuit with params instead.        
        simulator = cirq.Simulator()
        
        # Maybe this is not very elegant, but I am doing this in line (so not various solutions in parallel like in the LCB)
        for val, key in zip(params, self.parameters_dict.keys()):
            self.parameters_dict[key] = val
        
        # self.circuit continues to be the circuit with unresolved params, and we are resolving them for the circuit variable here
        circuit = cirq.resolve_parameters(self.circuit,
                cirq.ParamResolver(self.parameters_dict))
                
        simulator = cirq.Simulator(dtype=np.complex128)
        ev_list = simulator.simulate_expectation_values(
            circuit,
            observables=self.hamiltonian_terms,
        )
        
        ev_list = [x.real for x in ev_list]
        # These two lines are to avoind eigenvalues bigger than 1 or smaller than -1
        ev_list = [1 if x > 1 else x for x in ev_list]
        ev_list = [-1 if x < -1 else x for x in ev_list]
                
        # Here we transform the exact expectation value of the h_i's into a number between [0,1] so we can 
        # sample from the binomial
        p_list = [(element+1)/2 for element in ev_list]
        
        expectation_results = []
        
        for i in range(len(self.probabilities)):
            results_binomial = np.random.binomial(1, p_list[i], size=shots_per_term[i])
            # Transform de results of the binomial from {0,1} to {-1,1} and then do the formula 
            # c_i*expectation/p_i
            expectations = [self.coeffs[i]*(result*2-1)/self.probabilities[i] for result in results_binomial]
            expectation_results += expectations
        
        expectation_results = np.array(expectation_results)
                
        return expectation_results
    
    def hamiltonian_exact_result(self, params):
        
        # Maybe this is not very elegant, but I am doing this in line (so not various solutions in parallel like in the LCB)
        for val, key in zip(params, self.parameters_dict.keys()):
            self.parameters_dict[key] = val
        
        # self.circuit continues to be the circuit with unresolved params, and we are resolving them for the circuit varaible here
        circuit = cirq.resolve_parameters(self.circuit,
                cirq.ParamResolver(self.parameters_dict))
        
        simulator = cirq.Simulator(dtype=np.complex128)
        ev_list = simulator.simulate_expectation_values(
            circuit,
            observables=self.hamiltonian_terms,
        )
                
        exact_result = sum([self.coeffs[i]*ev_list[i] for i in range(len(self.probabilities))])
        return exact_result.real
        
    def evaluate_grad_var(self, i, params, shots):
        """
        Evaluate the gradient, as well as the variance in the gradient,
        for the ith parameter in params, using the parameter-shift rule.
        """
        shift = np.zeros(len(params))
        shift[i] = np.pi / 2
                
        si = multinomial(n=shots, p=self.probabilities)
        
        shots_per_term = si.rvs()[0]

        shift_forward = self.estimate_hamiltonian(params + shift, shots_per_term)
        shift_backward = self.estimate_hamiltonian(params - shift, shots_per_term)
        
        g = np.mean(shift_forward - shift_backward) / 2
        # This is how they calculate the variance, from an array of results from running the circuit
        s = np.var((shift_forward - shift_backward) / 2, ddof=1)

        return g, s

    def step(self, params):
        """Perform a single step of the Rosalin optimizer."""
        # keep track of the number of shots run
        self.shots_used += int(2 * np.sum(self.s))

        # compute the gradient, as well as the variance in the gradient, using the number of shots determined by the array s.
        grad = []
        S = []
        
        for i in range(len(params)):
            # loop through each parameter, performing the parameter-shift rule
            g_, s_ = self.evaluate_grad_var(i, params, self.s[i])
            grad.append(g_)
            S.append(s_)
            
        grad = np.array(grad)
        S = np.array(S)

        # gradient descent update
        params = params - self.lr * grad
        self.history_params.append(params)
        
        if self.xi is None:
            self.chi = np.zeros(len(params), dtype=np.float64)
            self.xi = np.zeros(len(params), dtype=np.float64)

        # running average of the gradient variance
        self.xi = self.mu * self.xi + (1 - self.mu) * S
        xi = self.xi / (1 - self.mu ** (self.k + 1))

        # running average of the gradient
        self.chi = self.mu * self.chi + (1 - self.mu) * grad
        chi = self.chi / (1 - self.mu ** (self.k + 1))

        # determine the new optimum shots distribution for the next iteration of the optimizer
        op_for_ceiling = (2 * self.lipschitz * self.lr * xi) / ((2 - self.lipschitz * self.lr) * (chi ** 2 + self.b * (self.mu ** self.k)))
        
        # New number of shots for each part of the observable
        s = np.ceil(op_for_ceiling)
        # print("s without clipping is:", s)

        # apply an upper and lower bound on the new shot distributions, to avoid the number of shots reducing 
        # below min(2, min_shots), or growing too significantly.
        gamma = (
            (self.lr - self.lipschitz * self.lr ** 2 / 2) * chi ** 2
            - xi * self.lipschitz * self.lr ** 2 / (2 * s)
        ) / s

        
        smax = s[np.argmax(gamma)]
        if smax < 2:
            smax = 2
            
        # .astype changes it to int. Changing here min to max
        self.s = np.clip(s, min(2, self.min_shots), smax).astype(int)

        self.k += 1
        return params

