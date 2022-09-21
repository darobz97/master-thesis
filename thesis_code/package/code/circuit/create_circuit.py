#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper function used to create the circuit whose parameters we are trying to optimize
"""

import numpy
import cirq
import openfermion as of

from sympy import Symbol



def create_circuit(num_qubits, depth):
    """
    Function to create the parametrised circuit to be used in the variational quantum
    algorithms.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the circuit. This refers to the number of building blocks that conform 
        the circuit. More information on this in the iCANS paper.

    Returns
    -------
    circuit : cirq.Circuit
        Parametrised circuit without values. Values can be added later when the solutions
        are generated.
    param_dict : Dict
        Dictionary containing the parameters that form the circuit. At the start the 
        parameters are Symbol objects without a definite value. Later, when a candidate
        is obtained, the Symbols are changed to the actual values.

    """
    circuit = cirq.Circuit()
    
    # This is what was written before, but it may be better to change it to go from num_qubits to qubits
    # num_qubits = len(qubits)
    
    qubits = cirq.LineQubit.range(num_qubits)

    # This is used to find the location of the unitaries based on the iCANS paper.
    location_unitaries = []
    location_unitaries += [2*i for i in range(num_qubits//2)]
    location_unitaries += [1+2*i for i in range((num_qubits-1)//2)]
    
    num_unitaries = len(location_unitaries)
    
    cont = 1
    
    
    for i in range(depth):
        # Creation of unitaries
        for j in range(num_unitaries):
            location_unitary = location_unitaries[j]
            first_qubit = qubits[location_unitary]
            second_qubit = qubits[location_unitary + 1]
                
            rot_1 = cirq.ry(Symbol(f'theta_{cont}'))
            rot_2 = cirq.ry(Symbol(f'theta_{cont+1}'))
            circuit.append(rot_1(first_qubit))
            circuit.append(rot_2(second_qubit))
            
            rot_1 = cirq.rz(Symbol(f'theta_{cont+2}'))
            rot_2 = cirq.rz(Symbol(f'theta_{cont+3}'))
            circuit.append(rot_1(first_qubit))
            circuit.append(rot_2(second_qubit))
            
            circuit.append(cirq.CNOT(second_qubit, first_qubit))
            
            rot_1 = cirq.rz(Symbol(f'theta_{cont+4}'))
            rot_2 = cirq.ry(Symbol(f'theta_{cont+5}'))
            circuit.append(rot_1(first_qubit))
            circuit.append(rot_2(second_qubit))
            
            circuit.append(cirq.CNOT(first_qubit, second_qubit))
            
            rot_2 = cirq.ry(Symbol(f'theta_{cont+6}'))
            circuit.append(rot_2(second_qubit))
            
            circuit.append(cirq.CNOT(second_qubit, first_qubit))
            
            rot_1 = cirq.rz(Symbol(f'theta_{cont+7}'))
            rot_2 = cirq.rz(Symbol(f'theta_{cont+8}'))
            circuit.append(rot_1(first_qubit))
            circuit.append(rot_2(second_qubit))
            
            cont += 9
        
        
    # Before it was cont here -- this is wrong, it should be i
    # Last Ry rotations
    for i in range(num_qubits):
        rot = cirq.ry(Symbol(f'theta_{cont}'))
        circuit.append(rot(qubits[i]))
        cont += 1
   
    cont -= 1 # Restore the last value because at the end of the last iteration of the loop another 2 are added.
    
    param_dict = {}
    for i in range(cont+1):
        param_dict[Symbol(f'theta_{i}')] = 1.0

    return(circuit, param_dict)
    

