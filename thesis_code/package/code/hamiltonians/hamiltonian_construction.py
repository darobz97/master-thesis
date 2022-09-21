# -*- coding: utf-8 -*-


import numpy as np
from openfermion import QubitOperator

def random_hamiltonian(n_qubits, n_paulis, max_locality, bounds, rng):
    '''
    Generate a random instance of a Hamiltonian.
    
    Args:
        n_qubits (int): Number of qubits.
        n_paulis (int): Number of Pauli operators of the Hamiltonian.
        max_locality (int): Maximum size of the Pauli operators.
        bounds: List of floats from which the coefficients will be sampled.
        rng: RandomState
    '''
    hamiltonian = QubitOperator() # from OpenFermion
    for _ in range(n_paulis):
        qubits = np.sort(rng.choice(n_qubits, rng.randint(0, max_locality+1), replace=False)) # From 0 to n_qubits, randint is 
        # the length of the vector of qubits
        # This makes a random array of paulis (X, Y, Z) of the length of qubits
        paulis = np.array(['X', 'Y', 'Z'])[rng.randint(0,3, size=len(qubits))]
        op_string = str()
        # Zip the qubits with the paulis, so one gate for each qubit
        for p, q in zip(paulis, qubits):
            op_string+=' '+p+str(q)
        # These gates perform on the qubits
        hamiltonian += rng.uniform(bounds[0], bounds[1])*QubitOperator(op_string)
        # TODO: see if they act randomly on only some qubits for each iteration
    return hamiltonian

def heisenberg_hamiltonian(n_qubits, jx, jy, jz, periodic=True):
    '''
    Generate an instance of the Heisenberg Hamiltonian.
    
    If jx = jy = jz, then XXX Hamiltonian.
    If jx = jy != jz, then XXZ Hamiltonian.
    if jx != jy != jz then XYZ Hamiltonian.
    
    Args:
        n_qubits (int): Number of qubits.
        jx (float): Strength in X-direction.
        jy (float): Strength in Y-direction
        jz (float): Strength in Z-direction
        periodic (bool): Periodic boundary conditions or not.
    '''
    hamiltonian = QubitOperator()
    if periodic:
        for i in range(n_qubits):
            hamiltonian += (
                QubitOperator(f'X{i} X{(i+1)%n_qubits}', jx) +
                QubitOperator(f'Y{i} Y{(i+1)%n_qubits}', jy) +
                QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz))

    else:
        for i in range(n_qubits-1):
            hamiltonian += (
                QubitOperator(f'X{i} X{(i+1)%n_qubits}', jx) +
                QubitOperator(f'Y{i} Y{(i+1)%n_qubits}', jy) +
                QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz))
    return hamiltonian

def ising_hamiltonian(n_qubits, jx, jz, periodic=True):
    '''
    Generate an instance of the Ising Hamiltonian.
    
    Args:
        n_qubits (int): Number of qubits.
        jx (float): Strength in X-direction.
        jz (float): Strength in Z-direction
        periodic (bool): Periodic boundary conditions or not.
    '''
    hamiltonian = QubitOperator()
    if periodic:
        for i in range(n_qubits):
            hamiltonian += (
                QubitOperator(f'X{i}', jx) +
                QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz))

    else:
        for i in range(n_qubits-1):
            hamiltonian += (QubitOperator(f'X{i}', jx)+
                            QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz))

        hamiltonian += QubitOperator(f'X{i+1}', jx)
    
    return hamiltonian

def ghz_hamiltonian(n_qubits, jz, periodic=True):
    '''
    Generate GHZ Hamiltonian.
    
    Args:
        n_qubits(int): Number of qubits.
        jz (float): Strength of the coefficient.
        periodic (bool): If systems is periodic.
    '''
    hamiltonian = QubitOperator()
    
    if periodic:
        for i in range(n_qubits):
            hamiltonian += QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz)

    else:
        for i in range(n_qubits-1):
            hamiltonian += QubitOperator(f'Z{i} Z{(i+1)%n_qubits}', jz)
    return hamiltonian