# Advanced Quantum Circuit Examples
# Complex circuits demonstrating advanced techniques

# Circuit 1: Quantum Approximate Optimization Algorithm (QAOA)
# Solves combinatorial optimization problems

reset
resize 4

# QAOA for Max-Cut problem on a simple graph
# Graph: 0-1-2-3 (linear chain)

# Initialize in equal superposition
h 0
h 1
h 2
h 3

# Cost Hamiltonian evolution (γ parameter)
# Apply ZZ interactions for each edge
gate CRZ 0 1 1.0    # Edge (0,1)
gate CRZ 1 2 1.0    # Edge (1,2)
gate CRZ 2 3 1.0    # Edge (2,3)

# Mixer Hamiltonian evolution (β parameter)
rx 0.8 0
rx 0.8 1
rx 0.8 2
rx 0.8 3

# Second QAOA layer
gate CRZ 0 1 0.5
gate CRZ 1 2 0.5
gate CRZ 2 3 0.5

rx 0.4 0
rx 0.4 1
rx 0.4 2
rx 0.4 3

# Measure for optimization result
probabilities
measure_all

# Circuit 2: Quantum Machine Learning - Variational Classifier
# Parameterized circuit for classification

reset
resize 3

# Data encoding layer (encode classical data into quantum state)
ry 0.5 0     # Encode feature 1
ry 0.3 1     # Encode feature 2
ry 0.7 2     # Encode feature 3

# Variational layer 1
cnot 0 1
cnot 1 2
cnot 2 0

ry 0.2 0     # Trainable parameter
ry 0.4 1     # Trainable parameter
ry 0.6 2     # Trainable parameter

# Variational layer 2
cnot 0 1
cnot 1 2

ry 0.1 0     # Trainable parameter
ry 0.8 1     # Trainable parameter

# Measurement for classification
probabilities

# Circuit 3: Quantum Simulation of Molecular Hamiltonian
# Simulates H2 molecule using VQE

reset
resize 4

# Jordan-Wigner transformation for fermionic Hamiltonian
# This represents a simplified H2 molecule

# Initialize trial state
ry 0.5 0
ry 0.5 1

# Trotterized time evolution
# XX interaction
h 0
h 1
cnot 0 1
rz 0.1 1
cnot 0 1
h 0
h 1

# YY interaction
rx -1.5708 0  # RX(-π/2) = -iY
rx -1.5708 1
cnot 0 1
rz 0.1 1
cnot 0 1
rx 1.5708 0   # RX(π/2) = iY
rx 1.5708 1

# ZZ interaction
cnot 0 1
rz 0.1 1
cnot 0 1

# Show molecular state
state
probabilities

# Circuit 4: Quantum Error Correction - Surface Code
# Simplified surface code for error correction

reset
resize 9

# Initialize logical |+⟩ state in surface code
# Data qubits: 0,2,4,6,8 (odd positions)
# Measure qubits: 1,3,5,7 (even positions)

h 0
h 2
h 4
h 6
h 8

# Syndrome measurements (simplified)
cnot 0 1
cnot 2 1
cnot 2 3
cnot 4 3
cnot 4 5
cnot 6 5
cnot 6 7
cnot 8 7

# Measure syndrome qubits
measure 1
measure 3
measure 5
measure 7

# Based on syndrome, apply corrections (simplified)
# In real implementation, this would be conditional

# Circuit 5: Quantum Fourier Transform for Period Finding
# Core component of Shor's algorithm

reset
resize 8

# Initialize register in superposition
h 0
h 1
h 2
h 3

# Apply function f(x) = a^x mod N (simplified)
# This would be the modular exponentiation oracle
cnot 0 4
cnot 1 5
cnot 2 6
cnot 3 7

# Inverse QFT on first register
# Full implementation for 4 qubits
swap 0 3
swap 1 2

gate CRZ 0 1 -0.7854  # -π/4
gate CRZ 0 2 -0.3927  # -π/8
gate CRZ 0 3 -0.1963  # -π/16
h 0

gate CRZ 1 2 -0.7854  # -π/4
gate CRZ 1 3 -0.3927  # -π/8
h 1

gate CRZ 2 3 -0.7854  # -π/4
h 2

h 3

# Measure to find period
measure 0
measure 1
measure 2
measure 3

# Circuit 6: Quantum Walks on Graphs
# Quantum walk on a cycle graph

reset
resize 6

# Position qubits (3 qubits for 8 positions)
# Coin qubit for direction

# Initialize at position 0, coin in superposition
h 5              # Coin qubit

# Quantum walk steps
# Step 1
cnot 5 0         # Conditional move based on coin
cnot 5 1
cnot 5 2

# Coin flip
h 5

# Step 2
cnot 5 0
cnot 5 1
cnot 5 2

h 5

# Show final position distribution
probabilities
