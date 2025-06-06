# Comprehensive Quantum Algorithms Collection
# This file demonstrates multiple quantum algorithms

# Algorithm 1: Deutsch-Jozsa Algorithm (Complete)
# Determines if a function is constant or balanced with just one query

reset
resize 4

# Initialize: 3 input qubits + 1 ancilla
x 3              # Ancilla in |1⟩
h 0              # Input qubits in superposition
h 1
h 2
h 3              # Ancilla in |−⟩ state

# Oracle for balanced function f(x) = x_0 ⊕ x_1 (XOR of first two bits)
# This flips ancilla if f(x) = 1
cnot 0 3         # Flip if x_0 = 1
cnot 1 3         # Flip if x_1 = 1

# Final Hadamards on input qubits
h 0
h 1
h 2

# Measure input qubits
# For balanced function: won't measure |000⟩
# For constant function: will measure |000⟩
measure 0
measure 1
measure 2

# Algorithm 2: Bernstein-Vazirani Algorithm
# Finds the hidden bit string in one query

reset
resize 4

# Hidden string is s = "101" (we'll implement f(x) = x·s)
x 3              # Ancilla in |1⟩
h 0              # Input qubits in superposition
h 1
h 2
h 3              # Ancilla in |−⟩

# Oracle: flip ancilla if x·s = 1 (where s = "101")
cnot 0 3         # x_0 · s_0 = x_0 · 1
cnot 2 3         # x_2 · s_2 = x_2 · 1

# Final Hadamards
h 0
h 1
h 2

# Measure - should get the hidden string "101"
measure 0
measure 1
measure 2

# Algorithm 3: Quantum Phase Kickback
# Demonstrates phase kickback in controlled operations

reset
resize 3

# Prepare control qubit in superposition
h 0

# Prepare target qubit as eigenstate of Z gate
x 1              # |1⟩ is eigenstate of Z with eigenvalue -1

# Apply controlled-Z
cz 0 1

# The phase kicks back to control qubit
# Control qubit is now (|0⟩ - |1⟩)/√2
bloch 0          # Should show Z = -1

# Algorithm 4: Quantum Interferometry
# Shows constructive and destructive interference

reset
resize 2

# Create equal superposition
h 0
h 1

# Apply relative phase
z 0              # Phase flip on qubit 0

# Interference - apply Hadamard again
h 0
h 1

# Measure - should see interference effects
probabilities
measure_all

# Algorithm 5: Quantum Random Walk
# Simple quantum walk on a line

reset
resize 4

# Position qubits (2 qubits for 4 positions) + coin qubit
# Start at position |00⟩, coin in superposition
h 2              # Coin qubit in superposition

# Step 1: Conditional shift based on coin
# If coin is |0⟩, move left; if |1⟩, move right
# This is a simplified version - full implementation would need more qubits

cnot 2 0         # Controlled operation based on coin
cnot 2 1

# Show position distribution
probabilities

# Algorithm 6: Quantum Amplitude Amplification
# Generalization of Grover's algorithm

reset
resize 3

# Initialize uniform superposition
h 0
h 1
h 2

# Mark state |110⟩ (flip its phase)
# Oracle: flip phase if qubits 0 and 1 are both 1
x 2              # Prepare ancilla
h 2              # Put in |−⟩ state
toffoli 0 1 2    # Three-qubit oracle
h 2              # Return to computational basis
x 2              # Return ancilla to |0⟩

# Diffusion operator (inversion about average)
h 0
h 1
h 2
x 0
x 1
x 2
toffoli 0 1 2    # Multi-controlled phase flip
x 0
x 1
x 2
h 0
h 1
h 2

# Check amplification
probabilities
measure_all
