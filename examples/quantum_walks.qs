examples/quantum_walks.qs

# Quantum Random Walk
# Demonstrates quantum analog of classical random walk

reset

# 1D Quantum Walk on a line
# Position encoded in qubits, coin qubit determines direction

# For 3-qubit system: 1 coin + 2 position qubits (4 positions)
# Position |00⟩, |01⟩, |10⟩, |11⟩ represent positions 0,1,2,3

# Start at position |01⟩ (position 1)
x 1

# Coin starts in |0⟩ (already there)
state            # |001⟩ = coin|0⟩ at position|01⟩

# Step 1: Apply coin operation (Hadamard)
h 0              # Coin in superposition

state            # Equal superposition of moving left/right

# Step 1: Conditional displacement based on coin
# If coin |0⟩: move left (subtract 1 from position)
# If coin |1⟩: move right (add 1 to position)

# This requires controlled arithmetic, simplified here:
# Use controlled operations to simulate position changes

cnot 0 1         # Simplified position control
cnot 0 2

state

# Step 2: Another coin flip
h 0

# Step 2: Another conditional displacement  
cnot 0 1
cnot 0 2
cnot 1 2         # Additional position coupling

state
probabilities    # Quantum interference creates non-classical pattern

# Compare with classical random walk:
# Classical: binomial distribution centered at start
# Quantum: interference creates complex probability pattern

# Multi-step walk
h 0              # Coin flip
cnot 0 1         # Position update
cnot 0 2

h 0              # Another step
cnot 0 1
cnot 0 2

h 0              # Another step
cnot 0 1
cnot 0 2

state
probabilities    # Final position distribution

# 2D Quantum Walk (simplified)
reset

# Need more qubits for 2D: 2 coin qubits + position qubits
# Simplified demonstration with current qubits

h 0              # X-direction coin
h 1              # Y-direction coin

# Position updates based on both coins
cnot 0 2         # X-movement
cnot 1 2         # Y-movement

state

# Multiple 2D steps
h 0
h 1
cnot 0 2
cnot 1 2

h 0  
h 1
cnot 0 2
cnot 1 2

state
probabilities

# Quantum walks are used for:
# - Quantum algorithms (search, element distinctness)
# - Quantum transport in physical systems
# - Quantum simulation of diffusion processes

draw
