ghz_detailed.qs

# Comprehensive GHZ State Tutorial
# Greenberger-Horne-Zeilinger States - Multi-qubit Entanglement

# GHZ states are maximally entangled states for n qubits
# For 3 qubits: |GHZ⟩ = (|000⟩ + |111⟩)/√2
# For n qubits: |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2

# Part 1: Basic 3-qubit GHZ state
reset
resize 3

# Method 1: Manual construction
h 0              # Create superposition on first qubit
cnot 0 1         # Entangle second qubit
cnot 0 2         # Entangle third qubit

# Analyze the state
state
probabilities

# Verify it's a GHZ state - should see only |000⟩ and |111⟩
# with equal probability (0.5 each)

# Check entanglement properties
entanglement

# Each qubit should be maximally mixed (entropy = 1)
bloch 0
bloch 1
bloch 2

# Show the circuit
draw

# Part 2: Test GHZ correlations
# GHZ states have special correlation properties
# If we measure any two qubits, the third is determined

# Save the GHZ state
snapshot ghz_3qubit

# Test correlation 1: Measure qubits 0 and 1
measure 0
measure 1
# Now measure qubit 2 - should match the XOR of first two
measure 2

# Restore and test again
load_snapshot ghz_3qubit

# Test correlation 2: Different measurement order
measure 1
measure 2
measure 0

# Part 3: 4-qubit GHZ state
reset
resize 4

# Create 4-qubit GHZ state
h 0
cnot 0 1
cnot 0 2
cnot 0 3

# Analyze 4-qubit GHZ
state
probabilities
entanglement

# Show circuit
draw

# Part 4: 5-qubit GHZ state
reset
resize 5

# Create 5-qubit GHZ state
h 0
cnot 0 1
cnot 0 2
cnot 0 3
cnot 0 4

# Analyze 5-qubit GHZ
state
probabilities
entanglement

# Part 5: Compare with W state
reset
resize 3

# Create W state for comparison
# W state: (|001⟩ + |010⟩ + |100⟩)/√3
w_state 0 1 2

# Compare entanglement properties
entanglement
bloch 0
bloch 1
bloch 2

# Part 6: GHZ state under noise (simulation)
reset
resize 3

# Create GHZ state
h 0
cnot 0 1
cnot 0 2

# Simulate bit flip error on qubit 1
x 1

# See how error affects GHZ state
state
probabilities

# The state is now (|010⟩ + |101⟩)/√2 - still entangled but different
entanglement

# Part 7: Using built-in GHZ command
reset
resize 4

# Quick GHZ creation
ghz 0 1 2 3

# Verify it's the same as manual construction
state
probabilities
