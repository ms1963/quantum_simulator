# Quantum Error Mitigation Techniques
# Demonstrates noise-resilient quantum computing methods

# Reset
reset

# Prepare a state that we want to protect
h 0
cnot 0 1

# Zero-noise extrapolation demonstration
# We'll simulate the same circuit with different "noise" levels

# Original circuit (no extra operations)
snapshot clean_circuit

# Circuit with identity gates (simulating noise)
x 0
x 0      # Net effect: identity, but adds "noise"
x 1  
x 1

# Show the effect
state

# Restore clean state for comparison
load_snapshot clean_circuit
state

# Symmetry verification
# Apply Pauli operators to verify state properties
x 0
x 1
state

# Restore
load_snapshot clean_circuit

# Randomized compiling (apply random Pauli gates)
z 0      # Random Pauli
y 1      # Random Pauli
y 1      # Undo
z 0      # Undo

# Final state should match original
state
