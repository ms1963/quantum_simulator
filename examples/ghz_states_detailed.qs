# GHZ State - Detailed Implementation and Analysis
# Greenberger-Horne-Zeilinger states demonstrate quantum non-locality

# GHZ states are multi-qubit entangled states of the form:
# |GHZ_n⟩ = (|00...0⟩ + |11...1⟩)/√2

reset

# Method 1: Manual GHZ construction for 3 qubits
h 0              # Create superposition on qubit 0
cnot 0 1         # Entangle qubit 0 with qubit 1
cnot 0 2         # Entangle qubit 0 with qubit 2

# Show the resulting GHZ state
state            # Should show (|000⟩ + |111⟩)/√2
probabilities    # 50% chance each for |000⟩ and |111⟩

# Analyze the entanglement structure
entanglement     # All qubits should show maximum entanglement

# Individual qubits in GHZ state have zero Bloch vector
bloch 0          # Should be (0, 0, 0) - maximally mixed
bloch 1          # Should be (0, 0, 0) - maximally mixed  
bloch 2          # Should be (0, 0, 0) - maximally mixed

# Show the circuit so far
draw

# Measure one qubit and see how it affects the others
measure 0        # This will determine the other qubits!

# Now the remaining qubits are in a definite state
state            # Should be either |000⟩ or |111⟩
bloch 1          # Now has definite Bloch vector
bloch 2          # Matches qubit 1

# Reset and try larger GHZ state
reset

# Method 2: 4-qubit GHZ state
h 0              # Superposition
cnot 0 1         # Chain of entanglement
cnot 0 2
cnot 0 3

state            # (|0000⟩ + |1111⟩)/√2
probabilities
entanglement     # Even stronger multi-party entanglement

# Test GHZ non-locality
# In a GHZ state, measuring X on all qubits gives:
# Odd number of +1 results with probability 0
# Even number of +1 results with probability 1

# Transform to X basis for measurement
h 0
h 1  
h 2
h 3

state            # Now in X eigenbasis
measure_all      # Count the +1 results (represented as 0s in Z basis)

# Reset and demonstrate built-in GHZ command
reset
ghz              # Creates GHZ state on all available qubits
state
entanglement
draw
