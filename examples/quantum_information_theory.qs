# Quantum Information Theory Examples
# Demonstrates key concepts in quantum information

# Example 1: Quantum No-Cloning Theorem Demonstration
# Shows that arbitrary quantum states cannot be perfectly cloned

reset
resize 3

# Prepare unknown state |ψ⟩ = α|0⟩ + β|1⟩
# For demo, use |+⟩ state
h 0

# Attempt to clone using a "cloning machine"
# This is doomed to fail for arbitrary states
cnot 0 1         # Try to copy qubit 0 to qubit 1

# Check if cloning worked
bloch 0          # Original qubit
bloch 1          # "Cloned" qubit

# They should be identical if cloning worked
# But this only works for specific states, not arbitrary ones

# Example 2: Quantum Dense Coding
# Send 2 classical bits using 1 qubit (with pre-shared entanglement)

reset
resize 2

# Step 1: Alice and Bob share a Bell state
h 0
cnot 0 1

# Step 2: Alice wants to send 2 bits: "11"
# She applies operations based on the message:
# "00" → I, "01" → X, "10" → Z, "11" → XZ
x 0              # For message "11"
z 0

# Step 3: Alice sends her qubit to Bob
# Bob now has both qubits and can decode

# Step 4: Bob decodes by Bell measurement
cnot 0 1
h 0

# Step 5: Bob measures to get the 2-bit message
measure 0
measure 1
# Result should be "11"

# Example 3: Quantum Key Distribution (BB84 Protocol Simulation)
# Demonstrates secure key distribution

reset
resize 4

# Alice prepares random bits in random bases
# Bit 0, basis + (rectilinear)
h 0              # |+⟩ state

# Bit 1, basis × (diagonal)  
x 1
h 1              # |−⟩ state

# Bit 0, basis +
# (already |0⟩)

# Bit 1, basis ×
x 3
h 3              # |−⟩ state

# Bob measures in random bases
# If bases match, he gets Alice's bit correctly
# If bases don't match, results are random

# Show the prepared states
bloch 0
bloch 1
bloch 2
bloch 3

# Example 4: Quantum Error Syndrome Detection
# Shows how to detect errors without disturbing the quantum state

reset
resize 5

# Prepare a logical qubit in |+⟩ state using 3-qubit repetition code
h 0
cnot 0 1
cnot 0 2

# Introduce an error (bit flip on qubit 1)
x 1

# Syndrome measurement using ancilla qubits
cnot 0 3         # Syndrome qubit 1
cnot 1 3
cnot 1 4         # Syndrome qubit 2
cnot 2 4

# Measure syndrome qubits to detect error location
measure 3
measure 4
# Result "01" indicates error on qubit 1

# Example 5: Quantum Fidelity Measurement
# Compare similarity between quantum states

reset
resize 2

# Prepare first state |ψ⟩
h 0
ry 0.5 0

# Save for comparison
snapshot state1

# Prepare second state |φ⟩ (slightly different)
reset
h 0
ry 0.6 0

# In a real implementation, we'd calculate fidelity |⟨ψ|φ⟩|²
# Here we just show the different states
bloch 0

# Restore first state for comparison
load_snapshot state1
bloch 0

# Example 6: Quantum Entanglement Swapping
# Create entanglement between qubits that never interacted

reset
resize 4

# Create two Bell pairs
h 0
cnot 0 1         # Bell pair 1: qubits 0,1

h 2
cnot 2 3         # Bell pair 2: qubits 2,3

# Now qubits 0,1 are entangled and qubits 2,3 are entangled
# But 0,2 and 1,3 are not entangled

# Bell measurement on qubits 1,2 (the middle qubits)
cnot 1 2
h 1

# After measurement, qubits 0 and 3 become entangled!
# This is entanglement swapping

# Check entanglement between all pairs
entanglement

# Example 7: Quantum Zeno Effect Simulation
# Frequent measurements can freeze quantum evolution

reset
resize 1

# Prepare |+⟩ state
h 0

# Apply a small rotation
ry 0.1 0

# In real Zeno effect, we'd measure repeatedly
# Here we just show the effect of measurement on evolution
bloch 0

# Measure (this would "freeze" the evolution in real experiment)
measure 0
