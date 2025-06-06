examples/quantum_key_distribution.qs

# Quantum Key Distribution (BB84 Protocol)
# Demonstrates quantum cryptographic key exchange

reset

# BB84 uses random bits and random bases to establish secure keys
# Alice prepares qubits, Bob measures them

# Simulation of BB84 protocol steps:

# Step 1: Alice prepares random qubits in random bases
# For demonstration, we'll show specific examples

# Example 1: Alice sends |0⟩ in Z basis
reset
# Qubit is already |0⟩
state

# Bob measures in Z basis (correct basis)
measure 0        # Should always get 0

# Example 2: Alice sends |1⟩ in Z basis  
reset
x 0              # Alice prepares |1⟩
state

# Bob measures in Z basis (correct basis)
measure 0        # Should always get 1

# Example 3: Alice sends |+⟩ in X basis
reset
h 0              # Alice prepares |+⟩ = (|0⟩ + |1⟩)/√2
state
probabilities    # 50/50 for 0 and 1

# Bob measures in X basis (correct basis)
h 0              # Bob applies H before measurement  
measure 0        # Should always get 0 (|+⟩ → |0⟩ after H)

# Example 4: Alice sends |−⟩ in X basis
reset
x 0              # Start with |1⟩
h 0              # Apply H to get |−⟩ = (|0⟩ - |1⟩)/√2
state

# Bob measures in X basis (correct basis)
h 0              # Bob applies H before measurement
measure 0        # Should always get 1 (|−⟩ → |1⟩ after H)

# Example 5: Basis mismatch - Alice sends |0⟩, Bob measures in X basis
reset
# Alice prepares |0⟩ in Z basis
state

# Bob mistakenly measures in X basis
h 0              # Bob applies H
measure 0        # Random result - 50/50

# Example 6: Multi-qubit key distribution simulation
reset

# Alice prepares 4 bits: 0101 in bases ZXZX
# Bit 0: |0⟩ in Z basis (already |0⟩)
h 1              # Bit 1: |+⟩ in X basis  
# Bit 2: |0⟩ in Z basis (already |0⟩)
x 3              # Bit 3: |1⟩ in Z basis
h 3              # Wait, that should be |−⟩ in X basis
x 3              # So start with |1⟩
h 3              # Then H to get |−⟩

state

# Bob measures in bases ZXXX (only first two match Alice's bases)
# Qubit 0: Z basis (matches Alice)
measure 0        # Should get 0

# Qubit 1: X basis (matches Alice) 
h 1              # Apply H before measurement
measure 1        # Should get 0 (|+⟩ → |0⟩)

# Qubit 2: X basis (Alice used Z - mismatch!)
h 2              # Wrong basis
measure 2        # Random result

# Qubit 3: X basis (matches Alice)
h 3              # Apply H before measurement  
measure 3        # Should get 1 (|−⟩ → |1⟩)

# In real BB84:
# 1. Alice and Bob compare bases publicly
# 2. Keep only bits where bases matched
# 3. Check subset for eavesdropping
# 4. Use remaining bits as secret key

draw
