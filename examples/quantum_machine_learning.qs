examples/quantum_machine_learning.qs

# Quantum Machine Learning - Variational Classifier
# Demonstrates parameterized quantum circuit for classification

reset

# Problem: Classify 2-bit input data using quantum circuit
# Training data: |00⟩ → class 0, |11⟩ → class 1

# Feature map: encode classical data into quantum state
# For input |00⟩ (classical):
# Already in |00⟩ quantum state

# Apply feature map transformation
ry 0.5 0         # Feature map parameter φ₁
ry 0.3 1         # Feature map parameter φ₂

# Add entanglement in feature map
cnot 0 1

state            # Encoded feature state

# Variational ansatz for classification
# Layer 1: Parameterized rotations (trainable parameters)
ry 0.8 0         # Trainable parameter θ₁
ry 0.6 1         # Trainable parameter θ₂

# Layer 1: Entangling gates
cnot 0 1
cnot 1 0

# Layer 2: More trainable parameters
ry 0.4 0         # Trainable parameter θ₃
ry 0.7 1         # Trainable parameter θ₄

# Final entanglement
cnot 0 1

state            # Final classifier state

# Measurement for classification
# Typically measure expectation value of Pauli-Z
probabilities    # Probability distribution
bloch 0          # Z-component gives classification score

# For input |11⟩ (different class):
reset
x 0              # Encode |11⟩
x 1

# Same feature map
ry 0.5 0         # Same feature map parameters
ry 0.3 1
cnot 0 1

# Same variational circuit (same trained parameters)
ry 0.8 0         # Same θ₁
ry 0.6 1         # Same θ₂
cnot 0 1
cnot 1 0
ry 0.4 0         # Same θ₃  
ry 0.7 1         # Same θ₄
cnot 0 1

state            # Different output for different class
bloch 0          # Different Z-component for classification

# In real quantum ML:
# 1. Parameters θᵢ are optimized on training data
# 2. Cost function measures classification accuracy
# 3. Classical optimizer updates parameters
# 4. Repeat until convergence

draw
count            # Show circuit complexity
