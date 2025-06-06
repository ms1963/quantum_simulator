# Quantum Sensing - Ramsey Interferometry
# Demonstrates quantum-enhanced parameter estimation

# Reset
reset

# Prepare sensor in superposition
h 0

# Evolution under unknown phase φ (parameter to estimate)
# For demo, φ = π/3
rz 1.047 0   # φ = π/3

# Add more sensor qubits for enhanced precision
h 1
rz 1.047 1   # Same phase evolution

# Entangle sensors for quantum enhancement
cnot 0 1

# More phase evolution
rz 1.047 0
rz 1.047 1

# Final analysis pulse
h 0
h 1

# Show measurement distribution
probabilities

# The interference fringes encode the phase information
# Quantum entanglement provides √N enhancement in precision

measure_all
