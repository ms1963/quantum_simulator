# Quantum Optimization - QAOA for Max-Cut
# Quantum Approximate Optimization Algorithm

# Reset
reset

# Initialize uniform superposition over all bit strings
h 0
h 1
h 2

# QAOA Level 1
# Problem Hamiltonian evolution (γ parameter)
gate CRZ 0 1 0.5    # Edge (0,1) with weight γ
gate CRZ 1 2 0.5    # Edge (1,2) with weight γ  
gate CRZ 0 2 0.5    # Edge (0,2) with weight γ

# Mixer Hamiltonian evolution (β parameter)
rx 0.3 0            # β parameter
rx 0.3 1
rx 0.3 2

# QAOA Level 2 (deeper circuit)
# Problem Hamiltonian with different parameter
gate CRZ 0 1 0.7
gate CRZ 1 2 0.7
gate CRZ 0 2 0.7

# Mixer with different parameter
rx 0.2 0
rx 0.2 1
rx 0.2 2

# Show final state
state
probabilities

# Sample solutions
measure_all

# In real QAOA, parameters are optimized classically
