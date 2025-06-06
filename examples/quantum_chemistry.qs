# Quantum Chemistry - H₂ Molecule Simulation
# VQE for hydrogen molecule using minimal basis

# Reset
reset

# Prepare trial wavefunction for H₂ ground state
# Using simplified Jordan-Wigner transformation

# Initialize in Hartree-Fock state |01⟩
x 1

# Apply parameterized ansatz (UCCSD-inspired)
# Single excitation amplitude
ry 0.2 0
cnot 0 1
ry -0.2 0
cnot 0 1

# Double excitation amplitude  
h 0
h 1
cz 0 1
ry 0.1 0
ry 0.1 1
cz 0 1
h 0
h 1

# Show molecular wavefunction
state
probabilities

# This state would be used to compute energy expectation values
# E = ⟨ψ|H|ψ⟩ where H is the molecular Hamiltonian

# Measure to sample from wavefunction
measure_all# Quantum Communication - Superdense Coding
# Send 2 classical bits using 1 quantum bit

# Reset and prepare Bell pair (shared between Alice and Bob)
reset
h 0          # Alice's qubit
cnot 0 1     # Entangle with Bob's qubit

# Alice encodes 2 classical bits into her qubit
# For message "11" (both bits = 1):
z 0          # Encode second bit
x 0          # Encode first bit

# Alice sends her qubit to Bob
# Bob now has both qubits and can decode

# Bob's decoding procedure
cnot 0 1     # Disentangle
h 0          # Complete decoding

# Show final state
state

# Bob measures both qubits to recover the 2-bit message
measure 0    # First bit
measure 1    # Second bit

# The measurement results give Alice's original 2-bit message
