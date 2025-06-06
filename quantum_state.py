#-------------------------
#
# Quantum Simulator
# (c) 2025, Michael Stal 
#
#-------------------------

"""
Quantum State Management
Handles quantum state vectors using NumPy tensors
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings


class QuantumState:
    """Represents and manages quantum state vectors using tensors."""
    
    def __init__(self, num_qubits: int):
        """Initialize quantum state with specified number of qubits."""
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        
        # Initialize to |0...0⟩ state
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0
        
        # Track measurement results
        self.measurement_results = []
        
    def reset(self):
        """Reset to |0...0⟩ state."""
        self.state_vector.fill(0)
        self.state_vector[0] = 1.0
        self.measurement_results.clear()
    
    def get_state_vector(self) -> np.ndarray:
        """Get copy of current state vector."""
        return self.state_vector.copy()
    
    def set_state_vector(self, state: np.ndarray):
        """Set state vector with normalization check."""
        if len(state) != self.num_states:
            raise ValueError(f"State vector must have {self.num_states} elements")
        
        # Normalize
        norm = np.linalg.norm(state)
        if abs(norm) < 1e-10:
            raise ValueError("State vector cannot be zero")
        
        self.state_vector = state / norm
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all computational basis states."""
        return np.abs(self.state_vector) ** 2
    
    def get_amplitude(self, state_index: int) -> complex:
        """Get amplitude for specific computational basis state."""
        if not 0 <= state_index < self.num_states:
            raise ValueError(f"State index must be between 0 and {self.num_states-1}")
        return self.state_vector[state_index]
    
    def get_probability(self, state_index: int) -> float:
        """Get probability for specific computational basis state."""
        return abs(self.get_amplitude(state_index)) ** 2
    
    def apply_unitary(self, unitary: np.ndarray, qubits: List[int]):
        """Apply unitary operation to specified qubits."""
        if len(qubits) == 0:
            raise ValueError("Must specify at least one qubit")
        
        # Validate qubits
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.num_qubits-1}]")
        
        if len(set(qubits)) != len(qubits):
            raise ValueError("Duplicate qubits not allowed")
        
        # Check unitary matrix dimensions
        expected_dim = 2 ** len(qubits)
        if unitary.shape != (expected_dim, expected_dim):
            raise ValueError(f"Unitary matrix must be {expected_dim}x{expected_dim}")
        
        # Check if matrix is unitary
        if not self._is_unitary(unitary):
            warnings.warn("Matrix may not be unitary", UserWarning)
        
        # Apply unitary using tensor operations
        if len(qubits) == 1:
            self._apply_single_qubit_unitary(unitary, qubits[0])
        else:
            self._apply_multi_qubit_unitary(unitary, qubits)
    
    def _is_unitary(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if matrix is unitary."""
        n = matrix.shape[0]
        identity = np.eye(n)
        product = np.dot(matrix, np.conj(matrix.T))
        return np.allclose(product, identity, atol=tolerance)
    
    def _apply_single_qubit_unitary(self, unitary: np.ndarray, qubit: int):
        """Apply single-qubit unitary using tensor operations."""
        # Reshape state vector to tensor form
        shape = [2] * self.num_qubits
        tensor = self.state_vector.reshape(shape)
        
        # Apply unitary to specified qubit axis
        tensor = np.tensordot(unitary, tensor, axes=([1], [qubit]))
        
        # Move the result back to the correct position
        tensor = np.moveaxis(tensor, 0, qubit)
        
        # Flatten back to state vector
        self.state_vector = tensor.flatten()
    
    def _apply_multi_qubit_unitary(self, unitary: np.ndarray, qubits: List[int]):
        """Apply multi-qubit unitary using tensor operations."""
        # Sort qubits for consistent ordering
        sorted_qubits = sorted(qubits)
        qubit_map = {q: i for i, q in enumerate(sorted_qubits)}
        
        # Reshape state vector to tensor form
        shape = [2] * self.num_qubits
        tensor = self.state_vector.reshape(shape)
        
        # Apply unitary to specified qubits
        axes_in = [qubit_map[q] + self.num_qubits for q in qubits]
        axes_tensor = sorted_qubits
        
        # Reshape unitary for tensor contraction
        unitary_shape = [2] * (2 * len(qubits))
        unitary_tensor = unitary.reshape(unitary_shape)
        
        # Contract tensors
        result = np.tensordot(unitary_tensor, tensor, axes=(axes_in, axes_tensor))
        
        # Rearrange axes back to original order
        axes_order = list(range(len(qubits))) + [i for i in range(self.num_qubits) if i not in sorted_qubits]
        result = np.transpose(result, axes_order)
        
        # Flatten back to state vector
        self.state_vector = result.flatten()
    
    def measure_qubit(self, qubit: int, collapse: bool = True) -> int:
        """Measure single qubit, optionally collapsing the state."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits-1}]")
        
        # Calculate probabilities for |0⟩ and |1⟩
        prob_0 = 0.0
        prob_1 = 0.0
        
        for state_idx in range(self.num_states):
            if (state_idx >> qubit) & 1 == 0:
                prob_0 += self.get_probability(state_idx)
            else:
                prob_1 += self.get_probability(state_idx)
        
        # Random measurement outcome
        result = 0 if np.random.random() < prob_0 else 1
        
        if collapse:
            self._collapse_state(qubit, result)
        
        self.measurement_results.append((qubit, result))
        return result
    
    def measure_all(self, collapse: bool = True) -> List[int]:
        """Measure all qubits."""
        probabilities = self.get_probabilities()
        
        # Random measurement outcome based on state probabilities
        state_idx = np.random.choice(self.num_states, p=probabilities)
        
        # Convert to binary representation
        results = []
        for q in range(self.num_qubits):
            results.append((state_idx >> q) & 1)
        
        if collapse:
            # Collapse to measured state
            self.state_vector.fill(0)
            self.state_vector[state_idx] = 1.0
        
        self.measurement_results.extend(enumerate(results))
        return results
    
    def _collapse_state(self, qubit: int, result: int):
        """Collapse state after measurement."""
        new_state = np.zeros_like(self.state_vector)
        norm_factor = 0.0
        
        for state_idx in range(self.num_states):
            qubit_value = (state_idx >> qubit) & 1
            if qubit_value == result:
                new_state[state_idx] = self.state_vector[state_idx]
                norm_factor += abs(self.state_vector[state_idx]) ** 2
        
        if norm_factor > 0:
            new_state /= np.sqrt(norm_factor)
            self.state_vector = new_state
    
    def get_bloch_vector(self, qubit: int) -> Tuple[float, float, float]:
        """Get Bloch sphere coordinates for a qubit (reduced density matrix)."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits-1}]")
        
        # Compute reduced density matrix for the qubit
        rho = self._reduced_density_matrix(qubit)
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Bloch vector components
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        
        return (x, y, z)
    
    def _reduced_density_matrix(self, qubit: int) -> np.ndarray:
        """Compute reduced density matrix for a single qubit."""
        # Reshape state vector to tensor
        shape = [2] * self.num_qubits
        tensor = self.state_vector.reshape(shape)
        
        # Compute density matrix tensor
        density_tensor = np.outer(tensor.flatten(), np.conj(tensor.flatten())).reshape([2] * (2 * self.num_qubits))
        
        # Trace out all qubits except the target
        other_qubits = [q for q in range(self.num_qubits) if q != qubit]
        
        # Perform partial trace
        for q in reversed(sorted(other_qubits)):
            # Sum over diagonal elements of qubit q
            density_tensor = np.trace(density_tensor, axis1=q, axis2=q + self.num_qubits)
        
        return density_tensor
    
    def __str__(self) -> str:
        """String representation of quantum state."""
        lines = [f"Quantum State ({self.num_qubits} qubits):"]
        
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:  # Only show non-zero amplitudes
                binary = format(i, f'0{self.num_qubits}b')
                prob = abs(amplitude) ** 2
                lines.append(f"  |{binary}⟩: {amplitude:.6f} (prob: {prob:.6f})")
        
        return "\n".join(lines)
