#-------------------------
#
# Quantum Simulator
# (c) 2025, Michael Stal 
#
#-------------------------

import numpy as np
from typing import List, Tuple, Optional
import cmath


class QuantumGates:
    """Library of quantum gates as unitary matrices."""
    
    def get_single_qubit_gate(self, gate_name: str, **kwargs) -> Optional[np.ndarray]:
        """Get single-qubit gate matrix by name"""
        gate_name = gate_name.upper()
        
        if gate_name == 'I' or gate_name == 'ID':
            return self.I()
        elif gate_name == 'X':
            return self.X()
        elif gate_name == 'Y':
            return self.Y()
        elif gate_name == 'Z':
            return self.Z()
        elif gate_name == 'H':
            return self.H()
        elif gate_name == 'S':
            return self.S()
        elif gate_name == 'T':
            return self.T()
        elif gate_name == 'S_DAGGER' or gate_name == 'SDAGGER':
            return self.S_dagger()
        elif gate_name == 'T_DAGGER' or gate_name == 'TDAGGER':
            return self.T_dagger()
        elif gate_name == 'RX':
            theta = kwargs.get('theta', 0)
            return self.RX(theta)
        elif gate_name == 'RY':
            theta = kwargs.get('theta', 0)
            return self.RY(theta)
        elif gate_name == 'RZ':
            theta = kwargs.get('theta', 0)
            return self.RZ(theta)
        elif gate_name == 'PHASE' or gate_name == 'P':
            phi = kwargs.get('phi', 0)
            return self.phase(phi)
        elif gate_name == 'U1':
            lam = kwargs.get('lambda', 0)
            return self.U1(lam)
        elif gate_name == 'U2':
            phi = kwargs.get('phi', 0)
            lam = kwargs.get('lambda', 0)
            return self.U2(phi, lam)
        elif gate_name == 'U3' or gate_name == 'U':
            theta = kwargs.get('theta', 0)
            phi = kwargs.get('phi', 0)
            lam = kwargs.get('lambda', 0)
            return self.U(theta, phi, lam)
        
        return None
    
    def get_two_qubit_gate(self, gate_name: str, **kwargs) -> Optional[np.ndarray]:
        """Get two-qubit gate matrix by name"""
        gate_name = gate_name.upper()
        
        if gate_name == 'CNOT' or gate_name == 'CX':
            return self.CNOT()
        elif gate_name == 'CZ':
            return self.CZ()
        elif gate_name == 'CY':
            return self.CY()
        elif gate_name == 'SWAP':
            return self.SWAP()
        elif gate_name == 'ISWAP':
            return self.iSWAP()
        elif gate_name == 'SQRT_SWAP':
            return self.SQRT_SWAP()
        elif gate_name == 'CRX':
            theta = kwargs.get('theta', 0)
            return self.CRX(theta)
        elif gate_name == 'CRY':
            theta = kwargs.get('theta', 0)
            return self.CRY(theta)
        elif gate_name == 'CRZ':
            theta = kwargs.get('theta', 0)
            return self.CRZ(theta)
        elif gate_name == 'CPHASE' or gate_name == 'CP':
            phi = kwargs.get('phi', 0)
            return self.controlled_gate(self.phase(phi))
        elif gate_name == 'CU':
            theta = kwargs.get('theta', 0)
            phi = kwargs.get('phi', 0)
            lam = kwargs.get('lambda', 0)
            gamma = kwargs.get('gamma', 0)
            return self.CU(theta, phi, lam, gamma)
        
        return None
    
    def get_three_qubit_gate(self, gate_name: str) -> Optional[np.ndarray]:
        """Get three-qubit gate matrix by name"""
        gate_name = gate_name.upper()
        
        if gate_name == 'TOFFOLI' or gate_name == 'CCX':
            return self.TOFFOLI()
        elif gate_name == 'FREDKIN' or gate_name == 'CSWAP':
            return self.FREDKIN()
        
        return None
    
    # Pauli Gates
    @staticmethod
    def I() -> np.ndarray:
        """Identity gate."""
        return np.array([[1, 0], [0, 1]], dtype=complex)
    
    @staticmethod
    def X() -> np.ndarray:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def Y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def Z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Hadamard Gate
    @staticmethod
    def H() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    # Phase Gates
    @staticmethod
    def S() -> np.ndarray:
        """S gate (phase gate)."""
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    
    @staticmethod
    def S_dagger() -> np.ndarray:
        """S† gate (conjugate of phase gate)."""
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    
    @staticmethod
    def T() -> np.ndarray:
        """T gate (π/8 gate)."""
        return np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]], dtype=complex)
    
    @staticmethod
    def T_dagger() -> np.ndarray:
        """T† gate (conjugate of T gate)."""
        return np.array([[1, 0], [0, cmath.exp(-1j * np.pi / 4)]], dtype=complex)
    
    # Rotation Gates
    @staticmethod
    def RX(theta: float) -> np.ndarray:
        """Rotation around X-axis."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -1j * sin_half], 
                        [-1j * sin_half, cos_half]], dtype=complex)
    
    @staticmethod
    def RY(theta: float) -> np.ndarray:
        """Rotation around Y-axis."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -sin_half], 
                        [sin_half, cos_half]], dtype=complex)
    
    @staticmethod
    def RZ(theta: float) -> np.ndarray:
        """Rotation around Z-axis."""
        exp_neg = cmath.exp(-1j * theta / 2)
        exp_pos = cmath.exp(1j * theta / 2)
        return np.array([[exp_neg, 0], [0, exp_pos]], dtype=complex)
    
    @staticmethod
    def phase(phi: float) -> np.ndarray:
        """Phase gate with arbitrary phase."""
        return np.array([[1, 0], [0, cmath.exp(1j * phi)]], dtype=complex)
    
    # Universal Single-Qubit Gate
    @staticmethod
    def U(theta: float, phi: float, lam: float) -> np.ndarray:
        """Universal single-qubit gate U(θ,φ,λ)."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([
            [cos_half, -cmath.exp(1j * lam) * sin_half],
            [cmath.exp(1j * phi) * sin_half, cmath.exp(1j * (phi + lam)) * cos_half]
        ], dtype=complex)
    
    @staticmethod
    def U1(lam: float) -> np.ndarray:
        """Single-parameter phase gate."""
        return np.array([[1, 0], [0, cmath.exp(1j * lam)]], dtype=complex)
    
    @staticmethod
    def U2(phi: float, lam: float) -> np.ndarray:
        """Two-parameter single-qubit gate."""
        return QuantumGates.U(np.pi/2, phi, lam)
    
    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> np.ndarray:
        """Three-parameter single-qubit gate (same as U)."""
        return QuantumGates.U(theta, phi, lam)
    
    # Two-Qubit Gates
    @staticmethod
    def CNOT() -> np.ndarray:
        """Controlled-NOT gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @staticmethod
    def CX() -> np.ndarray:
        """Alias for CNOT."""
        return QuantumGates.CNOT()
    
    @staticmethod
    def CZ() -> np.ndarray:
        """Controlled-Z gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
    
    @staticmethod
    def CY() -> np.ndarray:
        """Controlled-Y gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=complex)
    
    @staticmethod
    def SWAP() -> np.ndarray:
        """SWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @staticmethod
    def iSWAP() -> np.ndarray:
        """iSWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @staticmethod
    def SQRT_SWAP() -> np.ndarray:
        """Square root of SWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, (1+1j)/2, (1-1j)/2, 0],
            [0, (1-1j)/2, (1+1j)/2, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    # Controlled Rotation Gates
    @staticmethod
    def CRX(theta: float) -> np.ndarray:
        """Controlled rotation around X-axis."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos_half, -1j * sin_half],
            [0, 0, -1j * sin_half, cos_half]
        ], dtype=complex)
    
    @staticmethod
    def CRY(theta: float) -> np.ndarray:
        """Controlled rotation around Y-axis."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos_half, -sin_half],
            [0, 0, sin_half, cos_half]
        ], dtype=complex)
    
    @staticmethod
    def CRZ(theta: float) -> np.ndarray:
        """Controlled rotation around Z-axis."""
        exp_neg = cmath.exp(-1j * theta / 2)
        exp_pos = cmath.exp(1j * theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp_neg, 0],
            [0, 0, 0, exp_pos]
        ], dtype=complex)
    
    @staticmethod
    def CU(theta: float, phi: float, lam: float, gamma: float = 0) -> np.ndarray:
        """Controlled universal gate."""
        u_gate = QuantumGates.U(theta, phi, lam)
        phase_factor = cmath.exp(1j * gamma)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, phase_factor * u_gate[0,0], phase_factor * u_gate[0,1]],
            [0, 0, phase_factor * u_gate[1,0], phase_factor * u_gate[1,1]]
        ], dtype=complex)
    
    # Three-Qubit Gates
    @staticmethod
    def TOFFOLI() -> np.ndarray:
        """Toffoli (CCX) gate."""
        gate = np.eye(8, dtype=complex)
        gate[6, 6] = 0
        gate[6, 7] = 1
        gate[7, 6] = 1
        gate[7, 7] = 0
        return gate
    
    @staticmethod
    def CCX() -> np.ndarray:
        """Alias for Toffoli gate."""
        return QuantumGates.TOFFOLI()
    
    @staticmethod
    def FREDKIN() -> np.ndarray:
        """Fredkin (CSWAP) gate."""
        gate = np.eye(8, dtype=complex)
        gate[5, 5] = 0
        gate[5, 6] = 1
        gate[6, 5] = 1
        gate[6, 6] = 0
        return gate
    
    @staticmethod
    def CSWAP() -> np.ndarray:
        """Alias for Fredkin gate."""
        return QuantumGates.FREDKIN()
    
    # Custom Gate Construction
    @staticmethod
    def controlled_gate(base_gate: np.ndarray) -> np.ndarray:
        """Create controlled version of any single-qubit gate."""
        n = base_gate.shape[0]
        if base_gate.shape != (n, n):
            raise ValueError("Base gate must be square")
        
        controlled = np.eye(2 * n, dtype=complex)
        controlled[n:, n:] = base_gate
        return controlled
    
    @staticmethod
    def tensor_product(*gates: np.ndarray) -> np.ndarray:
        """Compute tensor product of multiple gates."""
        if not gates:
            raise ValueError("At least one gate required")
        
        result = gates[0]
        for gate in gates[1:]:
            result = np.kron(result, gate)
        return result
    
    @staticmethod
    def get_gate_list() -> List[str]:
        """Get list of all available gates."""
        return [
            'I', 'X', 'Y', 'Z', 'H', 'S', 'S_dagger', 'T', 'T_dagger',
            'RX', 'RY', 'RZ', 'phase', 'U', 'U1', 'U2', 'U3',
            'CNOT', 'CX', 'CZ', 'CY', 'SWAP', 'iSWAP', 'SQRT_SWAP',
            'CRX', 'CRY', 'CRZ', 'CU', 'TOFFOLI', 'CCX', 'FREDKIN', 'CSWAP'
        ]
    
    @staticmethod
    def get_gate_info(gate_name: str) -> str:
        """Get information about a specific gate."""
        info = {
            'I': 'Identity gate - does nothing',
            'X': 'Pauli-X gate - bit flip (NOT gate)',
            'Y': 'Pauli-Y gate - bit and phase flip',
            'Z': 'Pauli-Z gate - phase flip',
            'H': 'Hadamard gate - creates superposition',
            'S': 'S gate - phase gate (π/2 phase)',
            'S_dagger': 'S† gate - conjugate of S gate',
            'T': 'T gate - π/8 gate',
            'T_dagger': 'T† gate - conjugate of T gate',
            'RX': 'Rotation around X-axis (angle in radians)',
            'RY': 'Rotation around Y-axis (angle in radians)',
            'RZ': 'Rotation around Z-axis (angle in radians)',
            'phase': 'Phase gate with arbitrary angle',
            'U': 'Universal single-qubit gate U(θ,φ,λ)',
            'U1': 'Single-parameter phase gate',
            'U2': 'Two-parameter single-qubit gate',
            'U3': 'Three-parameter single-qubit gate',
            'CNOT': 'Controlled-NOT gate',
            'CX': 'Controlled-X gate (same as CNOT)',
            'CZ': 'Controlled-Z gate',
            'CY': 'Controlled-Y gate',
            'SWAP': 'SWAP gate - swaps two qubits',
            'iSWAP': 'iSWAP gate - SWAP with phase',
            'SQRT_SWAP': 'Square root of SWAP gate',
            'CRX': 'Controlled rotation around X-axis',
            'CRY': 'Controlled rotation around Y-axis',
            'CRZ': 'Controlled rotation around Z-axis',
            'CU': 'Controlled universal gate',
            'TOFFOLI': 'Toffoli gate - controlled-controlled-X',
            'CCX': 'Same as Toffoli gate',
            'FREDKIN': 'Fredkin gate - controlled-SWAP',
            'CSWAP': 'Same as Fredkin gate'
        }
        return info.get(gate_name, f"Unknown gate: {gate_name}")
