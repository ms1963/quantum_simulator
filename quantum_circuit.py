#-------------------------
#
# Quantum Simulator
# (c) 2025, Michael Stal 
#
#-------------------------

"""
Quantum Circuit Representation and Visualization
Handles circuit construction and rendering
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from quantum_gates import QuantumGates
import io


class QuantumOperation:
    """Represents a single quantum operation in a circuit."""
    
    def __init__(self, gate_name: str, qubits: List[int], parameters: Optional[List[float]] = None):
        self.gate_name = gate_name
        self.qubits = qubits.copy()
        self.parameters = parameters.copy() if parameters else []
        self.matrix = None  # Computed on demand
    
    def get_matrix(self) -> np.ndarray:
        """Get the unitary matrix for this operation."""
        if self.matrix is not None:
            return self.matrix
            
        gates = QuantumGates()
        
        # Single-qubit gates
        if self.gate_name == 'I':
            self.matrix = gates.I()
        elif self.gate_name == 'X':
            self.matrix = gates.X()
        elif self.gate_name == 'Y':
            self.matrix = gates.Y()
        elif self.gate_name == 'Z':
            self.matrix = gates.Z()
        elif self.gate_name == 'H':
            self.matrix = gates.H()
        elif self.gate_name == 'S':
            self.matrix = gates.S()
        elif self.gate_name == 'S_dagger':
            self.matrix = gates.S_dagger()
        elif self.gate_name == 'T':
            self.matrix = gates.T()
        elif self.gate_name == 'T_dagger':
            self.matrix = gates.T_dagger()
        
        # Parameterized single-qubit gates
        elif self.gate_name == 'RX':
            self.matrix = gates.RX(self.parameters[0])
        elif self.gate_name == 'RY':
            self.matrix = gates.RY(self.parameters[0])
        elif self.gate_name == 'RZ':
            self.matrix = gates.RZ(self.parameters[0])
        elif self.gate_name == 'phase':
            self.matrix = gates.phase(self.parameters[0])
        elif self.gate_name == 'U':
            self.matrix = gates.U(*self.parameters[:3])
        elif self.gate_name == 'U1':
            self.matrix = gates.U1(self.parameters[0])
        elif self.gate_name == 'U2':
            self.matrix = gates.U2(*self.parameters[:2])
        elif self.gate_name == 'U3':
            self.matrix = gates.U3(*self.parameters[:3])
        
        # Two-qubit gates
        elif self.gate_name in ['CNOT', 'CX']:
            self.matrix = gates.CNOT()
        elif self.gate_name == 'CZ':
            self.matrix = gates.CZ()
        elif self.gate_name == 'CY':
            self.matrix = gates.CY()
        elif self.gate_name == 'SWAP':
            self.matrix = gates.SWAP()
        elif self.gate_name == 'iSWAP':
            self.matrix = gates.iSWAP()
        elif self.gate_name == 'SQRT_SWAP':
            self.matrix = gates.SQRT_SWAP()
        
        # Controlled rotation gates
        elif self.gate_name == 'CRX':
            self.matrix = gates.CRX(self.parameters[0])
        elif self.gate_name == 'CRY':
            self.matrix = gates.CRY(self.parameters[0])
        elif self.gate_name == 'CRZ':
            self.matrix = gates.CRZ(self.parameters[0])
        elif self.gate_name == 'CU':
            self.matrix = gates.CU(*self.parameters[:4])
        
        # Three-qubit gates
        elif self.gate_name in ['TOFFOLI', 'CCX']:
            self.matrix = gates.TOFFOLI()
        elif self.gate_name in ['FREDKIN', 'CSWAP']:
            self.matrix = gates.FREDKIN()
        
        else:
            raise ValueError(f"Unknown gate: {self.gate_name}")
        
        return self.matrix
    
    def __str__(self) -> str:
        params_str = ""
        if self.parameters:
            params_str = f"({', '.join(f'{p:.3f}' for p in self.parameters)})"
        
        qubits_str = ', '.join(map(str, self.qubits))
        return f"{self.gate_name}{params_str} q[{qubits_str}]"


class QuantumCircuit:
    """Represents a quantum circuit with operations and measurements."""
    
    def __init__(self, num_qubits: int, name: str = "Circuit"):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        
        self.num_qubits = num_qubits
        self.name = name
        self.operations: List[QuantumOperation] = []
        self.measurements: List[Tuple[int, Optional[int]]] = []  # (qubit, classical_bit)
        self.classical_bits = 0
    
    def add_gate(self, gate_name: str, qubits: Union[int, List[int]], parameters: Optional[List[float]] = None):
        """Add a quantum gate to the circuit."""
        if isinstance(qubits, int):
            qubits = [qubits]
        
        # Validate qubits
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.num_qubits-1}]")
        
        # Validate gate requirements
        self._validate_gate(gate_name, qubits, parameters)
        
        operation = QuantumOperation(gate_name, qubits, parameters)
        self.operations.append(operation)
    
    def _validate_gate(self, gate_name: str, qubits: List[int], parameters: Optional[List[float]]):
        """Validate gate requirements."""
        single_qubit_gates = ['I', 'X', 'Y', 'Z', 'H', 'S', 'S_dagger', 'T', 'T_dagger']
        single_qubit_param_gates = ['RX', 'RY', 'RZ', 'phase', 'U1']
        two_qubit_gates = ['CNOT', 'CX', 'CZ', 'CY', 'SWAP', 'iSWAP', 'SQRT_SWAP']
        two_qubit_param_gates = ['CRX', 'CRY', 'CRZ']
        three_qubit_gates = ['TOFFOLI', 'CCX', 'FREDKIN', 'CSWAP']
        
        # Check qubit count
        if gate_name in single_qubit_gates or gate_name in single_qubit_param_gates:
            if len(qubits) != 1:
                raise ValueError(f"Gate {gate_name} requires exactly 1 qubit")
        elif gate_name in two_qubit_gates or gate_name in two_qubit_param_gates:
            if len(qubits) != 2:
                raise ValueError(f"Gate {gate_name} requires exactly 2 qubits")
        elif gate_name in three_qubit_gates:
            if len(qubits) != 3:
                raise ValueError(f"Gate {gate_name} requires exactly 3 qubits")
        elif gate_name == 'U2':
            if len(qubits) != 1:
                raise ValueError("Gate U2 requires exactly 1 qubit")
        elif gate_name == 'U' or gate_name == 'U3':
            if len(qubits) != 1:
                raise ValueError(f"Gate {gate_name} requires exactly 1 qubit")
        elif gate_name == 'CU':
            if len(qubits) != 2:
                raise ValueError("Gate CU requires exactly 2 qubits")
        
        # Check parameters
        param_requirements = {
            'RX': 1, 'RY': 1, 'RZ': 1, 'phase': 1, 'U1': 1,
            'U2': 2, 'U': 3, 'U3': 3, 'CU': 4,
            'CRX': 1, 'CRY': 1, 'CRZ': 1
        }
        
        if gate_name in param_requirements:
            required_params = param_requirements[gate_name]
            if not parameters or len(parameters) < required_params:
                raise ValueError(f"Gate {gate_name} requires {required_params} parameter(s)")
    
    def add_measurement(self, qubit: int, classical_bit: Optional[int] = None):
        """Add measurement operation."""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits-1}]")
        
        if classical_bit is None:
            classical_bit = self.classical_bits
            self.classical_bits += 1
        
        self.measurements.append((qubit, classical_bit))
    
    def measure_all(self):
        """Add measurements for all qubits."""
        for q in range(self.num_qubits):
            self.add_measurement(q)
    
    def depth(self) -> int:
        """Calculate circuit depth (longest path)."""
        if not self.operations:
            return 0
        
        # Track the latest operation time for each qubit
        qubit_times = [0] * self.num_qubits
        
        for op in self.operations:
            # Find the maximum time among qubits involved in this operation
            max_time = max(qubit_times[q] for q in op.qubits)
            # Update times for all qubits involved
            for q in op.qubits:
                qubit_times[q] = max_time + 1
        
        return max(qubit_times)
    
    def width(self) -> int:
        """Get circuit width (number of qubits)."""
        return self.num_qubits
    
    def count_ops(self) -> Dict[str, int]:
        """Count operations by gate type."""
        counts = {}
        for op in self.operations:
            counts[op.gate_name] = counts.get(op.gate_name, 0) + 1
        return counts
    
    def clear(self):
        """Clear all operations and measurements."""
        self.operations.clear()
        self.measurements.clear()
        self.classical_bits = 0
    
    def copy(self) -> 'QuantumCircuit':
        """Create a copy of the circuit."""
        new_circuit = QuantumCircuit(self.num_qubits, self.name + "_copy")
        
        for op in self.operations:
            new_circuit.add_gate(op.gate_name, op.qubits, op.parameters)
        
        for qubit, cbit in self.measurements:
            new_circuit.add_measurement(qubit, cbit)
        
        return new_circuit
    
    def reverse(self):
        """Reverse the order of operations."""
        self.operations.reverse()
    
    def append(self, other: 'QuantumCircuit'):
        """Append another circuit to this one."""
        if other.num_qubits != self.num_qubits:
            raise ValueError("Circuits must have the same number of qubits")
        
        for op in other.operations:
            self.operations.append(op)
        
        for qubit, cbit in other.measurements:
            self.add_measurement(qubit, cbit + self.classical_bits)
    
    def draw_ascii(self, show_measurements: bool = True) -> str:
        """Draw circuit in ASCII format."""
        if not self.operations and not self.measurements:
            return f"{self.name}: Empty circuit with {self.num_qubits} qubits"
        
        # Calculate circuit depth with measurements
        depth = self.depth()
        if show_measurements and self.measurements:
            depth += 1
        
        if depth == 0:
            depth = 1
        
        # Create grid
        lines = []
        for q in range(self.num_qubits):
            line = f"q{q}: |0⟩" + "─" * (depth * 6)
            lines.append(list(line))
        
        # Track position for each qubit
        positions = [5] * self.num_qubits  # Start after "|0⟩"
        
        # Add operations
        for op in self.operations:
            # Find maximum position among involved qubits
            max_pos = max(positions[q] for q in op.qubits)
            
            # Place gate symbol
            if len(op.qubits) == 1:
                q = op.qubits[0]
                gate_symbol = self._get_gate_symbol(op)
                
                # Center the symbol
                start_pos = max_pos
                for i, char in enumerate(gate_symbol):
                    if start_pos + i < len(lines[q]):
                        lines[q][start_pos + i] = char
                
                positions[q] = start_pos + len(gate_symbol) + 1
            
            elif len(op.qubits) == 2:
                q1, q2 = sorted(op.qubits)
                
                if op.gate_name in ['CNOT', 'CX']:
                    control_q = op.qubits[0]
                    target_q = op.qubits[1]
                    
                    # Control qubit
                    if max_pos < len(lines[control_q]):
                        lines[control_q][max_pos] = '●'
                    
                    # Target qubit
                    if max_pos < len(lines[target_q]):
                        lines[target_q][max_pos] = '⊕'
                    
                    # Vertical line
                    for q in range(min(control_q, target_q) + 1, max(control_q, target_q)):
                        if max_pos < len(lines[q]):
                            lines[q][max_pos] = '│'
                
                elif op.gate_name == 'CZ':
                    # Both qubits get control symbols
                    for q in op.qubits:
                        if max_pos < len(lines[q]):
                            lines[q][max_pos] = '●'
                    
                    # Vertical line
                    for q in range(q1 + 1, q2):
                        if max_pos < len(lines[q]):
                            lines[q][max_pos] = '│'
                
                elif op.gate_name == 'SWAP':
                    for q in op.qubits:
                        if max_pos < len(lines[q]):
                            lines[q][max_pos] = '×'
                    
                    # Vertical line
                    for q in range(q1 + 1, q2):
                        if max_pos < len(lines[q]):
                            lines[q][max_pos] = '│'
                
                # Update positions
                for q in op.qubits:
                    positions[q] = max_pos + 2
            
            elif len(op.qubits) == 3:
                # Three-qubit gates
                q1, q2, q3 = sorted(op.qubits)
                
                if op.gate_name in ['TOFFOLI', 'CCX']:
                    # Two controls, one target
                    controls = op.qubits[:2]
                    target = op.qubits[2]
                    
                    for q in controls:
                        if max_pos < len(lines[q]):
                            lines[q][max_pos] = '●'
                    
                    if max_pos < len(lines[target]):
                        lines[target][max_pos] = '⊕'
                    
                    # Vertical lines
                    for q in range(q1 + 1, q3):
                        if q not in op.qubits and max_pos < len(lines[q]):
                            lines[q][max_pos] = '│'
                
                # Update positions
                for q in op.qubits:
                    positions[q] = max_pos + 2
        
        # Add measurements
        if show_measurements and self.measurements:
            max_pos = max(positions)
            for qubit, cbit in self.measurements:
                if max_pos < len(lines[qubit]):
                    lines[qubit][max_pos:max_pos+3] = list('📊')
                positions[qubit] = max_pos + 4
        
        # Convert back to strings
        result_lines = [f"{self.name}:"]
        for line in lines:
            result_lines.append(''.join(line))
        
        # Add classical register info
        if self.measurements:
            result_lines.append(f"Classical bits: {self.classical_bits}")
        
        return '\n'.join(result_lines)
    
    def _get_gate_symbol(self, op: QuantumOperation) -> str:
        """Get ASCII symbol for gate."""
        symbols = {
            'I': '─I─',
            'X': '─X─',
            'Y': '─Y─',
            'Z': '─Z─',
            'H': '─H─',
            'S': '─S─',
            'T': '─T─',
            'S_dagger': '─S†',
            'T_dagger': '─T†'
        }
        
        if op.gate_name in symbols:
            return symbols[op.gate_name]
        elif op.gate_name.startswith('R'):
            angle = op.parameters[0] if op.parameters else 0
            return f'─R{op.gate_name[-1]}({angle:.2f})─'
        else:
            return f'─{op.gate_name}─'
    
    def to_qiskit(self) -> str:
        """Generate Qiskit-compatible code."""
        lines = [
            "from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister",
            "",
            f"# {self.name}",
            f"q = QuantumRegister({self.num_qubits}, 'q')"
        ]
        
        if self.measurements:
            lines.append(f"c = ClassicalRegister({self.classical_bits}, 'c')")
            lines.append(f"circuit = QuantumCircuit(q, c)")
        else:
            lines.append(f"circuit = QuantumCircuit(q)")
        
        lines.append("")
        
        # Add operations
        for op in self.operations:
            qubits_str = ', '.join(f'q[{q}]' for q in op.qubits)
            
            if op.parameters:
                params_str = ', '.join(f'{p:.6f}' for p in op.parameters)
                lines.append(f"circuit.{op.gate_name.lower()}({params_str}, {qubits_str})")
            else:
                lines.append(f"circuit.{op.gate_name.lower()}({qubits_str})")
        
        # Add measurements
        for qubit, cbit in self.measurements:
            lines.append(f"circuit.measure(q[{qubit}], c[{cbit}])")
        
        lines.extend([
            "",
            "print(circuit)",
            "# circuit.draw('mpl')  # For matplotlib visualization"
        ])
        
        return '\n'.join(lines)
    
    def save_qiskit(self, filename: str):
        """Save circuit as Qiskit code to file."""
        with open(filename, 'w') as f:
            f.write(self.to_qiskit())
    
    def save_ascii(self, filename: str):
        """Save ASCII representation to file."""
        with open(filename, 'w') as f:
            f.write(self.draw_ascii())
    
    def __str__(self) -> str:
        """String representation of the circuit."""
        return self.draw_ascii()
    
    def __len__(self) -> int:
        """Number of operations in the circuit."""
        return len(self.operations)
