#-------------------------
#
# Quantum Simulator
# (c) 2025, Michael Stal 
#
#-------------------------

"""
Quantum Circuit Drawing and Visualization
ASCII art circuit diagrams and Qiskit export
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import math


class QuantumCircuitDrawer:
    """Draws quantum circuits in ASCII format and exports to Qiskit."""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.circuit_operations = []
        self.circuit_depth = 0
        
    def add_operation(self, operation: Dict[str, Any]):
        """Add an operation to the circuit for drawing."""
        self.circuit_operations.append(operation)
        self.circuit_depth = max(self.circuit_depth, len(self.circuit_operations))
    
    def clear(self):
        """Clear the circuit."""
        self.circuit_operations.clear()
        self.circuit_depth = 0
    
    def draw_ascii(self) -> str:
        """Draw the circuit in ASCII format."""
        if not self.circuit_operations:
            return "Empty circuit"
        
        # Calculate circuit layout
        lines = []
        
        # Create header
        header = "Quantum Circuit"
        lines.append(header)
        lines.append("=" * len(header))
        lines.append("")
        
        # Initialize qubit lines
        qubit_lines = []
        for i in range(self.num_qubits):
            qubit_lines.append(f"q{i}: |0⟩")
        
        # Process operations in order
        max_width = max(len(line) for line in qubit_lines)
        
        for op in self.circuit_operations:
            gate_name = op.get('gate', 'UNKNOWN')
            qubits = op.get('qubits', [])
            gate_type = op.get('type', 'unknown')
            params = op.get('params', {})
            
            # Calculate gate representation
            if gate_type == 'single_qubit':
                self._add_single_qubit_gate(qubit_lines, gate_name, qubits[0], params)
            elif gate_type == 'two_qubit':
                self._add_two_qubit_gate(qubit_lines, gate_name, qubits, params)
            elif gate_type == 'three_qubit':
                self._add_three_qubit_gate(qubit_lines, gate_name, qubits)
            elif gate_type == 'measurement':
                self._add_measurement(qubit_lines, qubits[0])
        
        # Add final state indicators
        for i, line in enumerate(qubit_lines):
            lines.append(line + "─")
        
        return "\n".join(lines)
    
    def _add_single_qubit_gate(self, qubit_lines: List[str], gate_name: str, qubit: int, params: Dict):
        """Add a single-qubit gate to the ASCII representation."""
        # Ensure all lines have the same length
        max_len = max(len(line) for line in qubit_lines)
        for i in range(len(qubit_lines)):
            qubit_lines[i] = qubit_lines[i].ljust(max_len)
        
        # Add gate symbol
        gate_symbol = self._get_gate_symbol(gate_name, params)
        
        # Add connections
        for i in range(len(qubit_lines)):
            if i == qubit:
                qubit_lines[i] += f"─{gate_symbol}"
            else:
                qubit_lines[i] += "─" + "─" * len(gate_symbol)
    
    def _add_two_qubit_gate(self, qubit_lines: List[str], gate_name: str, qubits: List[int], params: Dict):
        """Add a two-qubit gate to the ASCII representation."""
        # Ensure all lines have the same length
        max_len = max(len(line) for line in qubit_lines)
        for i in range(len(qubit_lines)):
            qubit_lines[i] = qubit_lines[i].ljust(max_len)
        
        control, target = qubits[0], qubits[1]
        
        if gate_name in ['CNOT', 'CX']:
            # CNOT gate
            for i in range(len(qubit_lines)):
                if i == control:
                    qubit_lines[i] += "─●"
                elif i == target:
                    qubit_lines[i] += "─⊕"
                elif min(control, target) < i < max(control, target):
                    qubit_lines[i] += "─│"
                else:
                    qubit_lines[i] += "──"
        
        elif gate_name == 'CZ':
            # Controlled-Z gate
            for i in range(len(qubit_lines)):
                if i == control:
                    qubit_lines[i] += "─●"
                elif i == target:
                    qubit_lines[i] += "─●"
                elif min(control, target) < i < max(control, target):
                    qubit_lines[i] += "─│"
                else:
                    qubit_lines[i] += "──"
        
        elif gate_name == 'SWAP':
            # SWAP gate
            for i in range(len(qubit_lines)):
                if i in [control, target]:
                    qubit_lines[i] += "─×"
                elif min(control, target) < i < max(control, target):
                    qubit_lines[i] += "─│"
                else:
                    qubit_lines[i] += "──"
        
        else:
            # Generic controlled gate
            gate_symbol = self._get_gate_symbol(gate_name, params)
            for i in range(len(qubit_lines)):
                if i == control:
                    qubit_lines[i] += "─●"
                elif i == target:
                    qubit_lines[i] += f"─{gate_symbol}"
                elif min(control, target) < i < max(control, target):
                    qubit_lines[i] += "─│"
                else:
                    qubit_lines[i] += "─" + "─" * len(gate_symbol)
    
    def _add_three_qubit_gate(self, qubit_lines: List[str], gate_name: str, qubits: List[int]):
        """Add a three-qubit gate to the ASCII representation."""
        # Ensure all lines have the same length
        max_len = max(len(line) for line in qubit_lines)
        for i in range(len(qubit_lines)):
            qubit_lines[i] = qubit_lines[i].ljust(max_len)
        
        control1, control2, target = qubits[0], qubits[1], qubits[2]
        
        if gate_name in ['TOFFOLI', 'CCX']:
            # Toffoli gate
            for i in range(len(qubit_lines)):
                if i in [control1, control2]:
                    qubit_lines[i] += "─●"
                elif i == target:
                    qubit_lines[i] += "─⊕"
                elif min(qubits) < i < max(qubits):
                    qubit_lines[i] += "─│"
                else:
                    qubit_lines[i] += "──"
        
        elif gate_name in ['FREDKIN', 'CSWAP']:
            # Fredkin gate
            for i in range(len(qubit_lines)):
                if i == control1:
                    qubit_lines[i] += "─●"
                elif i in [control2, target]:
                    qubit_lines[i] += "─×"
                elif min(qubits) < i < max(qubits):
                    qubit_lines[i] += "─│"
                else:
                    qubit_lines[i] += "──"
    
    def _add_measurement(self, qubit_lines: List[str], qubit: int):
        """Add a measurement symbol to the ASCII representation."""
        # Ensure all lines have the same length
        max_len = max(len(line) for line in qubit_lines)
        for i in range(len(qubit_lines)):
            qubit_lines[i] = qubit_lines[i].ljust(max_len)
        
        # Add measurement symbol
        for i in range(len(qubit_lines)):
            if i == qubit:
                qubit_lines[i] += "─┤M├"
            else:
                qubit_lines[i] += "─────"
    
    def _get_gate_symbol(self, gate_name: str, params: Dict) -> str:
        """Get the symbol for a gate."""
        symbols = {
            'I': 'I',
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'H': 'H',
            'S': 'S',
            'T': 'T',
            'S_DAGGER': 'S†',
            'SDAGGER': 'S†',
            'T_DAGGER': 'T†',
            'TDAGGER': 'T†'
        }
        
        if gate_name in symbols:
            return symbols[gate_name]
        elif gate_name in ['RX', 'RY', 'RZ']:
            angle = params.get('theta', 0)
            if abs(angle - math.pi) < 1e-6:
                return gate_name  # Full rotation
            elif abs(angle - math.pi/2) < 1e-6:
                return gate_name  # π/2 rotation
            else:
                return f"{gate_name}({angle:.2f})"
        elif gate_name == 'PHASE':
            phi = params.get('phi', 0)
            return f"P({phi:.2f})"
        elif gate_name in ['U', 'U3']:
            theta = params.get('theta', 0)
            phi = params.get('phi', 0)
            lam = params.get('lambda', 0)
            return f"U({theta:.1f},{phi:.1f},{lam:.1f})"
        elif gate_name == 'U1':
            lam = params.get('lambda', 0)
            return f"U1({lam:.2f})"
        elif gate_name == 'U2':
            phi = params.get('phi', 0)
            lam = params.get('lambda', 0)
            return f"U2({phi:.1f},{lam:.1f})"
        else:
            return gate_name[:3]  # Truncate long names
    
    def export_qiskit(self, circuit_name: str = "circuit") -> str:
        """Export the circuit as Qiskit Python code."""
        lines = []
        
        # Header
        lines.append("# Quantum circuit exported from Quantum Simulator")
        lines.append("from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister")
        lines.append("from qiskit_aer import Aer")
        lines.append("import numpy as np")
        lines.append("")
        
        # Create circuit
        lines.append(f"# Create quantum circuit with {self.num_qubits} qubits")
        lines.append(f"q = QuantumRegister({self.num_qubits}, 'q')")
        lines.append(f"c = ClassicalRegister({self.num_qubits}, 'c')")
        lines.append(f"{circuit_name} = QuantumCircuit(q, c)")
        lines.append("")
        
        # Add operations
        lines.append("# Add quantum operations")
        for op in self.circuit_operations:
            gate_name = op.get('gate', 'UNKNOWN')
            qubits = op.get('qubits', [])
            gate_type = op.get('type', 'unknown')
            params = op.get('params', {})
            
            qiskit_code = self._get_qiskit_code(gate_name, qubits, gate_type, params, circuit_name)
            if qiskit_code:
                lines.append(qiskit_code)
        
        lines.append("")
        
        # Add execution example
        lines.append("# Execute the circuit")
        lines.append("# backend = Aer.get_backend('qasm_simulator')")
	lines.append("# tqc = transpile(circuit, backend)"
        lines.append("# job = backend.run(tqc, shots=1024)")
        lines.append("# result = job.result()")
        lines.append("# counts = result.get_counts()")
        lines.append("# print(counts)")
        lines.append("")
        
        # Add circuit drawing
        lines.append("# Draw the circuit")
        lines.append(f"print({circuit_name})")
        
        return "\n".join(lines)
    
    def _get_qiskit_code(self, gate_name: str, qubits: List[int], gate_type: str, params: Dict, circuit_name: str) -> str:
        """Convert a gate to Qiskit code."""
        if gate_type == 'single_qubit':
            qubit = qubits[0]
            
            if gate_name == 'I':
                return f"{circuit_name}.i(q[{qubit}])"
            elif gate_name == 'X':
                return f"{circuit_name}.x(q[{qubit}])"
            elif gate_name == 'Y':
                return f"{circuit_name}.y(q[{qubit}])"
            elif gate_name == 'Z':
                return f"{circuit_name}.z(q[{qubit}])"
            elif gate_name == 'H':
                return f"{circuit_name}.h(q[{qubit}])"
            elif gate_name == 'S':
                return f"{circuit_name}.s(q[{qubit}])"
            elif gate_name == 'T':
                return f"{circuit_name}.t(q[{qubit}])"
            elif gate_name in ['S_DAGGER', 'SDAGGER']:
                return f"{circuit_name}.sdg(q[{qubit}])"
            elif gate_name in ['T_DAGGER', 'TDAGGER']:
                return f"{circuit_name}.tdg(q[{qubit}])"
            elif gate_name == 'RX':
                theta = params.get('theta', 0)
                return f"{circuit_name}.rx({theta}, q[{qubit}])"
            elif gate_name == 'RY':
                theta = params.get('theta', 0)
                return f"{circuit_name}.ry({theta}, q[{qubit}])"
            elif gate_name == 'RZ':
                theta = params.get('theta', 0)
                return f"{circuit_name}.rz({theta}, q[{qubit}])"
            elif gate_name == 'PHASE':
                phi = params.get('phi', 0)
                return f"{circuit_name}.p({phi}, q[{qubit}])"
            elif gate_name == 'U1':
                lam = params.get('lambda', 0)
                return f"{circuit_name}.u1({lam}, q[{qubit}])"
            elif gate_name == 'U2':
                phi = params.get('phi', 0)
                lam = params.get('lambda', 0)
                return f"{circuit_name}.u2({phi}, {lam}, q[{qubit}])"
            elif gate_name in ['U', 'U3']:
                theta = params.get('theta', 0)
                phi = params.get('phi', 0)
                lam = params.get('lambda', 0)
                return f"{circuit_name}.u3({theta}, {phi}, {lam}, q[{qubit}])"
        
        elif gate_type == 'two_qubit':
            control, target = qubits[0], qubits[1]
            
            if gate_name in ['CNOT', 'CX']:
                return f"{circuit_name}.cx(q[{control}], q[{target}])"
            elif gate_name == 'CZ':
                return f"{circuit_name}.cz(q[{control}], q[{target}])"
            elif gate_name == 'CY':
                return f"{circuit_name}.cy(q[{control}], q[{target}])"
            elif gate_name == 'SWAP':
                return f"{circuit_name}.swap(q[{control}], q[{target}])"
            elif gate_name == 'CRX':
                theta = params.get('theta', 0)
                return f"{circuit_name}.crx({theta}, q[{control}], q[{target}])"
            elif gate_name == 'CRY':
                theta = params.get('theta', 0)
                return f"{circuit_name}.cry({theta}, q[{control}], q[{target}])"
            elif gate_name == 'CRZ':
                theta = params.get('theta', 0)
                return f"{circuit_name}.crz({theta}, q[{control}], q[{target}])"
            elif gate_name == 'CPHASE':
                phi = params.get('phi', 0)
                return f"{circuit_name}.cp({phi}, q[{control}], q[{target}])"
            elif gate_name == 'CU':
                theta = params.get('theta', 0)
                phi = params.get('phi', 0)
                lam = params.get('lambda', 0)
                gamma = params.get('gamma', 0)
                return f"{circuit_name}.cu3({theta}, {phi}, {lam}, q[{control}], q[{target}])"
        
        elif gate_type == 'three_qubit':
            control1, control2, target = qubits[0], qubits[1], qubits[2]
            
            if gate_name in ['TOFFOLI', 'CCX']:
                return f"{circuit_name}.ccx(q[{control1}], q[{control2}], q[{target}])"
            elif gate_name in ['FREDKIN', 'CSWAP']:
                return f"{circuit_name}.cswap(q[{control1}], q[{control2}], q[{target}])"
        
        elif gate_type == 'measurement':
            qubit = qubits[0]
            return f"{circuit_name}.measure(q[{qubit}], c[{qubit}])"
        
        return f"# Unknown gate: {gate_name}"

