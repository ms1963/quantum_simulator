#-------------------------
#
# Quantum Simulator
# (c) 2025, Michael Stal 
#
#-------------------------

import numpy as np
import cmath
from typing import List, Tuple, Optional, Dict, Any, Union
from quantum_gates import QuantumGates

class QuantumSimulator:
    def __init__(self, num_qubits: int):
        if num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        self.num_qubits = num_qubits
        self.num_states = 2 ** num_qubits
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0  # Initialize to |000...0⟩
        
        self.gates = QuantumGates()
        self.circuit_history = []
        self.measurement_results = []
        self.snapshots = {}  # Initialize snapshots dictionary
        
        print(f"Initialized {num_qubits}-qubit quantum simulator")
        print(f"State space dimension: {self.num_states}")
        print(f"Memory usage: ~{self.num_states * 16 / 1024:.2f} KB")
    
    def reset(self):
        """Reset the quantum state to |000...0⟩"""
        self.state_vector.fill(0)
        self.state_vector[0] = 1.0
        self.circuit_history.clear()
        self.measurement_results.clear()
        print("Quantum state reset to |000...0⟩")
    
    def resize(self, new_num_qubits: int):
        """Resize the quantum system to a different number of qubits"""
        if new_num_qubits <= 0:
            raise ValueError("Number of qubits must be positive")
        
        if new_num_qubits == self.num_qubits:
            print(f"Already using {new_num_qubits} qubits")
            return
        
        if new_num_qubits > 30:
            memory_gb = 2**new_num_qubits * 16 / (1024**3)
            print(f"Warning: {new_num_qubits} qubits will require {memory_gb:.2f} GB of memory")
            response = input("Continue? (y/N): ").lower()
            if response != 'y':
                return
        
        self.num_qubits = new_num_qubits
        self.num_states = 2 ** new_num_qubits
        self.state_vector = np.zeros(self.num_states, dtype=complex)
        self.state_vector[0] = 1.0
        self.circuit_history.clear()
        self.measurement_results.clear()
        
        print(f"Resized to {new_num_qubits} qubits")
        print(f"State space dimension: {self.num_states}")
        print(f"Memory usage: ~{self.num_states * 16 / 1024:.2f} KB")
    
    def get_state_vector(self) -> np.ndarray:
        """Get a copy of the current state vector"""
        return self.state_vector.copy()
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states"""
        return np.abs(self.state_vector) ** 2
    
    def state(self):
        """Display the current quantum state"""
        self.print_state()
    
    def probabilities(self):
        """Show measurement probabilities for all basis states"""
        probs = self.get_probabilities()
        print("Measurement probabilities:")
        for i, prob in enumerate(probs):
            if prob > 1e-10:  # Only show non-zero probabilities
                binary = format(i, f'0{self.num_qubits}b')
                print(f"  |{binary}⟩: {prob:.6f}")
    
    def amplitude(self, state_index: int):
        """Show amplitude of specific basis state"""
        if not 0 <= state_index < self.num_states:
            print(f"State index must be between 0 and {self.num_states-1}")
            return
        
        amp = self.state_vector[state_index]
        prob = abs(amp) ** 2
        binary = format(state_index, f'0{self.num_qubits}b')
        print(f"|{binary}⟩: {amp:.6f} (probability: {prob:.6f})")
    
    def bloch(self, qubit: int):
        """Show Bloch vector for a qubit"""
        try:
            x, y, z = self.get_bloch_vector(qubit)
            print(f"Bloch vector for qubit {qubit}:")
            print(f"  X: {x:.6f}")
            print(f"  Y: {y:.6f}")
            print(f"  Z: {z:.6f}")
            print(f"  |r|: {np.sqrt(x*x + y*y + z*z):.6f}")
        except Exception as e:
            print(f"Error calculating Bloch vector: {e}")
    
    def entanglement(self):
        """Show entanglement analysis"""
        try:
            info = self.get_entanglement_info()
            print("Entanglement Analysis:")
            print(f"  Overall entanglement: {info['entanglement_level']}")
            print(f"  Maximum entropy: {info['max_entropy']:.6f}")
            print("  Qubit entropies:")
            for qubit_key, entropy in info['qubit_entropies'].items():
                qubit_num = qubit_key.split('_')[1]
                print(f"    Qubit {qubit_num}: {entropy:.6f}")
        except Exception as e:
            print(f"Error calculating entanglement: {e}")
    
    def stats(self):
        """Show simulation statistics"""
        statistics = self.get_statistics()
        print("Simulation Statistics:")
        print(f"  Qubits: {statistics['num_qubits']}")
        print(f"  Total operations: {statistics['total_operations']}")
        print(f"  Circuit depth: {statistics['circuit_depth']}")
        print(f"  Memory usage: {statistics['memory_usage_mb']:.2f} MB")
        print(f"  State norm: {statistics['state_norm']:.6f}")
        
        if statistics['gate_counts']:
            print("  Gate counts:")
            for gate, count in sorted(statistics['gate_counts'].items()):
                print(f"    {gate}: {count}")
    
    def save_snapshot(self, name: str):
        """Save current state as a snapshot"""
        self.snapshots[name] = {
            'state': self.state_vector.copy(),
            'num_qubits': self.num_qubits,
            'circuit_history': self.circuit_history.copy(),
            'measurement_results': self.measurement_results.copy()
        }
        print(f"Snapshot '{name}' saved")
    
    def load_snapshot(self, name: str):
        """Load a previously saved snapshot"""
        if name not in self.snapshots:
            raise ValueError(f"Snapshot '{name}' not found")
        
        snapshot = self.snapshots[name]
        
        # Resize if necessary
        if snapshot['num_qubits'] != self.num_qubits:
            self.num_qubits = snapshot['num_qubits']
            self.num_states = 2 ** self.num_qubits
        
        self.state_vector = snapshot['state'].copy()
        self.circuit_history = snapshot['circuit_history'].copy()
        self.measurement_results = snapshot['measurement_results'].copy()
        
        print(f"Snapshot '{name}' loaded")
    
    def list_snapshots(self) -> List[str]:
        """List all saved snapshots"""
        return list(self.snapshots.keys())
    
    def delete_snapshot(self, name: str):
        """Delete a snapshot"""
        if name in self.snapshots:
            del self.snapshots[name]
            print(f"Snapshot '{name}' deleted")
        else:
            print(f"Snapshot '{name}' not found")
    
    def snapshots_info(self):
        """Show snapshots information"""
        if self.snapshots:
            print("Saved snapshots:")
            for name in self.snapshots.keys():
                print(f"  {name}")
        else:
            print("No snapshots saved")
    
    def bell(self, qubit1: int = 0, qubit2: int = 1):
        """Create Bell state"""
        self.create_bell_state(qubit1, qubit2)
    
    def ghz(self, *qubits):
        """Create GHZ state"""
        if qubits:
            self.create_ghz_state(list(qubits))
        else:
            self.create_ghz_state()
    
    def w_state(self, *qubits):
        """Create W state"""
        if qubits:
            self.create_w_state(list(qubits))
        else:
            self.create_w_state()
    
    # Circuit Drawing Methods
    def draw_circuit_ascii(self) -> str:
        """Generate ASCII representation of the quantum circuit."""
        if not self.circuit_history:
            return "Empty circuit"
        
        # Filter out measurements for circuit drawing
        operations = [op for op in self.circuit_history if op.get('type') != 'measurement']
        
        if not operations:
            return "No gates in circuit"
        
        # Initialize circuit lines
        lines = []
        for i in range(self.num_qubits):
            lines.append(f"q{i}: |0⟩")
        
        # Process each operation
        for op in operations:
            gate_name = op['gate']
            qubits = op['qubits']
            params = op.get('params', {})
            
            # Calculate gate representation
            if params:
                param_str = f"({','.join(f'{v:.2f}' for v in params.values())})"
                gate_str = f"{gate_name}{param_str}"
            else:
                gate_str = gate_name
            
            gate_width = max(len(gate_str), 3)
            
            # Ensure all lines are same length before adding gate
            max_len = max(len(line) for line in lines)
            for i in range(self.num_qubits):
                lines[i] = lines[i].ljust(max_len)
            
            if len(qubits) == 1:
                # Single-qubit gate
                qubit = qubits[0]
                lines[qubit] += f"─{gate_str.center(gate_width)}─"
                
                # Add padding to other qubits
                for i in range(self.num_qubits):
                    if i != qubit:
                        lines[i] += "─" + "─" * gate_width + "─"
            
            elif len(qubits) == 2:
                # Two-qubit gate
                control, target = qubits[0], qubits[1]
                min_q, max_q = min(control, target), max(control, target)
                
                # Add gate symbols
                for i in range(self.num_qubits):
                    if i == control:
                        if gate_name in ['CNOT', 'CX']:
                            lines[i] += "─●─" + "─" * (gate_width - 1)
                        elif gate_name == 'CZ':
                            lines[i] += "─●─" + "─" * (gate_width - 1)
                        else:
                            lines[i] += f"─{gate_str[:gate_width].center(gate_width)}─"
                    elif i == target:
                        if gate_name in ['CNOT', 'CX']:
                            lines[i] += "─⊕─" + "─" * (gate_width - 1)
                        elif gate_name == 'CZ':
                            lines[i] += "─●─" + "─" * (gate_width - 1)
                        elif gate_name == 'SWAP':
                            lines[i] += "─×─" + "─" * (gate_width - 1)
                        else:
                            lines[i] += f"─{gate_str[:gate_width].center(gate_width)}─"
                    elif min_q < i < max_q:
                        # Connection line between control and target
                        lines[i] += "─│─" + "─" * (gate_width - 1)
                    else:
                        # Other qubits get straight line
                        lines[i] += "─" + "─" * gate_width + "─"
            
            elif len(qubits) == 3:
                # Three-qubit gate (Toffoli)
                control1, control2, target = qubits[0], qubits[1], qubits[2]
                min_q = min(qubits)
                max_q = max(qubits)
                
                for i in range(self.num_qubits):
                    if i in [control1, control2]:
                        lines[i] += "─●─" + "─" * (gate_width - 1)
                    elif i == target:
                        lines[i] += "─⊕─" + "─" * (gate_width - 1)
                    elif min_q < i < max_q:
                        lines[i] += "─│─" + "─" * (gate_width - 1)
                    else:
                        lines[i] += "─" + "─" * gate_width + "─"
        
        return "\n".join(lines)
    
    def print_circuit_ascii(self):
        """Print ASCII circuit diagram to console."""
        print("\nCircuit Diagram:")
        print("=" * 50)
        print(self.draw_circuit_ascii())
        print("\nGate Legend:")
        print("● = Control qubit, ⊕ = CNOT target, × = SWAP, │ = Connection, ─ = Wire")
    
    def save_circuit_ascii(self, filename: str):
        """Save ASCII circuit diagram to file."""
        try:
            ascii_circuit = self.draw_circuit_ascii()
            with open(filename, 'w') as f:
                f.write("ASCII Circuit Diagram\n")
                f.write("=" * 50 + "\n\n")
                f.write(ascii_circuit)
                f.write("\n\n")
                f.write("Gate Legend:\n")
                f.write("● = Control qubit\n")
                f.write("⊕ = CNOT target\n")
                f.write("× = SWAP\n")
                f.write("│ = Connection\n")
                f.write("─ = Wire\n")
                f.write("\nCircuit Statistics:\n")
                f.write(f"Qubits: {self.num_qubits}\n")
                f.write(f"Operations: {len(self.circuit_history)}\n")
                f.write(f"Depth: {self.get_circuit_depth()}\n")
                
                gate_counts = self.get_gate_count()
                if gate_counts:
                    f.write(f"\nGate Counts:\n")
                    for gate, count in sorted(gate_counts.items()):
                        f.write(f"  {gate}: {count}\n")
            print(f"ASCII circuit diagram saved to {filename}")
        except Exception as e:
            print(f"Error saving ASCII diagram: {e}")
    
    # Qiskit Export Methods
    def save_circuit_qiskit(self, filename: str):
        """Save circuit as Qiskit Python code."""
        if not self.circuit_history:
            print("No circuit to save")
            return
        
        # Filter out measurements for circuit export
        operations = [op for op in self.circuit_history if op.get('type') != 'measurement']
        
        code_lines = []
        code_lines.append("# Generated Qiskit circuit")
        code_lines.append("from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile")
        code_lines.append("from qiskit_aer import Aer")
        code_lines.append("from qiskit.visualization import plot_histogram")
        code_lines.append("import numpy as np")
        code_lines.append("")
        code_lines.append(f"# Create quantum circuit with {self.num_qubits} qubits")
        code_lines.append(f"qreg = QuantumRegister({self.num_qubits}, 'q')")
        code_lines.append(f"creg = ClassicalRegister({self.num_qubits}, 'c')")
        code_lines.append("circuit = QuantumCircuit(qreg, creg)")
        code_lines.append("")
        code_lines.append("# Apply gates")
        
        for op in operations:
            qiskit_gate = self._map_gate_to_qiskit(op['gate'], op['qubits'], op.get('params', {}))
            code_lines.append(qiskit_gate)
        
        code_lines.append("")
        code_lines.append("# Add measurements")
        code_lines.append("circuit.measure_all()")
        code_lines.append("")
        code_lines.append("# Print circuit")
        code_lines.append("print('Quantum Circuit:')")
        code_lines.append("print(circuit)")
        code_lines.append("")
        code_lines.append("# Optional: Draw circuit")
        code_lines.append("# print(circuit.draw())")
        code_lines.append("")
        code_lines.append("# Execute circuit")
        code_lines.append("def run_circuit():")
        code_lines.append("    backend = Aer.get_backend('qasm_simulator')")
        code_lines.append("    shots = 1024")
        code_lines.append("    tqc =  transpile(circuit, backend)")
        code_lines.append("    job = backend.run(tqc, shots=shots)")
        code_lines.append("    result = job.result()")
        code_lines.append("    counts = result.get_counts()")
        code_lines.append("    print('Measurement results:')")
        code_lines.append("    for state, count in counts.items():")
        code_lines.append("        print(f'  |{state}⟩: {count}/{shots} ({count/shots:.3f})')")
        code_lines.append("    return counts")
        code_lines.append("")
        code_lines.append("# Uncomment to run the circuit")
        code_lines.append("# counts = run_circuit()")
        code_lines.append("# plot_histogram(counts)")
        
        # Write to file
        try:
            with open(filename, 'w') as f:
                f.write('\n'.join(code_lines))
            print(f"Circuit saved as Qiskit code to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
    
    def _map_gate_to_qiskit(self, gate_name: str, qubits: List[int], params: Dict) -> str:
        """Map simulator gate to Qiskit code."""
        gate_name = gate_name.upper()
        
        # Single-qubit gates
        if len(qubits) == 1:
            q = qubits[0]
            if gate_name == 'I':
                return f"circuit.id(qreg[{q}])"
            elif gate_name == 'X':
                return f"circuit.x(qreg[{q}])"
            elif gate_name == 'Y':
                return f"circuit.y(qreg[{q}])"
            elif gate_name == 'Z':
                return f"circuit.z(qreg[{q}])"
            elif gate_name == 'H':
                return f"circuit.h(qreg[{q}])"
            elif gate_name == 'S':
                return f"circuit.s(qreg[{q}])"
            elif gate_name == 'T':
                return f"circuit.t(qreg[{q}])"
            elif gate_name in ['S_DAGGER', 'SDAGGER']:
                return f"circuit.sdg(qreg[{q}])"
            elif gate_name in ['T_DAGGER', 'TDAGGER']:
                return f"circuit.tdg(qreg[{q}])"
            elif gate_name == 'RX':
                theta = params.get('theta', 0)
                return f"circuit.rx({theta}, qreg[{q}])"
            elif gate_name == 'RY':
                theta = params.get('theta', 0)
                return f"circuit.ry({theta}, qreg[{q}])"
            elif gate_name == 'RZ':
                theta = params.get('theta', 0)
                return f"circuit.rz({theta}, qreg[{q}])"
            elif gate_name in ['PHASE', 'P']:
                phi = params.get('phi', 0)
                return f"circuit.p({phi}, qreg[{q}])"
            elif gate_name == 'U1':
                lam = params.get('lambda', 0)
                return f"circuit.p({lam}, qreg[{q}])  # U1 equivalent"
            elif gate_name == 'U2':
                phi = params.get('phi', 0)
                lam = params.get('lambda', 0)
                return f"circuit.u2({phi}, {lam}, qreg[{q}])"
            elif gate_name in ['U3', 'U']:
                theta = params.get('theta', 0)
                phi = params.get('phi', 0)
                lam = params.get('lambda', 0)
                return f"circuit.u3({theta}, {phi}, {lam}, qreg[{q}])"
        
        # Two-qubit gates
        elif len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            if gate_name in ['CNOT', 'CX']:
                return f"circuit.cx(qreg[{q1}], qreg[{q2}])"
            elif gate_name == 'CZ':
                return f"circuit.cz(qreg[{q1}], qreg[{q2}])"
            elif gate_name == 'CY':
                return f"circuit.cy(qreg[{q1}], qreg[{q2}])"
            elif gate_name == 'SWAP':
                return f"circuit.swap(qreg[{q1}], qreg[{q2}])"
            elif gate_name == 'ISWAP':
                return f"circuit.iswap(qreg[{q1}], qreg[{q2}])"
            elif gate_name == 'CRX':
                theta = params.get('theta', 0)
                return f"circuit.crx({theta}, qreg[{q1}], qreg[{q2}])"
            elif gate_name == 'CRY':
                theta = params.get('theta', 0)
                return f"circuit.cry({theta}, qreg[{q1}], qreg[{q2}])"
            elif gate_name == 'CRZ':
                theta = params.get('theta', 0)
                return f"circuit.crz({theta}, qreg[{q1}], qreg[{q2}])"
            elif gate_name in ['CPHASE', 'CP']:
                phi = params.get('phi', 0)
                return f"circuit.cp({phi}, qreg[{q1}], qreg[{q2}])"
        
        # Three-qubit gates
        elif len(qubits) == 3:
            q1, q2, q3 = qubits[0], qubits[1], qubits[2]
            if gate_name in ['TOFFOLI', 'CCX']:
                return f"circuit.ccx(qreg[{q1}], qreg[{q2}], qreg[{q3}])"
            elif gate_name in ['FREDKIN', 'CSWAP']:
                return f"circuit.cswap(qreg[{q1}], qreg[{q2}], qreg[{q3}])"
        
        return f"# Unsupported gate: {gate_name}"
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get comprehensive circuit information."""
        operations = [op for op in self.circuit_history if op.get('type') != 'measurement']
        measurements = [op for op in self.circuit_history if op.get('type') == 'measurement']
        
        return {
            'total_operations': len(self.circuit_history),
            'gate_operations': len(operations),
            'measurements': len(measurements),
            'circuit_depth': len(operations),
            'circuit_width': self.num_qubits,
            'gate_counts': self.get_gate_count(),
            'qubit_usage': self._get_qubit_usage(),
            'has_entangling_gates': self._has_entangling_gates()
        }
    
    def _get_qubit_usage(self) -> Dict[int, int]:
        """Get usage count for each qubit."""
        usage = {i: 0 for i in range(self.num_qubits)}
        for op in self.circuit_history:
            if op.get('type') != 'measurement':
                for qubit in op.get('qubits', []):
                    usage[qubit] += 1
        return usage
    
    def _has_entangling_gates(self) -> bool:
        """Check if circuit contains entangling gates."""
        entangling_gates = ['CNOT', 'CX', 'CZ', 'CY', 'TOFFOLI', 'CCX', 'FREDKIN', 'CSWAP']
        for op in self.circuit_history:
            if op.get('gate') in entangling_gates:
                return True
        return False
    
    def apply_gate(self, gate_name: str, qubits: Union[int, List[int]], parameters: Optional[List[float]] = None):
        """Apply a quantum gate - dispatcher method for REPL interface"""
        if isinstance(qubits, int):
            qubits = [qubits]
        
        # Validate qubits
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.num_qubits-1}]")
        
        # Convert parameters to kwargs if provided
        kwargs = {}
        if parameters:
            if gate_name.upper() in ['RX', 'RY', 'RZ']:
                kwargs['theta'] = parameters[0]
            elif gate_name.upper() in ['PHASE', 'P']:
                kwargs['phi'] = parameters[0]
            elif gate_name.upper() == 'U1':
                kwargs['lambda'] = parameters[0]
            elif gate_name.upper() == 'U2':
                kwargs['phi'] = parameters[0]
                if len(parameters) > 1:
                    kwargs['lambda'] = parameters[1]
            elif gate_name.upper() in ['U3', 'U']:
                kwargs['theta'] = parameters[0]
                if len(parameters) > 1:
                    kwargs['phi'] = parameters[1]
                if len(parameters) > 2:
                    kwargs['lambda'] = parameters[2]
            elif gate_name.upper() in ['CRX', 'CRY', 'CRZ']:
                kwargs['theta'] = parameters[0]
            elif gate_name.upper() in ['CPHASE', 'CP']:
                kwargs['phi'] = parameters[0]
            elif gate_name.upper() == 'CU':
                kwargs['theta'] = parameters[0]
                if len(parameters) > 1:
                    kwargs['phi'] = parameters[1]
                if len(parameters) > 2:
                    kwargs['lambda'] = parameters[2]
                if len(parameters) > 3:
                    kwargs['gamma'] = parameters[3]
        
        # Route to appropriate method based on number of qubits
        if len(qubits) == 1:
            self.apply_single_qubit_gate(gate_name, qubits[0], **kwargs)
        elif len(qubits) == 2:
            self.apply_two_qubit_gate(gate_name, qubits[0], qubits[1], **kwargs)
        elif len(qubits) == 3:
            self.apply_three_qubit_gate(gate_name, qubits[0], qubits[1], qubits[2])
        else:
            raise ValueError(f"Gates with {len(qubits)} qubits not supported")
    
    def apply_single_qubit_gate(self, gate_name: str, qubit: int, **kwargs):
        """Apply a single-qubit gate to the specified qubit"""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
        
        gate_matrix = self.gates.get_single_qubit_gate(gate_name, **kwargs)
        if gate_matrix is None:
            raise ValueError(f"Unknown gate: {gate_name}")
        
        # Apply single-qubit gate using tensor reshaping
        self._apply_single_qubit_unitary(gate_matrix, qubit)
        
        # Record in circuit history
        self.circuit_history.append({
            'type': 'single_qubit',
            'gate': gate_name,
            'qubits': [qubit],
            'params': kwargs
        })
        
        params_str = ""
        if kwargs:
            params_str = f"({', '.join(f'{v:.3f}' for v in kwargs.values())})"
        print(f"Applied {gate_name}{params_str} q[{qubit}]")
    
    def apply_two_qubit_gate(self, gate_name: str, control: int, target: int, **kwargs):
        """Apply a two-qubit gate"""
        if not 0 <= control < self.num_qubits:
            raise ValueError(f"Control qubit {control} out of range [0, {self.num_qubits-1}]")
        if not 0 <= target < self.num_qubits:
            raise ValueError(f"Target qubit {target} out of range [0, {self.num_qubits-1}]")
        if control == target:
            raise ValueError("Control and target qubits must be different")
        
        gate_matrix = self.gates.get_two_qubit_gate(gate_name, **kwargs)
        if gate_matrix is None:
            raise ValueError(f"Unknown two-qubit gate: {gate_name}")
        
        # Apply the two-qubit gate
        self._apply_two_qubit_unitary(gate_matrix, control, target)
        
        # Record in circuit history
        self.circuit_history.append({
            'type': 'two_qubit',
            'gate': gate_name,
            'qubits': [control, target],
            'params': kwargs
        })
        
        params_str = ""
        if kwargs:
            params_str = f"({', '.join(f'{v:.3f}' for v in kwargs.values())})"
        print(f"Applied {gate_name}{params_str} q[{control}, {target}]")
    
    def apply_three_qubit_gate(self, gate_name: str, control1: int, control2: int, target: int):
        """Apply a three-qubit gate (like Toffoli)"""
        qubits = [control1, control2, target]
        if len(set(qubits)) != 3:
            raise ValueError("All three qubits must be different")
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"Qubit {q} out of range [0, {self.num_qubits-1}]")
        
        gate_matrix = self.gates.get_three_qubit_gate(gate_name)
        if gate_matrix is None:
            raise ValueError(f"Unknown three-qubit gate: {gate_name}")
        
        # Apply the three-qubit gate
        self._apply_three_qubit_unitary(gate_matrix, control1, control2, target)
        
        # Record in circuit history
        self.circuit_history.append({
            'type': 'three_qubit',
            'gate': gate_name,
            'qubits': [control1, control2, target],
            'params': {}
        })
        
        print(f"Applied {gate_name} q[{control1}, {control2}, {target}]")
    
    def _apply_single_qubit_unitary(self, unitary: np.ndarray, qubit: int):
        """Apply single-qubit unitary using tensor operations"""
        # Reshape state vector to tensor form
        shape = [2] * self.num_qubits
        tensor = self.state_vector.reshape(shape)
        
        # Apply unitary to specified qubit axis
        tensor = np.tensordot(unitary, tensor, axes=([1], [qubit]))
        
        # Move the result back to the correct position
        tensor = np.moveaxis(tensor, 0, qubit)
        
        # Flatten back to state vector
        self.state_vector = tensor.flatten()
    
    def _apply_two_qubit_unitary(self, unitary: np.ndarray, control: int, target: int):
        """Apply two-qubit unitary using direct state manipulation"""
        new_state = np.zeros_like(self.state_vector)
        
        for state_idx in range(self.num_states):
            # Extract control and target bits
            control_bit = (state_idx >> (self.num_qubits - 1 - control)) & 1
            target_bit = (state_idx >> (self.num_qubits - 1 - target)) & 1
            
            # Map to 2-qubit subspace (control comes first in gate matrix)
            two_qubit_input = control_bit * 2 + target_bit
            
            # Apply unitary in 2-qubit subspace
            for two_qubit_output in range(4):
                new_control_bit = two_qubit_output >> 1
                new_target_bit = two_qubit_output & 1
                
                # Compute new state index
                new_state_idx = state_idx
                # Clear the control and target bits
                new_state_idx &= ~(1 << (self.num_qubits - 1 - control))
                new_state_idx &= ~(1 << (self.num_qubits - 1 - target))
                # Set new control and target bits
                new_state_idx |= new_control_bit << (self.num_qubits - 1 - control)
                new_state_idx |= new_target_bit << (self.num_qubits - 1 - target)
                
                # Add contribution from unitary matrix
                new_state[new_state_idx] += unitary[two_qubit_output, two_qubit_input] * self.state_vector[state_idx]
        
        self.state_vector = new_state
    
    def _apply_three_qubit_unitary(self, unitary: np.ndarray, control1: int, control2: int, target: int):
        """Apply three-qubit unitary using direct state manipulation"""
        new_state = np.zeros_like(self.state_vector)
        
        for state_idx in range(self.num_states):
            # Extract qubit bits
            c1_bit = (state_idx >> (self.num_qubits - 1 - control1)) & 1
            c2_bit = (state_idx >> (self.num_qubits - 1 - control2)) & 1
            t_bit = (state_idx >> (self.num_qubits - 1 - target)) & 1
            
            # Map to 3-qubit subspace
            three_qubit_input = c1_bit * 4 + c2_bit * 2 + t_bit
            
            # Apply unitary in 3-qubit subspace
            for three_qubit_output in range(8):
                new_c1_bit = (three_qubit_output >> 2) & 1
                new_c2_bit = (three_qubit_output >> 1) & 1
                new_t_bit = three_qubit_output & 1
                
                # Compute new state index
                new_state_idx = state_idx
                # Clear the bits
                new_state_idx &= ~(1 << (self.num_qubits - 1 - control1))
                new_state_idx &= ~(1 << (self.num_qubits - 1 - control2))
                new_state_idx &= ~(1 << (self.num_qubits - 1 - target))
                # Set new bits
                new_state_idx |= new_c1_bit << (self.num_qubits - 1 - control1)
                new_state_idx |= new_c2_bit << (self.num_qubits - 1 - control2)
                new_state_idx |= new_t_bit << (self.num_qubits - 1 - target)
                
                # Add contribution
                new_state[new_state_idx] += unitary[three_qubit_output, three_qubit_input] * self.state_vector[state_idx]
        
        self.state_vector = new_state
    
    def measure_qubit(self, qubit: int) -> int:
        """Measure a single qubit and collapse the state"""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
        
        # Calculate probabilities for this qubit being 0 or 1
        prob_0 = 0.0
        prob_1 = 0.0
        
        for state in range(self.num_states):
            qubit_value = (state >> (self.num_qubits - 1 - qubit)) & 1
            prob = abs(self.state_vector[state]) ** 2
            if qubit_value == 0:
                prob_0 += prob
            else:
                prob_1 += prob
        
        # Random measurement based on probabilities
        result = 1 if np.random.random() < prob_1 else 0
        
        # Collapse the state
        new_state = np.zeros_like(self.state_vector)
        norm = 0.0
        
        for state in range(self.num_states):
            qubit_value = (state >> (self.num_qubits - 1 - qubit)) & 1
            if qubit_value == result:
                new_state[state] = self.state_vector[state]
                norm += abs(self.state_vector[state]) ** 2
        
        if norm > 0:
            new_state /= np.sqrt(norm)
        
        self.state_vector = new_state
        
        # Record measurement
        self.measurement_results.append({'qubit': qubit, 'result': result})
        self.circuit_history.append({
            'type': 'measurement',
            'gate': 'MEASURE',
            'qubits': [qubit],
            'result': result
        })
        
        print(f"Measured qubit {qubit}: {result} (probability: {prob_1 if result else prob_0:.4f})")
        return result
    
    def measure_all(self) -> List[int]:
        """Measure all qubits"""
        results = []
        for i in range(self.num_qubits):
            results.append(self.measure_qubit(i))
        return results
    
    def get_bloch_vector(self, qubit: int) -> Tuple[float, float, float]:
        """Get Bloch sphere coordinates for a single qubit"""
        if not 0 <= qubit < self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range")
        
        # Compute reduced density matrix for the qubit
        rho = self._compute_single_qubit_density_matrix(qubit)
        
        # Extract Bloch vector components
        x = 2 * np.real(rho[0, 1])
        y = -2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])
        
        return (x, y, z)
    
    def _compute_single_qubit_density_matrix(self, qubit: int) -> np.ndarray:
        """Compute reduced density matrix for a single qubit"""
        rho = np.zeros((2, 2), dtype=complex)
        
        for i in range(2):
            for j in range(2):
                for state in range(self.num_states):
                    # Check if qubit has value i
                    if ((state >> (self.num_qubits - 1 - qubit)) & 1) == i:
                        for state2 in range(self.num_states):
                            # Check if qubit has value j and other qubits match
                            if ((state2 >> (self.num_qubits - 1 - qubit)) & 1) == j:
                                # Check if other qubits are the same
                                mask = ~(1 << (self.num_qubits - 1 - qubit))
                                if (state & mask) == (state2 & mask):
                                    rho[i, j] += self.state_vector[state] * np.conj(self.state_vector[state2])
        
        return rho
    
    def print_state(self, show_phases: bool = True, threshold: float = 1e-10):
        """Print the current quantum state"""
        print(f"\nQuantum State ({self.num_qubits} qubits):")
        print("=" * 50)
        
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > threshold:
                # Convert state index to binary string
                binary_state = format(i, f'0{self.num_qubits}b')
                
                # Format amplitude
                if show_phases:
                    mag = abs(amplitude)
                    phase = cmath.phase(amplitude)
                    if abs(phase) < threshold:
                        amp_str = f"{mag:.6f}"
                    else:
                        amp_str = f"{mag:.6f} * e^(i*{phase:.4f})"
                else:
                    amp_str = f"{abs(amplitude):.6f}"
                
                probability = abs(amplitude) ** 2
                print(f"|{binary_state}⟩: {amp_str} (prob: {probability:.6f})")
        
        # Print measurement probabilities by qubit
        print("\nQubit Measurement Probabilities:")
        for q in range(self.num_qubits):
            prob_0 = sum(abs(self.state_vector[i])**2 
                        for i in range(self.num_states) 
                        if ((i >> (self.num_qubits - 1 - q)) & 1) == 0)
            prob_1 = 1.0 - prob_0
            print(f"Qubit {q}: |0⟩ = {prob_0:.6f}, |1⟩ = {prob_1:.6f}")
    
    def get_circuit_depth(self) -> int:
        """Calculate the depth of the current circuit"""
        return len(self.circuit_history)
    
    def get_gate_count(self) -> Dict[str, int]:
        """Get count of each gate type used"""
        counts = {}
        for op in self.circuit_history:
            gate = op['gate']
            counts[gate] = counts.get(gate, 0) + 1
        return counts
    
    def create_bell_state(self, qubit1: int = 0, qubit2: int = 1):
        """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
        if qubit1 >= self.num_qubits or qubit2 >= self.num_qubits:
            raise ValueError("Qubit indices out of range")
        if qubit1 == qubit2:
            raise ValueError("Need two different qubits")
        
        # Reset and create Bell state
        self.reset()
        self.apply_gate('H', [qubit1])
        self.apply_gate('CNOT', [qubit1, qubit2])
        print(f"Created Bell state on qubits {qubit1} and {qubit2}")
    
    def create_ghz_state(self, qubits: Optional[List[int]] = None):
        """Create GHZ state (|000...⟩ + |111...⟩)/√2."""
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        if len(qubits) < 2:
            raise ValueError("Need at least 2 qubits for GHZ state")
        
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"Qubit {q} out of range")
        
        # Reset and create GHZ state
        self.reset()
        self.apply_gate('H', [qubits[0]])
        for i in range(1, len(qubits)):
            self.apply_gate('CNOT', [qubits[0], qubits[i]])
        
        print(f"Created GHZ state on qubits {qubits}")
    
    def create_w_state(self, qubits: Optional[List[int]] = None):
        """Create W state - equal superposition of all single-excitation states."""
        if qubits is None:
            qubits = list(range(self.num_qubits))
        
        n = len(qubits)
        if n < 2:
            raise ValueError("Need at least 2 qubits for W state")
        
        # W state requires specific amplitude for each basis state
        state_vector = np.zeros(2**self.num_qubits, dtype=complex)
        
        # Add equal amplitude for each single-excitation state
        for i, q in enumerate(qubits):
            state_idx = 1 << (self.num_qubits - 1 - q)  # Single excitation on qubit q
            state_vector[state_idx] = 1.0 / np.sqrt(n)
        
        self.state_vector = state_vector
        print(f"Created W state on qubits {qubits}")
    
    def get_entanglement_info(self) -> Dict[str, Any]:
        """Get entanglement information."""
        # Calculate von Neumann entropy for each qubit
        entropies = {}
        for q in range(self.num_qubits):
            try:
                rho = self._compute_single_qubit_density_matrix(q)
                eigenvals = np.linalg.eigvals(rho).real
                eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zero eigenvalues
                if len(eigenvals) > 0:
                    # Ensure positive values and handle log(0)
                    eigenvals = eigenvals / np.sum(eigenvals)  # Normalize
                    entropy = -np.sum(eigenvals * np.log2(np.maximum(eigenvals, 1e-12)))
                    entropy = max(0.0, entropy)  # Ensure non-negative
                else:
                    entropy = 0.0
                entropies[f'qubit_{q}'] = entropy
            except:
                entropies[f'qubit_{q}'] = 0.0
        
        # Overall entanglement measure
        max_entropy = max(entropies.values()) if entropies else 0.0
        
        return {
            'qubit_entropies': entropies,
            'max_entropy': max_entropy,
            'is_entangled': max_entropy > 0.01,
            'entanglement_level': 'high' if max_entropy > 0.9 else 'medium' if max_entropy > 0.1 else 'low'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            'num_qubits': self.num_qubits,
            'total_operations': len(self.circuit_history),
            'circuit_depth': self.get_circuit_depth(),
            'circuit_width': self.num_qubits,
            'gate_counts': self.get_gate_count(),
            'measurements': len(self.measurement_results),
            'state_norm': np.linalg.norm(self.state_vector),
            'memory_usage_mb': self.state_vector.nbytes / (1024 * 1024)
        }
    
    def __str__(self) -> str:
        """String representation of simulator."""
        stats = self.get_statistics()
        lines = [
            f"Quantum Simulator ({self.num_qubits} qubits)",
            f"Operations: {stats['total_operations']}",
            f"Memory usage: {stats['memory_usage_mb']:.2f} MB",
            "",
            "Current state:"
        ]
        
        # Show non-zero amplitudes
        for i, amplitude in enumerate(self.state_vector):
            if abs(amplitude) > 1e-10:
                binary = format(i, f'0{self.num_qubits}b')
                prob = abs(amplitude) ** 2
                lines.append(f"  |{binary}⟩: {amplitude:.6f} (prob: {prob:.6f})")
        
        return '\n'.join(lines)
