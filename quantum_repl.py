#-------------------------
#
# Quantum Simulator
# (c) 2025, Michael Stal 
#
#-------------------------

"""
Quantum REPL Interface
Interactive command-line interface for the quantum simulator
"""

import sys
import os
import traceback
import argparse
import shlex
from typing import List, Dict, Optional, Any, Callable
import numpy as np
import math
from quantum_simulator import QuantumSimulator


class QuantumREPL:
    """Interactive REPL for quantum simulation."""
    
    def __init__(self, num_qubits: int = 3):
        """Initialize REPL with quantum simulator."""
        self.simulator = QuantumSimulator(num_qubits)
        self.loaded_files = []
        self.script_commands = []
        self.running = True
        
        # Command registry
        self.commands = {
            'help': self.cmd_help,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'bye': self.cmd_exit,
            'reset': self.cmd_reset,
            'state': self.cmd_state,
            'qubits': self.cmd_qubits,
            'resize': self.cmd_resize,
            
            # Gate operations
            'gate': self.cmd_gate,
            'x': lambda args: self.cmd_gate(['X'] + args),
            'y': lambda args: self.cmd_gate(['Y'] + args),
            'z': lambda args: self.cmd_gate(['Z'] + args),
            'h': lambda args: self.cmd_gate(['H'] + args),
            's': lambda args: self.cmd_gate(['S'] + args),
            't': lambda args: self.cmd_gate(['T'] + args),
            'cnot': lambda args: self.cmd_gate(['CNOT'] + args),
            'cx': lambda args: self.cmd_gate(['CX'] + args),
            'cz': lambda args: self.cmd_gate(['CZ'] + args),
            'swap': lambda args: self.cmd_gate(['SWAP'] + args),
            'toffoli': lambda args: self.cmd_gate(['TOFFOLI'] + args),
            'ccx': lambda args: self.cmd_gate(['CCX'] + args),
            'rx': self.cmd_rx,
            'ry': self.cmd_ry,
            'rz': self.cmd_rz,
            'u': self.cmd_u,
            
            # Measurements
            'measure': self.cmd_measure,
            'measure_all': self.cmd_measure_all,
            
            # Circuit operations
            'circuit': self.cmd_circuit,
            'print_circuit': self.cmd_print_circuit,
            'draw': self.cmd_draw,
            'draw_ascii': self.cmd_draw_ascii,
            'save_qiskit': self.cmd_save_qiskit,
            'save_ascii': self.cmd_save_ascii,
            'depth': self.cmd_depth,
            'width': self.cmd_width,
            'count': self.cmd_count,
            
            # Special states
            'bell': self.cmd_bell,
            'ghz': self.cmd_ghz,
            'w_state': self.cmd_w_state,
            
            # Advanced features
            'probabilities': self.cmd_probabilities,
            'amplitude': self.cmd_amplitude,
            'bloch': self.cmd_bloch,
            'entanglement': self.cmd_entanglement,
            'stats': self.cmd_stats,
            
            # Snapshots
            'snapshot': self.cmd_snapshot,
            'save_snapshot': self.cmd_snapshot,  # alias
            'snapshots': self.cmd_snapshots,
            'load_snapshot': self.cmd_load_snapshot,
            'delete_snapshot': self.cmd_delete_snapshot,
            
            # File operations
            'load': self.cmd_load,
            'run': self.cmd_run,
            'files': self.cmd_files,
            
            # Information
            'gates': self.cmd_gates,
            'gate_info': self.cmd_gate_info,
            'version': self.cmd_version
        }
    
    def run(self):
        """Main REPL loop."""
        print("Quantum Simulator REPL v1.0")
        print(f"Initialized with {self.simulator.num_qubits} qubits")
        print("Type 'help' for available commands, 'exit' to quit")
        print()
        
        while self.running:
            try:
                # Get input
                prompt = f"quantum[{self.simulator.num_qubits}]> "
                line = input(prompt).strip()
                
                if not line:
                    continue
                
                # Parse and execute command
                self.execute_command(line)
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                if "--debug" in sys.argv:
                    traceback.print_exc()
    
    def execute_command(self, line: str):
        """Parse and execute a command line."""
        try:
            # Remove comments (everything after #)
            if '#' in line:
                line = line[:line.index('#')].strip()
            
            # Skip empty lines after comment removal
            if not line:
                return
            
            # Split command line
            parts = shlex.split(line)
            if not parts:
                return
            
            command = parts[0].lower()
            args = parts[1:]
            
            # Execute command
            if command in self.commands:
                self.commands[command](args)
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
        
        except Exception as e:
            print(f"Command error: {e}")
            if "--debug" in sys.argv:
                traceback.print_exc()
    
    def cmd_help(self, args: List[str]):
        """Show help information."""
        if not args:
            print("Quantum Simulator Commands:")
            print()
            print("Basic Operations:")
            print("  help [command]     - Show help (or help for specific command)")
            print("  exit/quit/bye      - Exit the simulator")
            print("  reset              - Reset quantum state to |0...0⟩")
            print("  state              - Show current quantum state")
            print("  qubits             - Show number of qubits")
            print("  resize <n>         - Resize to n qubits")
            print()
            print("Quantum Gates:")
            print("  gate <name> <qubits> [params] - Apply quantum gate")
            print("  x/y/z/h/s/t <qubit>          - Apply single-qubit gates")
            print("  cnot/cx <ctrl> <target>      - Apply CNOT gate")
            print("  cz <q1> <q2>                 - Apply controlled-Z gate")
            print("  swap <q1> <q2>               - Apply SWAP gate")
            print("  toffoli <c1> <c2> <target>   - Apply Toffoli gate")
            print("  rx/ry/rz <angle> <qubit>     - Apply rotation gates")
            print("  u <theta> <phi> <lambda> <q> - Apply universal gate")
            print()
            print("Measurements:")
            print("  measure <qubit>    - Measure single qubit")
            print("  measure_all        - Measure all qubits")
            print()
            print("Circuit Operations:")
            print("  circuit            - Show current circuit info")
            print("  draw               - Draw circuit in ASCII")
            print("  save_qiskit <file> - Save circuit as Qiskit code")
            print("  save_ascii <file>  - Save ASCII circuit diagram")
            print("  depth              - Show circuit depth")
            print("  count              - Count gates by type")
            print()
            print("Analysis:")
            print("  probabilities      - Show measurement probabilities")
            print("  amplitude <state>  - Show amplitude of basis state")
            print("  bloch <qubit>      - Show Bloch vector")
            print("  entanglement       - Show entanglement information")
            print("  stats              - Show simulation statistics")
            print()
            print("Special States:")
            print("  bell [q1] [q2]     - Create Bell state")
            print("  ghz [qubits...]    - Create GHZ state")
            print("  w_state [qubits...]- Create W state")
            print()
            print("Snapshots:")
            print("  snapshot <name>    - Save current state snapshot")
            print("  snapshots          - List snapshots")
            print("  load_snapshot <name> - Load snapshot")
            print("  delete_snapshot <name> - Delete snapshot")
            print()
            print("File Operations:")
            print("  load <file>        - Load quantum script")
            print("  run                - Run loaded script")
            print()
            print("Information:")
            print("  gates              - List available gates")
            print("  version            - Show version information")
        else:
            # Help for specific command
            command = args[0].lower()
            if command in self.commands:
                func = self.commands[command]
                if func.__doc__:
                    print(f"{command}: {func.__doc__}")
                else:
                    print(f"Command: {command}")
            else:
                print(f"Unknown command: {command}")
    
    def cmd_exit(self, args: List[str]):
        """Exit the simulator."""
        print("Goodbye!")
        self.running = False
    
    def cmd_reset(self, args: List[str]):
        """Reset quantum state to |0...0⟩."""
        self.simulator.reset()
    
    def cmd_state(self, args: List[str]):
        """Show current quantum state."""
        self.simulator.state()
    
    def cmd_qubits(self, args: List[str]):
        """Show number of qubits."""
        print(f"Number of qubits: {self.simulator.num_qubits}")
    
    def cmd_resize(self, args: List[str]):
        """Resize to n qubits."""
        if not args:
            print("Usage: resize <num_qubits>")
            return
        
        try:
            num_qubits = int(args[0])
            self.simulator.resize(num_qubits)
        except ValueError:
            print("Invalid number of qubits")
        except Exception as e:
            print(f"Error resizing: {e}")
    
    def cmd_gate(self, args: List[str]):
        """Apply quantum gate."""
        if len(args) < 2:
            print("Usage: gate <gate_name> <qubit1> [qubit2] [params...]")
            return
        
        try:
            gate_name = args[0].upper()
            
            # Parse qubits and parameters
            qubits = []
            params = []
            
            for i, arg in enumerate(args[1:], 1):
                try:
                    val = float(arg)
                    # Check if this looks like a qubit index or parameter
                    if '.' in arg or val >= self.simulator.num_qubits:
                        # Likely a parameter
                        params = [float(x) for x in args[i:]]
                        break
                    else:
                        qubits.append(int(val))
                except ValueError:
                    print(f"Invalid argument: {arg}")
                    return
            
            # Apply gate
            self.simulator.apply_gate(gate_name, qubits, params if params else None)
            
        except Exception as e:
            print(f"Error applying gate: {e}")
    
    def cmd_rx(self, args: List[str]):
        """Apply RX rotation gate."""
        if len(args) != 2:
            print("Usage: rx <angle> <qubit>")
            return
        
        try:
            angle = float(args[0])
            qubit = int(args[1])
            self.simulator.apply_gate('RX', [qubit], [angle])
        except ValueError:
            print("Invalid angle or qubit number")
    
    def cmd_ry(self, args: List[str]):
        """Apply RY rotation gate."""
        if len(args) != 2:
            print("Usage: ry <angle> <qubit>")
            return
        
        try:
            angle = float(args[0])
            qubit = int(args[1])
            self.simulator.apply_gate('RY', [qubit], [angle])
        except ValueError:
            print("Invalid angle or qubit number")
    
    def cmd_rz(self, args: List[str]):
        """Apply RZ rotation gate."""
        if len(args) != 2:
            print("Usage: rz <angle> <qubit>")
            return
        
        try:
            angle = float(args[0])
            qubit = int(args[1])
            self.simulator.apply_gate('RZ', [qubit], [angle])
        except ValueError:
            print("Invalid angle or qubit number")
    
    def cmd_u(self, args: List[str]):
        """Apply universal single-qubit gate."""
        if len(args) != 4:
            print("Usage: u <theta> <phi> <lambda> <qubit>")
            return
        
        try:
            theta = float(args[0])
            phi = float(args[1])
            lam = float(args[2])
            qubit = int(args[3])
            self.simulator.apply_gate('U', [qubit], [theta, phi, lam])
        except ValueError:
            print("Invalid parameters or qubit number")
    
    def cmd_measure(self, args: List[str]):
        """Measure single qubit."""
        if len(args) != 1:
            print("Usage: measure <qubit>")
            return
        
        try:
            qubit = int(args[0])
            result = self.simulator.measure_qubit(qubit)
            return result
        except ValueError:
            print("Invalid qubit number")
    
    def cmd_measure_all(self, args: List[str]):
        """Measure all qubits."""
        results = self.simulator.measure_all()
        return results
    
    def cmd_circuit(self, args: List[str]):
        """Show current circuit information."""
        info = self.simulator.get_circuit_info()
        print(f"Circuit Information:")
        print(f"  Total operations: {info['total_operations']}")
        print(f"  Gate operations: {info['gate_operations']}")
        print(f"  Measurements: {info['measurements']}")
        print(f"  Circuit depth: {info['circuit_depth']}")
        print(f"  Circuit width: {info['circuit_width']}")
        print(f"  Has entangling gates: {info['has_entangling_gates']}")
        
        if info['gate_counts']:
            print("  Gate counts:")
            for gate, count in sorted(info['gate_counts'].items()):
                print(f"    {gate}: {count}")
        
        print("  Qubit usage:")
        for qubit, count in info['qubit_usage'].items():
            print(f"    q{qubit}: {count} operations")
    
    def cmd_print_circuit(self, args: List[str]):
        """Print detailed circuit information."""
        self.cmd_circuit(args)
        print()
        self.simulator.print_circuit_ascii()
    
    def cmd_draw(self, args: List[str]):
        """Draw circuit in ASCII."""
        self.simulator.print_circuit_ascii()
    
    def cmd_draw_ascii(self, args: List[str]):
        """Draw circuit in ASCII format."""
        self.cmd_draw(args)
    
    def cmd_save_qiskit(self, args: List[str]):
        """Save circuit as Qiskit code."""
        if not args:
            print("Usage: save_qiskit <filename>")
            return
        
        filename = args[0]
        if not filename.endswith('.py'):
            filename += '.py'
        
        try:
            self.simulator.save_circuit_qiskit(filename)
        except Exception as e:
            print(f"Error saving Qiskit code: {e}")
    
    def cmd_save_ascii(self, args: List[str]):
        """Save ASCII circuit diagram."""
        if not args:
            print("Usage: save_ascii <filename>")
            return
        
        filename = args[0]
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        try:
            self.simulator.save_circuit_ascii(filename)
        except Exception as e:
            print(f"Error saving ASCII diagram: {e}")
    
    def cmd_depth(self, args: List[str]):
        """Show circuit depth."""
        depth = self.simulator.get_circuit_depth()
        print(f"Circuit depth: {depth}")
    
    def cmd_width(self, args: List[str]):
        """Show circuit width."""
        print(f"Circuit width: {self.simulator.num_qubits}")
    
    def cmd_count(self, args: List[str]):
        """Count gates by type."""
        counts = self.simulator.get_gate_count()
        if counts:
            print("Gate counts:")
            for gate, count in sorted(counts.items()):
                print(f"  {gate}: {count}")
        else:
            print("No gates in circuit")
    
    def cmd_bell(self, args: List[str]):
        """Create Bell state."""
        if len(args) == 0:
            self.simulator.bell()
        elif len(args) == 2:
            try:
                q1, q2 = int(args[0]), int(args[1])
                self.simulator.bell(q1, q2)
            except ValueError:
                print("Invalid qubit numbers")
        else:
            print("Usage: bell [qubit1] [qubit2]")
    
    def cmd_ghz(self, args: List[str]):
        """Create GHZ state."""
        if not args:
            self.simulator.ghz()
        else:
            try:
                qubits = [int(q) for q in args]
                self.simulator.ghz(*qubits)
            except ValueError:
                print("Invalid qubit numbers")
    
    def cmd_w_state(self, args: List[str]):
        """Create W state."""
        if not args:
            self.simulator.w_state()
        else:
            try:
                qubits = [int(q) for q in args]
                self.simulator.w_state(*qubits)
            except ValueError:
                print("Invalid qubit numbers")
    
    def cmd_probabilities(self, args: List[str]):
        """Show measurement probabilities."""
        self.simulator.probabilities()
    
    def cmd_amplitude(self, args: List[str]):
        """Show amplitude of basis state."""
        if not args:
            print("Usage: amplitude <state_index>")
            return
        
        try:
            state_idx = int(args[0])
            self.simulator.amplitude(state_idx)
        except ValueError:
            print("Invalid state index")
        except Exception as e:
            print(f"Error: {e}")
    
    def cmd_bloch(self, args: List[str]):
        """Show Bloch vector."""
        if not args:
            print("Usage: bloch <qubit>")
            return
        
        try:
            qubit = int(args[0])
            self.simulator.bloch(qubit)
        except ValueError:
            print("Invalid qubit number")
        except Exception as e:
            print(f"Error: {e}")
    
    def cmd_entanglement(self, args: List[str]):
        """Show entanglement information."""
        self.simulator.entanglement()
    
    def cmd_stats(self, args: List[str]):
        """Show simulation statistics."""
        self.simulator.stats()
    
    def cmd_snapshot(self, args: List[str]):
        """Save current state snapshot."""
        if not args:
            print("Usage: snapshot <name>")
            return
        
        name = args[0]
        try:
            self.simulator.save_snapshot(name)
        except Exception as e:
            print(f"Error saving snapshot: {e}")
    
    def cmd_snapshots(self, args: List[str]):
        """List snapshots."""
        try:
            self.simulator.snapshots_info()
        except Exception as e:
            print(f"Error listing snapshots: {e}")
    
    def cmd_load_snapshot(self, args: List[str]):
        """Load snapshot."""
        if not args:
            print("Usage: load_snapshot <name>")
            return
        
        name = args[0]
        try:
            self.simulator.load_snapshot(name)
        except Exception as e:
            print(f"Error loading snapshot: {e}")
    
    def cmd_delete_snapshot(self, args: List[str]):
        """Delete snapshot."""
        if not args:
            print("Usage: delete_snapshot <name>")
            return
        
        name = args[0]
        try:
            self.simulator.delete_snapshot(name)
        except Exception as e:
            print(f"Error deleting snapshot: {e}")
    
    def cmd_load(self, args: List[str]):
        """Load quantum script."""
        if not args:
            print("Usage: load <filename>")
            return
        
        filename = args[0]
        try:
            self.load_file(filename)
        except Exception as e:
            print(f"Error loading file: {e}")
    
    def load_file(self, filename: str):
        """Load commands from file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse commands
        commands = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                commands.append((line_num, line))
        
        self.script_commands = commands
        self.loaded_files.append(filename)
        print(f"Loaded {len(commands)} commands from {filename}")
    
    def cmd_run(self, args: List[str]):
        """Run loaded script."""
        if not self.script_commands:
            print("No script loaded. Use 'load <filename>' first.")
            return
        
        print(f"Executing {len(self.script_commands)} commands...")
        
        for line_num, command in self.script_commands:
            try:
                print(f"[{line_num}] {command}")
                self.execute_command(command)
            except Exception as e:
                print(f"Error at line {line_num}: {e}")
                if input("Continue? (y/N): ").lower() != 'y':
                    break
        
        print("Script execution completed")
    
    def cmd_files(self, args: List[str]):
        """Show loaded files."""
        if self.loaded_files:
            print("Loaded files:")
            for f in self.loaded_files:
                print(f"  {f}")
        else:
            print("No files loaded")
    
    def cmd_gates(self, args: List[str]):
        """List available gates."""
        from quantum_gates import QuantumGates
        gates = QuantumGates.get_gate_list()
        print("Available quantum gates:")
        
        # Group gates by type
        single_qubit = ['I', 'X', 'Y', 'Z', 'H', 'S', 'S_dagger', 'T', 'T_dagger']
        rotation = ['RX', 'RY', 'RZ', 'phase', 'U', 'U1', 'U2', 'U3']
        two_qubit = ['CNOT', 'CX', 'CZ', 'CY', 'SWAP', 'iSWAP', 'SQRT_SWAP']
        controlled = ['CRX', 'CRY', 'CRZ', 'CU']
        multi_qubit = ['TOFFOLI', 'CCX', 'FREDKIN', 'CSWAP']
        
        print("\n  Single-qubit gates:")
        print("   ", ', '.join(single_qubit))
        
        print("\n  Rotation gates:")
        print("   ", ', '.join(rotation))
        
        print("\n  Two-qubit gates:")
        print("   ", ', '.join(two_qubit))
        
        print("\n  Controlled rotation gates:")
        print("   ", ', '.join(controlled))
        
        print("\n  Multi-qubit gates:")
        print("   ", ', '.join(multi_qubit))
    
    def cmd_gate_info(self, args: List[str]):
        """Information about specific gate."""
        if not args:
            print("Usage: gate_info <gate_name>")
            return
        
        gate_name = args[0].upper()
        from quantum_gates import QuantumGates
        info = QuantumGates.get_gate_info(gate_name)
        print(f"{gate_name}: {info}")
    
    def cmd_version(self, args: List[str]):
        """Show version information."""
        print("Quantum Simulator v1.0")
        print("Features:")
        print("  - Complete quantum gate library")
        print("  - Interactive REPL interface")
        print("  - ASCII circuit visualization")
        print("  - Qiskit code export")
        print("  - Advanced state analysis")
        print("  - Entanglement measurement")
        print("  - Performance statistics")
        print("  - Script execution")
        print("  - Snapshot system")
        print("  - No artificial limitations")
