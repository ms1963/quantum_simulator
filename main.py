#!/usr/bin/env python3


#-------------------------
#
# Quantum Simulator
# (c) 2025, Michael Stal 
#
#-------------------------


"""
Sophisticated Quantum Simulator
Main entry point with command-line argument parsing
"""

import argparse
import sys
from quantum_repl import QuantumREPL


def main():
    parser = argparse.ArgumentParser(
        description="Sophisticated Quantum Simulator with REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start with default 3 qubits
  python main.py -q 10              # Start with 10 qubits
  python main.py --qubits 5         # Start with 5 qubits
  python main.py -f script.qs       # Load and run quantum script
        """
    )
    
    parser.add_argument(
        '-q', '--qubits',
        type=int,
        default=3,
        help='Number of qubits (default: 3)'
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Quantum script file to load on startup'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Quantum Simulator v1.0'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.qubits < 1:
        print("Error: Number of qubits must be at least 1")
        sys.exit(1)
    
    if args.qubits > 50:
        print(f"Warning: {args.qubits} qubits will require {2**args.qubits * 16 / (1024**3):.2f} GB of memory")
        response = input("Continue? (y/N): ").lower()
        if response != 'y':
            sys.exit(0)
    
    try:
        # Initialize REPL
        repl = QuantumREPL(num_qubits=args.qubits)
        
        # Load file if specified
        if args.file:
            try:
                repl.load_file(args.file)
                print(f"Loaded quantum script: {args.file}")
            except Exception as e:
                print(f"Error loading file {args.file}: {e}")
                sys.exit(1)
        
        # Start REPL
        repl.run()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
