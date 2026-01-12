{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    elan      # Lean version manager
    lean4     # Lean 4 compiler
  ];

  shellHook = ''
    echo "LA-ACIQ Formal Verification Environment"
    echo "Lean 4 with Mathlib for quantization theory proofs"
    echo ""
    echo "Commands:"
    echo "  lake build    - Build the project"
    echo "  lake exe      - Run executables"
    echo "  lean --version"
  '';
}
