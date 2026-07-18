"""
benchmark_runner.py

The end-to-end pipeline for one sequence: build QUBO -> solve (classically
and/or via QAOA) -> compare to ViennaRNA MFE -> report. This is what
deliverable #4 ("Implementation and benchmarking") is asking you to run
and write up for each of your test sequences.

Run directly:
    cd src && python3 benchmark_runner.py

Works with only numpy/networkx installed (everything not requiring
ViennaRNA/qiskit still runs and prints); installs are checked at runtime
and skipped with a clear message if missing, rather than crashing, so you
can always see the classical baseline even before your quantum environment
is fully set up.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sequence_utils import pairs_to_dot_bracket
from qubo_builder import build_qubo, decode_solution
from classical_solvers import nussinov_max_pairs, solve_qubo_brute_force


def run_pipeline(sequence, min_loop=3, stack_bonus=1.0, use_qaoa=False, qaoa_reps=2, qaoa_shots=2048):
    print("=" * 70)
    print(f"sequence ({len(sequence)} nt): {sequence}")
    print("=" * 70)

    # --- 1. build the QUBO ---
    linear, quadratic, pairs = build_qubo(sequence, min_loop=min_loop, stack_bonus=stack_bonus)
    m = len(pairs)
    print(f"\n[QUBO] {m} candidate pairs -> {m} qubits (naive encoding), "
          f"{len(quadratic)} two-qubit interaction terms")

    # --- 2. classical ground truth for the SIMPLIFIED (pair-counting) objective ---
    nuss_structure, nuss_pairs, nuss_count = nussinov_max_pairs(sequence, min_loop=min_loop)
    print(f"\n[Nussinov DP -- exact solution to the simplified objective]")
    print(f"  structure: {nuss_structure}")
    print(f"  # pairs:   {nuss_count}")

    # --- 3. solve the QUBO exactly by brute force, IF small enough ---
    qubo_structure = None
    if m <= 20:
        x, e = solve_qubo_brute_force(linear, quadratic, m)
        qubo_pairs = decode_solution(x, pairs)
        qubo_structure = pairs_to_dot_bracket(sequence, qubo_pairs)
        print(f"\n[QUBO, exact brute force]")
        print(f"  structure: {qubo_structure}")
        print(f"  energy:    {e:.3f}")
        print(f"  matches Nussinov structure? {qubo_structure == nuss_structure}")
    else:
        print(f"\n[QUBO, exact brute force] skipped: {m} qubits is too many for 2^{m} "
              f"exhaustive search in this quick demo (raise the threshold in the code "
              f"if you have the patience / a faster machine).")

    # --- 4. QAOA (optional -- requires qiskit + qiskit-aer) ---
    if use_qaoa:
        try:
            from quantum_solver_qaoa import solve_qubo_qaoa
        except ImportError as exc:
            print(f"\n[QAOA] skipped -- missing dependency ({exc}). "
                  f"Run: pip install qiskit qiskit-aer")
        else:
            result = solve_qubo_qaoa(linear, quadratic, m, reps=qaoa_reps,
                                      shots=qaoa_shots, verbose=False)
            qaoa_pairs = decode_solution(result["x"], pairs)
            qaoa_structure = pairs_to_dot_bracket(sequence, qaoa_pairs)
            print(f"\n[QAOA]")
            print(f"  structure: {qaoa_structure}")
            print(f"  energy:    {result['energy']:.3f}")
            if qubo_structure is not None:
                print(f"  matches brute-force QUBO optimum? {qaoa_structure == qubo_structure}")

    # --- 5. ViennaRNA reference (optional -- requires the ViennaRNA package) ---
    try:
        from classical_reference import get_mfe_structure, compare_to_reference
    except ImportError as exc:
        print(f"\n[ViennaRNA reference] skipped -- missing dependency ({exc}). "
              f"Run: pip install ViennaRNA")
    else:
        mfe_structure, mfe_energy = get_mfe_structure(sequence)
        print(f"\n[ViennaRNA MFE -- the real biological ground truth]")
        print(f"  structure: {mfe_structure}")
        print(f"  MFE energy: {mfe_energy} kcal/mol")
        compare_target = qubo_structure or nuss_structure
        print(f"  comparison vs. our QUBO/Nussinov structure: {compare_to_reference(compare_target, mfe_structure)}")

    print()
    return {
        "sequence": sequence,
        "n_qubits": m,
        "nussinov_structure": nuss_structure,
        "qubo_structure": qubo_structure,
    }


if __name__ == "__main__":
    from data.example_sequences import EXAMPLE_SEQUENCES

    for name, seq in EXAMPLE_SEQUENCES.items():
        print(f"\n### {name} ###")
        run_pipeline(seq, use_qaoa=True)  # flip to True once qiskit is installed
