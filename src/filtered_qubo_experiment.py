"""
filtered_qubo_experiment.py

Closes the loop on the filtering result from the last full sweep: base-pair-
probability filtering brought challenge_example_44nt from 313 candidate
pairs down to 26. That number was previously just *reported* --
n_qubits_filtered in run_experiments.py's CSV was computed but never
actually used to build a smaller QUBO. This script does that: filter,
rebuild a restricted QUBO, solve it (classically first as a sanity check,
then via QAOA), and compare the result to ViennaRNA's real MFE for the
literal example sequence given in the WISER/Moderna challenge document.

Usage:
    python3 filtered_qubo_experiment.py                 # classical only
    python3 filtered_qubo_experiment.py --qaoa           # + QAOA
    python3 filtered_qubo_experiment.py --qaoa --maxiter 25 --ram-gb 4
"""

import argparse
import csv
import os
import time

from sequence_utils import generate_candidate_pairs, pairs_to_dot_bracket
from qubo_builder import build_qubo, decode_solution, energy_of_assignment
from classical_solvers import solve_qubo_brute_force, nussinov_max_pairs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SEQUENCE = "GGAGCAAAACUUGUCGAUUGAGAACAAAAUACAGAAUUUGCUUG"  # from the challenge doc
BPP_THRESHOLD = 0.01


def run(use_qaoa=False, maxiter=25, ram_gb=4, reps=2, shots=2048):
    seq = SEQUENCE
    print(f"sequence ({len(seq)} nt): {seq}")

    naive_pairs = generate_candidate_pairs(seq, min_loop=3)
    print(f"naive candidate pairs (qubits): {len(naive_pairs)}")

    try:
        from classical_reference import (filter_candidates_by_probability,
                                          get_mfe_structure, compare_to_reference)
    except ImportError:
        print("ViennaRNA not installed -- can't filter or compare to real MFE. "
              "Run: pip install ViennaRNA")
        return

    filtered_pairs = filter_candidates_by_probability(seq, naive_pairs, threshold=BPP_THRESHOLD)
    print(f"filtered candidate pairs (qubits): {len(filtered_pairs)}  "
          f"({100*(1 - len(filtered_pairs)/len(naive_pairs)):.1f}% reduction)")

    mfe_structure, mfe_energy = get_mfe_structure(seq)
    print(f"ViennaRNA MFE: {mfe_structure}  ({mfe_energy} kcal/mol)")

    # --- build the RESTRICTED QUBO (this is the actual fix) ---
    linear, quadratic, pairs = build_qubo(seq, min_loop=3, stack_bonus=1.0,
                                          restrict_to_pairs=filtered_pairs)
    m = len(pairs)
    print(f"\nrestricted QUBO: {m} qubits, {len(quadratic)} two-qubit terms")

    # --- classical sanity check first, exactly like every other sequence in this project ---
    classical_structure = None
    if m <= 22:
        t0 = time.time()
        x, e = solve_qubo_brute_force(linear, quadratic, m)
        classical_structure = pairs_to_dot_bracket(seq, decode_solution(x, pairs))
        print(f"\n[exact brute force, {time.time()-t0:.1f}s]")
        print(f"  structure: {classical_structure}")
        print(f"  energy: {e:.3f}")
        comparison = compare_to_reference(classical_structure, mfe_structure)
        print(f"  vs. real ViennaRNA MFE: {comparison}")
    else:
        print(f"\n[exact brute force] skipped: {m} qubits still too many for "
              f"2^{m} exhaustive search (raise threshold in code if needed).")

    # --- QAOA on the restricted problem -- the actual point of this script ---
    if use_qaoa:
        try:
            from quantum_solver_qaoa import solve_qubo_qaoa, TooManyQubitsForStatevector
        except ImportError as exc:
            print(f"\n[QAOA] skipped -- missing dependency ({exc})")
            return

        print(f"\n[QAOA] {m} qubits, maxiter={maxiter}, reps={reps}, ram_gb={ram_gb}")
        try:
            t0 = time.time()
            result = solve_qubo_qaoa(linear, quadratic, m, reps=reps, shots=shots,
                                      maxiter=maxiter, seed=0, verbose=False,
                                      available_ram_gb=ram_gb)
            elapsed = time.time() - t0
            qaoa_structure = pairs_to_dot_bracket(seq, decode_solution(result["x"], pairs))
            print(f"  structure: {qaoa_structure}")
            print(f"  energy: {result['energy']:.3f}   ({elapsed:.1f}s)")
            if classical_structure is not None:
                print(f"  matches classical brute-force optimum: {qaoa_structure == classical_structure}")
            comparison = compare_to_reference(qaoa_structure, mfe_structure)
            print(f"  vs. real ViennaRNA MFE: {comparison}")

            from sequence_utils import dot_bracket_to_pairs
            mfe_pairs = set(dot_bracket_to_pairs(mfe_structure))
            qaoa_pairs = set(dot_bracket_to_pairs(qaoa_structure))

            os.makedirs(RESULTS_DIR, exist_ok=True)
            csv_path = os.path.join(RESULTS_DIR, "filtered_qubo_44nt_result.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "sequence", "n_nt", "n_qubits_naive", "n_qubits_filtered",
                    "reps", "maxiter", "shots", "runtime_sec",
                    "qaoa_structure", "qaoa_qubo_energy",
                    "vienna_mfe_structure", "vienna_mfe_energy_kcal_mol",
                    "n_pairs_matching", "n_pairs_mfe_only", "n_pairs_qaoa_only",
                    "base_pair_distance", "precision", "recall", "f1", "exact_match",
                ])
                writer.writeheader()
                writer.writerow({
                    "sequence": seq, "n_nt": len(seq),
                    "n_qubits_naive": len(naive_pairs), "n_qubits_filtered": len(filtered_pairs),
                    "reps": reps, "maxiter": maxiter, "shots": shots,
                    "runtime_sec": round(elapsed, 1),
                    "qaoa_structure": qaoa_structure, "qaoa_qubo_energy": round(result["energy"], 3),
                    "vienna_mfe_structure": mfe_structure, "vienna_mfe_energy_kcal_mol": mfe_energy,
                    "n_pairs_matching": len(mfe_pairs & qaoa_pairs),
                    "n_pairs_mfe_only": len(mfe_pairs - qaoa_pairs),
                    "n_pairs_qaoa_only": len(qaoa_pairs - mfe_pairs),
                    "base_pair_distance": comparison["base_pair_distance"],
                    "precision": round(comparison["precision"], 4),
                    "recall": round(comparison["recall"], 4),
                    "f1": round(comparison["f1"], 4),
                    "exact_match": comparison["exact_match"],
                })
            print(f"\n  saved -> {csv_path}")
        except TooManyQubitsForStatevector as exc:
            print(f"  skipped -- {exc}")


def time_one_iteration(maxiter=1, ram_gb=4):
    """
    Diagnostic: time a single COBYLA iteration on the restricted QUBO,
    the same approach used earlier for stem_bulge_15nt, so you get a real
    number for THIS specific problem before committing to a full run.
    """
    seq = SEQUENCE
    naive_pairs = generate_candidate_pairs(seq, min_loop=3)
    try:
        from classical_reference import filter_candidates_by_probability
    except ImportError:
        print("ViennaRNA not installed -- can't filter. Run: pip install ViennaRNA")
        return
    filtered_pairs = filter_candidates_by_probability(seq, naive_pairs, threshold=BPP_THRESHOLD)
    linear, quadratic, pairs = build_qubo(seq, min_loop=3, stack_bonus=1.0,
                                          restrict_to_pairs=filtered_pairs)
    m = len(pairs)
    print(f"qubits: {m}")

    try:
        from quantum_solver_qaoa import solve_qubo_qaoa
    except ImportError:
        print("qiskit/qiskit-aer not installed -- can't time QAOA. "
              "Run: pip install qiskit qiskit-aer")
        return
    # request enough evaluations that COBYLA/PRIMA won't need to silently
    # raise it (minimum viable is num_vars+2 = 2*reps+2; reps=1 needs >=4).
    # Then divide by the ACTUAL number of evaluations that ran (from the
    # returned cvar_trace, one entry per real objective() call), not by
    # the requested maxiter -- if scipy still adjusts it for any reason,
    # this stays correct regardless.
    safe_maxiter = max(maxiter, 6)
    t0 = time.time()
    result = solve_qubo_qaoa(linear, quadratic, m, reps=1, shots=2048, maxiter=safe_maxiter,
                              seed=0, verbose=False, available_ram_gb=ram_gb)
    actual_evals = len(result["cvar_trace"])
    per_iter = (time.time() - t0) / actual_evals
    print(f"requested maxiter={safe_maxiter}, actual evaluations={actual_evals}")
    print(f"~{per_iter:.2f}s per COBYLA iteration at reps=1")
    print(f"full run (maxiter=25), reps=1: ~{per_iter*25/60:.1f} min")
    print(f"full run (maxiter=25), reps=2: ~{per_iter*25*2/60:.1f} min (roughly 2x gates)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qaoa", action="store_true")
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--ram-gb", type=float, default=4)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--time-check", action="store_true",
                        help="just time one iteration and estimate, don't run the full thing")
    args = parser.parse_args()

    if args.time_check:
        time_one_iteration(ram_gb=args.ram_gb)
    else:
        run(use_qaoa=args.qaoa, maxiter=args.maxiter, ram_gb=args.ram_gb, reps=args.reps)