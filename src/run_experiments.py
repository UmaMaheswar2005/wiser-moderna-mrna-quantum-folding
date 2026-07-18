"""
run_experiments.py

Runs the full pipeline over every sequence in data/example_sequences.py
(plus any you add) and logs one row of results per sequence to
results/experiments.csv -- this CSV is your evidence base for the
"Implementation and benchmarking" and "Scaling and quantum resource
analysis" sections of the report.

Degrades gracefully exactly like benchmark_runner.py: columns that need
ViennaRNA or qiskit are left blank (not crashed) if those aren't installed
yet, so you can start logging classical-only results immediately and see
the rest fill in as you install dependencies.

Usage:
    cd src
    python3 run_experiments.py                  # classical only (fast)
    python3 run_experiments.py --qaoa            # include QAOA (slower)
    python3 run_experiments.py --qaoa --max-qubits-qaoa 40   # cap QAOA cost
"""

import argparse
import csv
import os
import time

from sequence_utils import pairs_to_dot_bracket
from qubo_builder import build_qubo, decode_solution
from classical_solvers import nussinov_max_pairs, solve_qubo_brute_force

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "experiments.csv")

FIELDNAMES = [
    "sequence_name", "n_nt", "sequence",
    "n_qubits_naive", "n_qubits_filtered",
    "nussinov_structure", "nussinov_n_pairs",
    "qubo_exact_structure", "qubo_exact_energy", "qubo_matches_nussinov",
    "qaoa_structure", "qaoa_energy", "qaoa_runtime_sec", "qaoa_matches_exact",
    "vienna_mfe_structure", "vienna_mfe_energy",
    "base_pair_distance", "precision", "recall", "f1", "exact_match",
]


def run_one(name, sequence, use_qaoa=False, max_qubits_qaoa=25, bpp_threshold=0.01,
            qaoa_reps=2, qaoa_shots=2048):
    row = {k: "" for k in FIELDNAMES}
    row["sequence_name"] = name
    row["n_nt"] = len(sequence)
    row["sequence"] = sequence

    linear, quadratic, pairs = build_qubo(sequence, min_loop=3, stack_bonus=1.0)
    m = len(pairs)
    row["n_qubits_naive"] = m

    # base-pair-probability filtering (only if ViennaRNA is installed)
    try:
        from classical_reference import filter_candidates_by_probability
        filtered = filter_candidates_by_probability(sequence, pairs, threshold=bpp_threshold)
        row["n_qubits_filtered"] = len(filtered)
    except ImportError:
        pass

    nuss_structure, nuss_pairs, nuss_count = nussinov_max_pairs(sequence, min_loop=3)
    row["nussinov_structure"] = nuss_structure
    row["nussinov_n_pairs"] = nuss_count

    qubo_structure = None
    if m <= 20:
        x, e = solve_qubo_brute_force(linear, quadratic, m)
        qubo_pairs = decode_solution(x, pairs)
        qubo_structure = pairs_to_dot_bracket(sequence, qubo_pairs)
        row["qubo_exact_structure"] = qubo_structure
        row["qubo_exact_energy"] = round(e, 4)
        row["qubo_matches_nussinov"] = (qubo_structure == nuss_structure)

    if use_qaoa and m <= max_qubits_qaoa:
        try:
            from quantum_solver_qaoa import solve_qubo_qaoa
            t0 = time.time()
            result = solve_qubo_qaoa(linear, quadratic, m, reps=qaoa_reps,
                                      shots=qaoa_shots, verbose=False)
            runtime = time.time() - t0
            qaoa_pairs = decode_solution(result["x"], pairs)
            qaoa_structure = pairs_to_dot_bracket(sequence, qaoa_pairs)
            row["qaoa_structure"] = qaoa_structure
            row["qaoa_energy"] = round(result["energy"], 4)
            row["qaoa_runtime_sec"] = round(runtime, 2)
            if qubo_structure is not None:
                row["qaoa_matches_exact"] = (qaoa_structure == qubo_structure)
        except (ImportError, ModuleNotFoundError) as exc:
            row["qaoa_structure"] = f"skipped (missing dependency: {exc})"
    elif use_qaoa:
        row["qaoa_structure"] = f"skipped (m={m} > max_qubits_qaoa={max_qubits_qaoa})"

    try:
        from classical_reference import get_mfe_structure, compare_to_reference
    except ImportError:
        pass
    else:
        mfe_structure, mfe_energy = get_mfe_structure(sequence)
        row["vienna_mfe_structure"] = mfe_structure
        row["vienna_mfe_energy"] = mfe_energy

        compare_target = qubo_structure or nuss_structure
        if use_qaoa and row["qaoa_structure"] and "skipped" not in str(row["qaoa_structure"]):
            compare_target = row["qaoa_structure"]
        metrics = compare_to_reference(compare_target, mfe_structure)
        row["base_pair_distance"] = metrics["base_pair_distance"]
        row["precision"] = round(metrics["precision"], 3)
        row["recall"] = round(metrics["recall"], 3)
        row["f1"] = round(metrics["f1"], 3)
        row["exact_match"] = metrics["exact_match"]

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qaoa", action="store_true", help="also run QAOA (needs qiskit + qiskit-aer)")
    parser.add_argument("--max-qubits-qaoa", type=int, default=25,
                         help="skip QAOA above this qubit count (keeps runtime sane)")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.example_sequences import EXAMPLE_SEQUENCES

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    rows = []
    for name, seq in EXAMPLE_SEQUENCES.items():
        print(f"running: {name} ({len(seq)} nt)...")
        rows.append(run_one(name, seq, use_qaoa=args.qaoa, max_qubits_qaoa=args.max_qubits_qaoa))

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} rows to {RESULTS_CSV}")
    print("open it in Excel/Google Sheets/pandas -- this is your results table for the report.")


if __name__ == "__main__":
    main()
