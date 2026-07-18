"""
classical_solvers.py

Two classical solvers that need NO quantum SDK and NO ViennaRNA. They exist
to validate the QUBO formulation itself, independently of whichever quantum
method you point at it later:

  1. solve_qubo_brute_force -- exhaustively checks every 2^m assignment and
     returns the true QUBO ground state. Only tractable for small m (roughly
     m <= 22 or so before it gets slow), which is exactly the regime the
     challenge tells you to start in ("reproduce known benchmark structures
     for small RNA sequences"). This is your ground truth for "did my
     quantum solver actually find the QUBO optimum" -- a question you must
     answer *before* asking "does the QUBO optimum match real biology".

  2. nussinov_max_pairs -- the classical O(n^3) dynamic-programming algorithm
     (Nussinov 1978) that exactly maximizes the number of non-crossing base
     pairs. Our QUBO's reward term is doing the same thing the Nussinov
     recursion does, so with stack_bonus=0 the two should always find
     solutions with the same number of pairs. Any mismatch means there is a
     bug in the QUBO (wrong penalty weights, wrong candidate-pair
     enumeration, etc.) -- catch that here, cheaply, before ever touching
     a quantum backend.

Keeping these two independent is deliberate: Nussinov re-derives the answer
from scratch by dynamic programming, so it can't inherit a bug from
qubo_builder.py. Agreement between the two is real evidence of correctness.
"""

import itertools
import numpy as np

from sequence_utils import generate_candidate_pairs, is_valid_pair, pairs_to_dot_bracket
from qubo_builder import energy_of_assignment, decode_solution


def solve_qubo_brute_force(linear, quadratic, m):
    """
    Exhaustive search over all 2^m assignments. Returns (best_x, best_energy).
    Intended for m up to about 20-22 (2^22 ~= 4M evaluations, a few seconds
    in pure Python; use PyPy or vectorize with numpy if you push further).
    """
    best_x, best_e = None, float("inf")
    for bits in itertools.product([0, 1], repeat=m):
        x = np.array(bits)
        e = energy_of_assignment(x, linear, quadratic)
        if e < best_e:
            best_e, best_x = e, x
    return best_x, best_e


def nussinov_max_pairs(sequence, min_loop=3):
    """
    Classical Nussinov DP: maximize the number of non-crossing base pairs.
    Returns (dot_bracket_string, list_of_pairs, max_pair_count).

    This is the *exact* solution to the simplified "maximize base pairs"
    objective -- i.e. what our QUBO is a quantum-optimization relaxation of
    when stack_bonus=0. It is NOT the same as the true thermodynamic MFE
    (ViennaRNA) -- see classical_reference.py and the README for that
    comparison, which is the more biologically meaningful one.
    """
    seq = sequence.upper().replace("T", "U")
    n = len(seq)
    dp = np.zeros((n, n), dtype=int)

    for span in range(min_loop + 1, n):
        for i in range(0, n - span):
            j = i + span
            best = dp[i + 1][j]  # i unpaired
            best = max(best, dp[i][j - 1])  # j unpaired
            if is_valid_pair(seq[i], seq[j]):
                paired = 1 + (dp[i + 1][j - 1] if j - 1 >= i + 1 else 0)
                best = max(best, paired)
            for k in range(i, j):  # bifurcation: independent substructures
                best = max(best, dp[i][k] + dp[k + 1][j])
            dp[i][j] = best

    # traceback
    pairs = []

    def traceback(i, j):
        if j - i <= min_loop:
            return
        if dp[i][j] == dp[i + 1][j]:
            traceback(i + 1, j)
        elif dp[i][j] == dp[i][j - 1]:
            traceback(i, j - 1)
        elif is_valid_pair(seq[i], seq[j]) and dp[i][j] == 1 + (
            dp[i + 1][j - 1] if j - 1 >= i + 1 else 0
        ):
            pairs.append((i, j))
            traceback(i + 1, j - 1)
        else:
            for k in range(i, j):
                if dp[i][j] == dp[i][k] + dp[k + 1][j]:
                    traceback(i, k)
                    traceback(k + 1, j)
                    break

    traceback(0, n - 1)
    pairs.sort()
    structure = pairs_to_dot_bracket(seq, pairs)
    return structure, pairs, int(dp[0][n - 1])


if __name__ == "__main__":
    from qubo_builder import build_qubo

    seq = "GGGAAACCC"
    print(f"sequence: {seq}\n")

    # --- Nussinov ground truth (exact DP) ---
    structure, pairs, n_pairs = nussinov_max_pairs(seq, min_loop=3)
    print(f"[Nussinov DP]        structure = {structure}   pairs = {pairs}")

    # --- QUBO, solved exactly by brute force, stack_bonus=0 (should match Nussinov's pair COUNT) ---
    linear, quadratic, cand_pairs = build_qubo(seq, min_loop=3, stack_bonus=0.0)
    x, e = solve_qubo_brute_force(linear, quadratic, len(cand_pairs))
    qubo_pairs = decode_solution(x, cand_pairs)
    qubo_structure = pairs_to_dot_bracket(seq, qubo_pairs)
    print(f"[QUBO brute force]   structure = {qubo_structure}   pairs = {qubo_pairs}   energy = {e}")
    print(f"cross-check: same number of pairs? {len(qubo_pairs) == n_pairs}")

    # --- now with stacking bonus on: should still find >= as many pairs, prefers helices ---
    linear2, quadratic2, cand_pairs2 = build_qubo(seq, min_loop=3, stack_bonus=1.0)
    x2, e2 = solve_qubo_brute_force(linear2, quadratic2, len(cand_pairs2))
    qubo_pairs2 = decode_solution(x2, cand_pairs2)
    print(f"[QUBO + stacking]   structure = {pairs_to_dot_bracket(seq, qubo_pairs2)}   pairs = {qubo_pairs2}   energy = {e2}")
