"""
quantum_solver_qaoa.py

A from-scratch QAOA solver for the RNA-folding QUBO, using CVaR
(Conditional Value at Risk) aggregation of samples -- the same core trick
used in the real IBM/Moderna papers this project is grounded in (Alevras
et al. 2024, arXiv:2405.20328; Kumar et al. 2025, arXiv:2505.05782), just
with plain QAOA instead of their hardware-tuned VQE ansatz. Swapping in
their ansatz later is a natural "depth over breadth" extension.

Requires (NOT installed in the sandbox this was written in -- no network
access here to fetch packages; install in your own environment):
    pip install qiskit qiskit-aer

Everything downstream of `qubo_builder.py` (i.e. this whole file) is
therefore written carefully but UNTESTED in this session. Before you trust
it for real results: run it on the tiny "GGGAAACCC" example first and
confirm it recovers the same "(((...)))" structure that
classical_solvers.py already proved is the true optimum for both the
Nussinov DP and brute-force QUBO solve (see the README's "Verified in
this sandbox" section). If QAOA
doesn't recover it on a 9-nt / 9-qubit toy problem, the bug is in this
file, not in your biology.

Why CVaR and not plain expectation value: with `alpha` < 1, the classical
optimizer only sees the mean of the best `alpha`-fraction of samples each
iteration, rather than being dragged around by the (typically much larger)
population of bad samples a short, noisy near-term circuit produces. This
matters more, not less, on a simulator with a shallow ansatz -- it's not
a noise-mitigation hack, it's a better-shaped optimization objective.
"""

import numpy as np
from qubo_builder import qubo_to_ising, energy_of_assignment


def build_qaoa_circuit(h, J, m, reps, gammas, betas):
    """Standard QAOA circuit: H^{⊗m}, then `reps` layers of (cost, mixer)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(m, m)
    qc.h(range(m))
    for p in range(reps):
        gamma, beta = gammas[p], betas[p]
        for i in range(m):
            if h[i] != 0:
                qc.rz(2 * gamma * h[i], i)
        for (a, b), coeff in J.items():
            if coeff != 0:
                qc.rzz(2 * gamma * coeff, a, b)
        for i in range(m):
            qc.rx(2 * beta, i)
    qc.measure(range(m), range(m))
    return qc


def run_circuit_get_counts(qc, shots=2048, seed=42):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    backend = AerSimulator(seed_simulator=seed)
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    return result.get_counts()


def counts_to_energies(counts, linear, quadratic):
    """Decode every sampled bitstring and score it under the QUBO. Returns
    a list of (energy, x_array) pairs, one entry per shot (duplicates kept,
    so this list IS the empirical energy distribution -- useful for the
    plots the challenge asks for, not just the numbers)."""
    out = []
    for bitstring, count in counts.items():
        x = np.array([int(b) for b in bitstring[::-1]])  # qiskit -> qubit-index order
        e = energy_of_assignment(x, linear, quadratic)
        out.extend([(e, x)] * count)
    return out


def cvar_value(energies_and_x, alpha=0.25):
    """Mean energy of the best alpha-fraction of samples (the CVaR objective)."""
    sorted_pairs = sorted(energies_and_x, key=lambda p: p[0])
    k = max(1, int(alpha * len(sorted_pairs)))
    best_slice = sorted_pairs[:k]
    return float(np.mean([e for e, _ in best_slice]))


def solve_qubo_qaoa(linear, quadratic, m, reps=2, shots=2048, alpha=0.25,
                     maxiter=150, seed=42, verbose=True):
    """
    Optimizes QAOA angles to minimize the CVaR of the QUBO energy, then
    returns the single lowest-energy bitstring seen across every shot of
    every iteration (not just the final one -- with a stochastic sampler,
    "best ever seen" is the right thing to report, and is what the cited
    papers do too).

    Returns: dict with keys x (best bitstring, numpy array), energy,
    optimized_params, and n_qubits.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    h, J, offset = qubo_to_ising(linear, quadratic, m)

    history = {"best_x": None, "best_e": float("inf"), "trace": []}

    def objective(params):
        gammas, betas = params[:reps], params[reps:]
        qc = build_qaoa_circuit(h, J, m, reps, gammas, betas)
        counts = run_circuit_get_counts(qc, shots=shots, seed=seed)
        energies_and_x = counts_to_energies(counts, linear, quadratic)

        # track best-ever bitstring as a side effect
        round_best_e, round_best_x = min(energies_and_x, key=lambda p: p[0])
        if round_best_e < history["best_e"]:
            history["best_e"] = round_best_e
            history["best_x"] = round_best_x
        history["trace"].append(cvar_value(energies_and_x, alpha=alpha))

        return cvar_value(energies_and_x, alpha=alpha)

    x0 = rng.uniform(0, 2 * np.pi, size=2 * reps)
    result = minimize(objective, x0, method="COBYLA",
                       options={"maxiter": maxiter, "disp": verbose})

    return {
        "x": history["best_x"],
        "energy": history["best_e"],
        "optimized_params": result.x,
        "n_qubits": m,
        "cvar_trace": history["trace"],
    }


if __name__ == "__main__":
    # Requires: pip install qiskit qiskit-aer
    from qubo_builder import build_qubo, decode_solution
    from sequence_utils import pairs_to_dot_bracket
    from classical_solvers import nussinov_max_pairs

    seq = "GGGAAACCC"  # the same toy example validated in classical_solvers.py
    linear, quadratic, pairs = build_qubo(seq, min_loop=3, stack_bonus=1.0)
    print(f"sequence: {seq}  |  qubits needed: {len(pairs)}")

    ref_structure, ref_pairs, _ = nussinov_max_pairs(seq, min_loop=3)
    print(f"Nussinov reference: {ref_structure}")

    result = solve_qubo_qaoa(linear, quadratic, len(pairs), reps=3, shots=4096)
    predicted_pairs = decode_solution(result["x"], pairs)
    predicted_structure = pairs_to_dot_bracket(seq, predicted_pairs)
    print(f"QAOA best structure: {predicted_structure}   energy: {result['energy']}")
    print(f"matches Nussinov reference? {predicted_structure == ref_structure}")
