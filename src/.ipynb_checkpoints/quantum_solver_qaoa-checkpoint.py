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


class TooManyQubitsForStatevector(RuntimeError):
    """Raised before ever touching Aer, with the real reason (RAM), not
    Aer's own confusing 'CircuitTooWideForTarget ... max 31' message."""


def max_safe_qubits(available_ram_gb, headroom_fraction=0.5):
    """
    Exact statevector simulation needs 2^n complex128 amplitudes = 2^n * 16
    bytes. `AerSimulator()`'s default 'automatic' method picks a target
    that happens to cap out around 30-31 qubits regardless of your actual
    RAM (a known Aer/Qiskit interaction -- see Qiskit/qiskit#14017 and
    Qiskit/qiskit-aer#2084) -- but that default is, not coincidentally,
    close to what 16-32 GB of RAM can actually hold: 2^31 * 16 bytes is
    already 34 GB, more than even qBraid's 25 GB "Large" tier. This isn't
    a bug to route around; it's the real ceiling of exact simulation on a
    single machine. `headroom_fraction` reserves the rest of RAM for the
    OS, Python, and Qiskit's own overhead during transpilation.
    """
    usable_bytes = available_ram_gb * (1024**3) * (1 - headroom_fraction)
    import math
    return max(1, int(math.log2(usable_bytes / 16)))


def run_circuit_get_counts(qc, shots=2048, seed=42, available_ram_gb=4):
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    m = qc.num_qubits
    limit = max_safe_qubits(available_ram_gb)
    if m > limit:
        needed_gb = (2 ** m * 16) / (1024**3)
        raise TooManyQubitsForStatevector(
            f"{m} qubits needs ~{needed_gb:,.1f} GB for exact statevector simulation "
            f"(2^{m} amplitudes x 16 bytes); with available_ram_gb={available_ram_gb} "
            f"the safe limit is {limit} qubits. This is a real hardware limit, not a "
            f"config issue -- see max_safe_qubits()'s docstring. Options: lower "
            f"max_qubits_qaoa to {limit} or below, filter candidate pairs first "
            f"(classical_reference.filter_candidates_by_probability), or move to a "
            f"qubit-efficient encoding (see the Pauli Correlation Encoding paper in "
            f"the README references)."
        )

    # method='statevector' explicitly, rather than 'automatic': 'automatic'
    # is what triggers Aer's own separate (and lower, ~31-qubit) default cap
    # via a coupling-map check unrelated to your actual RAM -- see the
    # GitHub issues cited above. We've already done our own, RAM-based
    # check above, so we bypass Aer's independent one deliberately.
    backend = AerSimulator(seed_simulator=seed, method="statevector")
    tqc = transpile(qc, backend, coupling_map=None, optimization_level=1)
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
                     maxiter=150, seed=42, verbose=True, available_ram_gb=4):
    """
    Optimizes QAOA angles to minimize the CVaR of the QUBO energy, then
    returns the single lowest-energy bitstring seen across every shot of
    every iteration (not just the final one -- with a stochastic sampler,
    "best ever seen" is the right thing to report, and is what the cited
    papers do too).

    `available_ram_gb`: set this to match your actual qBraid compute tier
    (Small=4, Medium=8, Large=25) so the pre-flight qubit check is accurate
    -- fails immediately with a clear message instead of burning a COBYLA
    iteration (and qBraid CPU-hours) before crashing.

    Returns: dict with keys x (best bitstring, numpy array), energy,
    optimized_params, and n_qubits.
    """
    from scipy.optimize import minimize

    # fail before the optimizer loop even starts, not on iteration 1
    limit = max_safe_qubits(available_ram_gb)
    if m > limit:
        raise TooManyQubitsForStatevector(
            f"{m} qubits exceeds the ~{limit}-qubit safe limit for "
            f"available_ram_gb={available_ram_gb}. See max_safe_qubits() docstring."
        )

    rng = np.random.default_rng(seed)
    h, J, offset = qubo_to_ising(linear, quadratic, m)

    history = {"best_x": None, "best_e": float("inf"), "trace": []}

    def objective(params):
        gammas, betas = params[:reps], params[reps:]
        qc = build_qaoa_circuit(h, J, m, reps, gammas, betas)
        counts = run_circuit_get_counts(qc, shots=shots, seed=seed, available_ram_gb=available_ram_gb)
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
