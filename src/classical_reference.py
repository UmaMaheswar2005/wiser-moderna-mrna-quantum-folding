"""
classical_reference.py

Thin wrapper around ViennaRNA -- this is your "ground truth" (challenge
deliverable #2) and your energy evaluator (deliverable #3).

Requires the `ViennaRNA` PyPI package (the Python bindings, imported as
`RNA`):
    pip install ViennaRNA

NOTE ON THIS SANDBOX: ViennaRNA isn't installed here (no network access
in this environment to fetch it), so this file is written but not
executed in this session -- everything else in this project (qubo_builder,
classical_solvers, scaling_analysis) *was* run and verified here. Install
ViennaRNA in your own environment (locally or Colab -- `pip install
ViennaRNA` works fine on both) and this module should work as-is; the API
calls mirror the exact usage shown in the challenge document.
"""

import RNA  # pip install ViennaRNA

from sequence_utils import dot_bracket_to_pairs, base_pair_distance


def get_mfe_structure(sequence):
    """Wraps RNA.fold(). Returns (dot_bracket_structure, mfe_energy_kcal_mol)."""
    structure, mfe = RNA.fold(sequence)
    return structure, mfe


def evaluate_structure_energy(sequence, structure):
    """
    Wraps RNA.fold_compound(...).eval_structure(...) -- the actual Turner
    nearest-neighbor free energy of a GIVEN structure (not necessarily the
    MFE one). This is how you score whatever your quantum solver returns.
    """
    fc = RNA.fold_compound(sequence)
    return fc.eval_structure(structure)


def base_pair_probabilities(sequence):
    """
    Runs the partition-function (McCaskill) algorithm and returns the full
    base-pair probability matrix as a dict {(i, j): probability}.
    Useful for pruning candidate QUBO variables (see
    `filter_candidates_by_probability` below) and, more generally, for
    reasoning about which pairs are thermodynamically plausible at all.
    """
    fc = RNA.fold_compound(sequence)
    fc.pf()  # runs the partition function calculation
    bpp = fc.bpp()  # (n+1) x (n+1) upper-triangular matrix, 1-indexed
    probs = {}
    n = len(sequence)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            p = bpp[i][j]
            if p > 0:
                probs[(i - 1, j - 1)] = p  # convert to 0-indexed
    return probs


def filter_candidates_by_probability(sequence, candidate_pairs, threshold=0.01):
    """
    Addresses the qubit-blowup problem found in scaling_analysis.py: the
    naive "every Watson-Crick/wobble pair passing the loop-length rule" QUBO
    needs ~530 qubits for a 60-nt sequence, while the published IBM/Moderna
    work solves 60-nt problems in the 80-156 qubit range. The gap is exactly
    this filtering step -- most geometrically-valid candidate pairs have
    negligible equilibrium probability and can be dropped before the QUBO
    is even built. Rerun qubo_builder.build_qubo with only the surviving
    pairs (you'll need a small modification to accept a pre-computed
    candidate list instead of regenerating it -- see the README).
    """
    probs = base_pair_probabilities(sequence)
    return [p for p in candidate_pairs if probs.get(p, 0.0) >= threshold]


def compare_to_reference(predicted_structure, reference_structure):
    """
    Structure-vs-structure comparison for challenge deliverable #4
    ("Report accuracy, energy gap from the reference MFE structure...").
    Returns base-pair distance plus precision/recall/F1 over base pairs,
    which is the standard way RNA-folding papers report "accuracy" when
    the predicted structure isn't an exact match.
    """
    pred_pairs = set(dot_bracket_to_pairs(predicted_structure))
    ref_pairs = set(dot_bracket_to_pairs(reference_structure))

    tp = len(pred_pairs & ref_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else (1.0 if not ref_pairs else 0.0)
    recall = tp / len(ref_pairs) if ref_pairs else (1.0 if not pred_pairs else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "base_pair_distance": base_pair_distance(pred_pairs, ref_pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": predicted_structure == reference_structure,
    }


if __name__ == "__main__":
    # example from the challenge document
    sequence = "GGAGCAAAACUUGUCGAUUGAGAACAAAAUACAGAAUUUGCUUG"
    structure, mfe = get_mfe_structure(sequence)
    print(f"MFE structure: {structure}")
    print(f"MFE energy: {mfe} kcal/mol")

    candidate_structure = "." * len(sequence)  # placeholder: your solver's output goes here
    energy = evaluate_structure_energy(sequence, candidate_structure)
    print(f"energy of candidate structure: {energy} kcal/mol")

    print(compare_to_reference(candidate_structure, structure))
