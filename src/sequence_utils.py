"""
sequence_utils.py

Low-level helpers shared by every other module:
  - which nucleotides are allowed to pair (Watson-Crick + G-U wobble)
  - enumerating candidate base pairs for a sequence
  - converting a set of chosen pairs <-> dot-bracket notation
  - structural validity checks and a base-pair-distance metric for scoring

No external dependencies (pure Python) so this runs anywhere, including
this sandbox, your laptop, or Colab.
"""

from itertools import combinations

# Watson-Crick pairs + G-U wobble (the standard set used in nearly all
# RNA secondary structure tools, including ViennaRNA's default model)
VALID_PAIRS = {
    ("A", "U"), ("U", "A"),
    ("G", "C"), ("C", "G"),
    ("G", "U"), ("U", "G"),
}


def is_valid_pair(b1: str, b2: str) -> bool:
    """True if two nucleotides are allowed to base-pair."""
    return (b1.upper(), b2.upper()) in VALID_PAIRS


def generate_candidate_pairs(sequence: str, min_loop: int = 3):
    """
    Enumerate every (i, j) with i < j that is BOTH:
      - chemically valid (Watson-Crick or G-U wobble), and
      - geometrically valid (j - i > min_loop, i.e. a hairpin closed by
        (i, j) would enclose at least `min_loop` unpaired bases -- the
        standard minimum hairpin loop size used by ViennaRNA and friends).

    These (i, j) pairs become the binary decision variables x_{i,j} of the
    QUBO: x_{i,j} = 1 means "base i is paired with base j".

    Returns a list of (i, j) tuples. The index of a pair in this list is
    its QUBO variable index everywhere else in this codebase.
    """
    seq = sequence.upper().replace("T", "U")  # tolerate DNA-style input
    n = len(seq)
    pairs = []
    for i in range(n):
        for j in range(i + min_loop + 1, n):
            if is_valid_pair(seq[i], seq[j]):
                pairs.append((i, j))
    return pairs


def pairs_to_dot_bracket(sequence: str, chosen_pairs) -> str:
    """
    Convert a set/list of (i, j) index pairs into ViennaRNA-style
    dot-bracket notation, e.g. "..((...))..".

    Assumes `chosen_pairs` is already a valid nested (pseudoknot-free)
    non-conflicting structure; use `is_valid_secondary_structure` first
    if you need to check that.
    """
    n = len(sequence)
    dots = ["."] * n
    for (i, j) in chosen_pairs:
        dots[i] = "("
        dots[j] = ")"
    return "".join(dots)


def is_valid_secondary_structure(chosen_pairs) -> bool:
    """
    Sanity check that a set of pairs forms a legal secondary structure:
      - no base appears in more than one pair
      - no two pairs cross (no pseudoknots)
    Used as an independent check on QUBO/solver output -- if the QUBO
    penalty weights were too weak, this will catch it.
    """
    seen = set()
    plist = list(chosen_pairs)
    for (i, j) in plist:
        if i in seen or j in seen:
            return False
        seen.add(i)
        seen.add(j)
    for (i1, j1), (i2, j2) in combinations(plist, 2):
        if i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1:
            return False  # crossing => pseudoknot
    return True


def base_pair_distance(pairs_a, pairs_b) -> int:
    """
    Standard RNA structure comparison metric: size of the symmetric
    difference between two base-pair sets. 0 = identical structures.
    This is the metric ViennaRNA's own RNAdistance-style comparisons
    and most benchmarking papers use for "how close is my prediction
    to the reference structure".
    """
    set_a, set_b = set(pairs_a), set(pairs_b)
    return len(set_a.symmetric_difference(set_b))


def dot_bracket_to_pairs(structure: str):
    """Inverse of pairs_to_dot_bracket, for parsing ViennaRNA output."""
    stack = []
    pairs = []
    for idx, ch in enumerate(structure):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            i = stack.pop()
            pairs.append((i, idx))
    return sorted(pairs)


if __name__ == "__main__":
    # quick smoke test
    seq = "GGGAAACCC"
    cands = generate_candidate_pairs(seq, min_loop=3)
    print(f"sequence: {seq}  (n={len(seq)})")
    print(f"candidate pairs (min_loop=3): {cands}")
    chosen = [(0, 8)] if (0, 8) in cands else []
    print(f"dot-bracket for {chosen}: {pairs_to_dot_bracket(seq, chosen)}")
    print(f"valid structure? {is_valid_secondary_structure(chosen)}")
