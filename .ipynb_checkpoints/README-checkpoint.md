# mRNA Secondary Structure Prediction via Quantum Optimization
### WISER Global Quantum+AI Program 2026 — WISER × Moderna challenge

A working starter codebase for the challenge: formulate RNA secondary-structure
prediction as a QUBO, solve it classically (as an exact sanity check) and via
QAOA, benchmark against ViennaRNA's real thermodynamic MFE structures, and
analyze how quantum resource requirements scale with sequence length.

Everything under `src/` except `classical_reference.py` and
`quantum_solver_qaoa.py` has been **run and verified** in the environment this
was built in (see "Verified in this sandbox" below) — those two need
ViennaRNA and Qiskit respectively, which weren't installed there, so they're
written carefully but you should re-verify them the moment you have both
installed, before trusting any downstream result.

---

## 1. Background review (challenge deliverable #1)

**The biological problem.** mRNA is single-stranded, but it doesn't stay a
straight line — it folds back on itself, and complementary bases (A–U, G–C,
and the weaker G–U "wobble" pair) zip up into a pattern of stems, hairpin
loops, bulges, and multi-branch junctions. That folding pattern is the
*secondary structure*. It isn't a side detail: it affects how stable the
mRNA is against degradation, how efficiently ribosomes can translate it into
protein, and how manufacturable it is at scale — which is exactly why a
company building mRNA therapeutics cares about predicting and steering it
computationally rather than only observing it after the fact.

**The computational problem.** For a sequence of length *n*, the number of
distinct ways it can fold grows exponentially — Moderna's own intro deck
(linked below) puts it at roughly 2.3ⁿ candidate structures, so even a
modest 45-nt sequence already has an astronomically large search space.
Classical tools (ViennaRNA's `RNAfold`, chief among them) don't enumerate
that space; they use dynamic programming over thermodynamic
nearest-neighbor energy rules to find the Minimum Free Energy (MFE)
structure in roughly O(n³) time. That's a genuinely good algorithm — but it
only works efficiently because it assumes structures are non-crossing
("pseudoknot-free"). The moment you allow crossing base pairs (real,
biologically important pseudoknots), exact classical MFE prediction becomes
NP-hard, which is precisely the opening for combinatorial quantum
optimization to be relevant at all, rather than a solved problem looking for
an excuse to use a quantum computer.

**The quantum angle.** RNA folding can be written as a Quadratic
Unconstrained Binary Optimization (QUBO) problem: one binary variable per
candidate base pair, a reward for forming energetically favorable pairs, and
penalty terms enforcing that (a) no base pairs with more than one partner
and (b) pairs don't cross (in the pseudoknot-free baseline model). A QUBO is
exactly the object QAOA, VQE, and quantum annealers are built to consume —
which is why this is a legitimate, currently-active research question and
not a forced fit. It also isn't hypothetical: Moderna and IBM Quantum have
already published real results on this exact problem on real IBM hardware
(see References) — this challenge is asking you to reproduce a
small-scale version of genuine, ongoing research, not a toy exercise
invented for a summer program.

---

## 2. The QUBO formulation this codebase implements

Variables: one binary `x_k` per candidate pair `(i, j)`, where candidates
must be (a) chemically valid (A-U, G-C, or G-U) and (b) geometrically valid
(`j - i > min_loop`, so a hairpin closed at `(i,j)` has room for a real
loop). Enumerated in `sequence_utils.generate_candidate_pairs`.

Minimize:

```
H(x) = − pair_reward   · Σ_k x_k                              [reward: form pairs]
       − stack_bonus    · Σ_{(k,l) stacked}     x_k · x_l       [reward: prefer helices over isolated pairs]
       + penalty_overlap    · Σ_{(k,l) share an endpoint} x_k · x_l   [constraint: 1 partner max per base]
       + penalty_pseudoknot · Σ_{(k,l) crossing}          x_k · x_l   [constraint: no pseudoknots, in the baseline model]
```

"Stacked" means `(i,j)` and `(i+1, j-1)` are both candidates — i.e. forming
both would extend a helix by one base pair, which is the dominant
stabilizing term in real RNA thermodynamics (stacked pairs are much more
favorable than the same pairs in isolation). Building that in, even in this
simplified reward-based form, measurably changes what the optimizer prefers
— see `classical_solvers.py`'s `stack_bonus=0` vs `stack_bonus=1` comparison.

This is deliberately the same family of model reviewed as "model 1" in
Zaborniak et al. 2022 (maximize stacked base pairs, penalize overlaps and
pseudoknots) — a real, published, defensible starting point, not something
invented for this README. Implemented in `qubo_builder.py`, converted to
Ising form (for a hand-rolled circuit) by `qubo_to_ising`.

### From simplified QUBO to real thermodynamics — where the marks are

The formulation above treats every base pair as worth the same
`pair_reward`, which is a real simplification (a G-C pair and a G-U wobble
pair are not equally stable, and loop-size/composition matters too). Two
upgrade paths, in order of effort:

1. **Weight pairs individually.** Replace the uniform `pair_reward` with a
   per-pair-type weight (G-C > A-U > G-U is the right ordering) and a
   per-stack bonus that depends on which two pair-types are stacking,
   pulled from a nearest-neighbor parameter table. Still O(pairs) variables,
   just better linear/quadratic coefficients.
2. **Move from base pairs to stems.** Enumerate candidate stems (maximal
   runs of stacked pairs) instead of individual pairs, score each stem's
   contribution using ViennaRNA itself (`eval_structure` on the isolated
   stem, or by pulling loop-energy tables directly), and formulate a
   Maximum-Weight-Independent-Set QUBO over stems (compatible stems = no
   shared bases, no crossing). Fewer, richer variables; this is closer to
   what the IBM/Moderna papers and the Pauli Correlation Encoding paper
   describe as a "dense Maximum Independent Set-like" formulation. It's
   more work but is exactly the kind of "quality of algorithm design" +
   "creativity and novelty" upgrade the judging criteria reward.

---

## 3. Verified in this sandbox

Everything below was actually executed while building this (not just
written and assumed to work) — treat it as your project's first real
result, not a hypothetical.

**Correctness cross-check**, on the hand-built 9-nt toy hairpin `GGGAAACCC`:

| method | structure | notes |
|---|---|---|
| Nussinov DP (exact, `stack_bonus`-agnostic) | `(((...)))` | 3 pairs, the true optimum of the simplified "maximize pairs" objective |
| QUBO, solved by brute force, `stack_bonus=0` | `(((...)))` | **matches Nussinov exactly** — same pair count, same structure |
| QUBO, solved by brute force, `stack_bonus=1` | `(((...)))` | same structure (already optimal), lower energy (−5.0 vs −3.0) because both stacking bonuses now get counted |

This is the single most important check to reproduce first in your own
environment, and to screenshot/log for your submission: it proves the QUBO
formulation and the exact solver agree with an independently-derived
classical algorithm, *before* any quantum hardware or simulator noise is
anywhere in the picture. If you change the formulation later (different
penalty weights, added pair-type weighting, etc.), rerun this check.

**Qubit-scaling reality check**, on the actual 44-nt sequence given in the
challenge document:

| quantity | value |
|---|---|
| candidate pairs (= qubits, naive 1-var-per-pair encoding) | **313** |
| two-qubit interaction terms in the QUBO | 23,307 |
| Nussinov-optimal structure | `.((((.(((..(.(((((((....)))..).).)))))))))).` (15 pairs) |

313 qubits for a 44-nt sequence is a real, measured number from this
project's own code — and it's *much* higher than the ~10–80 qubits
Alevras et al. 2024 report for sequences up to 60 nt, or the ~156 qubits
Kumar et al. 2025 report at 60 nt. The gap is not a bug — it's exactly the
candidate-pair-filtering step described in §2 and implemented in
`classical_reference.filter_candidates_by_probability`: most geometrically-
valid candidate pairs have negligible equilibrium base-pair probability and
can be dropped before the QUBO is even built. **Running this filter and
reporting the before/after qubit count on your own sequences is a strong,
concrete result for deliverable #6** (and it's a direct, defensible
explanation for a number your judges may well ask about, since they'll
likely know the published qubit counts).

Full sweep from `scaling_analysis.py` (random synthetic sequences,
`min_loop=3`, naive encoding, no filtering — your post-filtering numbers
should sit well below this "before" baseline):

| n (nt) | qubits (naive) | 2-qubit terms | depth bound (Vizing, per QAOA layer) |
|---|---|---|---|
| 8 | 4 | 6 | 4 |
| 16 | 30 | 352 | 29 |
| 25 | 90 | 2,323 | 68 |
| 40 | 272 | 17,910 | 177 |
| 60 | 530 | 62,987 | 327 |
| 100 | 1,660 | 558,111 | 971 |

(Plot version saved to `results/scaling_analysis.png`, generated by the
same run.) The "depth bound" is a Vizing-theorem idealization (assumes
all-to-all qubit connectivity — real heavy-hex/Pegasus hardware needs SWAP
networks on top of this). Say that caveat explicitly in your writeup;
judges checking scaling-analysis depth will be checking whether you know
the difference between an idealized bound and a hardware-realistic one.

---

## 4. Project structure

```
mrna_quantum_folding/
├── README.md                    <- you are here
├── requirements.txt
├── data/
│   └── example_sequences.py     <- toy + challenge-provided + random synthetic sequences
├── results/
│   └── scaling_analysis.png     <- generated by src/scaling_analysis.py
└── src/
    ├── sequence_utils.py        <- complementarity rules, candidate pairs, dot-bracket <-> pairs
    ├── qubo_builder.py          <- builds the QUBO (and its Ising form) from a sequence
    ├── classical_solvers.py     <- Nussinov DP + exact brute-force QUBO solver (no quantum SDK needed)
    ├── scaling_analysis.py      <- qubit-count / circuit-depth vs. sequence length
    ├── classical_reference.py  <- ViennaRNA wrapper: MFE ground truth, energy eval, bpp filtering
    ├── quantum_solver_qaoa.py   <- from-scratch CVaR-QAOA solver (Qiskit + Qiskit Aer)
    └── benchmark_runner.py      <- ties it all together end-to-end, degrades gracefully if deps are missing
```

## 5. Setup

```bash
python3 -m venv venv && source venv/bin/activate     # or use Colab, which skips this
pip install -r requirements.txt
```

`ViennaRNA` and `qiskit`/`qiskit-aer` are genuine installs (compiled C++
bindings / real quantum SDKs) — if you're on Colab, `!pip install` works
directly with no extra setup; locally, ViennaRNA occasionally needs
`build-essential`/Xcode command-line tools present, since parts of it
compile from source depending on your platform's available wheels.

## 6. Running it

```bash
cd src

python3 sequence_utils.py        # smoke test
python3 qubo_builder.py          # inspect a QUBO for the 9-nt toy example
python3 classical_solvers.py     # the Nussinov <-> QUBO cross-check from §3
python3 scaling_analysis.py      # regenerates results/scaling_analysis.png
python3 benchmark_runner.py      # full pipeline over every sequence in data/example_sequences.py

# once ViennaRNA is installed:
python3 classical_reference.py

# once qiskit + qiskit-aer are installed:
python3 quantum_solver_qaoa.py
```

`benchmark_runner.run_pipeline(sequence, use_qaoa=True)` runs the complete
QUBO -> classical baseline -> QAOA -> ViennaRNA-comparison chain for one
sequence — this is your main loop for Task 4/5 once every dependency is in
place.

---

## 7. Roadmap against the actual challenge deliverables

- [x] **Background review** — §1 above; expand it with your own reading of
      the references below and put it in your final report nearly verbatim.
- [x] **Classical benchmark generation** — `classical_reference.get_mfe_structure`
      wraps exactly the `RNA.fold()` call shown in the challenge doc.
- [x] **Energy evaluation** — `classical_reference.evaluate_structure_energy`
      wraps `fold_compound(...).eval_structure(...)`, also exactly as shown.
- [x] **Quantum/quantum-inspired formulation** — §2 + `qubo_builder.py`.
- [ ] **Implementation and benchmarking** — run `benchmark_runner.run_pipeline`
      with `use_qaoa=True` across a range of `data/example_sequences.py`
      sequences (once qiskit is installed) plus a few of your own; report
      accuracy (`classical_reference.compare_to_reference`'s precision/
      recall/F1/base-pair-distance), energy gap to MFE, and runtime.
- [ ] **Scaling and quantum resource analysis** — §3's table is your
      starting point; extend `scaling_analysis.py` with the bpp-filtered
      qubit counts (the "after filtering" numbers) alongside the naive ones.
- [ ] **Final submission package** — see §8.

**Suggested order of attack, concretely:** get ViennaRNA installed and rerun
`benchmark_runner.py` today so you have real MFE references. In parallel,
install qiskit and confirm `quantum_solver_qaoa.py` reproduces the
`(((...)))` result on the 9-nt toy example — that's your "the quantum part
isn't broken" checkpoint before you point it at anything bigger. Only after
both of those independently check out should you start running QAOA against
real ViennaRNA references on `random_20nt_seed7` and up.

### Optional advanced tasks — realistic pick

Of the challenge's three optional tasks, **base-pair-probability filtering
+ requantifying the qubit-count gap from §3** is the best effort-to-payoff
ratio available to you: the code is already sketched in
`classical_reference.filter_candidates_by_probability`, it directly answers
a question your own numbers raise, and it's a genuine, citable technique
(McCaskill's partition-function algorithm) rather than a toy exercise. If
your team has bandwidth for a second one, comparing 2-3 different penalty
weightings (`stack_bonus`, `penalty_overlap`, `penalty_pseudoknot`) and
reporting how the recovered structure and required qubit count change is
cheap to run (everything needed is already in `qubo_builder.py` /
`classical_solvers.py`) and speaks directly to "quality of algorithm
design." Pseudoknot-aware modeling is the most interesting option
intellectually but also the most expensive — worth explicitly scoping out
("we exclude pseudoknots for X reason, see [Zaborniak et al.] for how a
graded penalty could reintroduce them") rather than attempting partially.

---

## 8. Putting the submission together

Per the judging criteria, structure your final package as:

1. **Report/slide deck** covering: problem formulation (§1-2, in your own
   words), methodology (your actual penalty weights and any deviations from
   this starter, justified), results (structures + energies + accuracy
   metrics vs. ViennaRNA, honestly including where it *doesn't* match and
   why you think that is), and the scaling analysis from §3 extended with
   your own filtered numbers.
2. **Code** — this repo, plus whatever you build on top, with a working
   `requirements.txt` and a README section on how to reproduce your
   headline numbers specifically (not just "run the scripts").
3. Keep the whole thing reproducible from a clean clone/Colab — that's an
   explicit judging criterion, not a nice-to-have, and it's the kind of
   thing that's easy to lose track of once you've been iterating for weeks
   in one long-running notebook.

---

## References

**Directly reused/cited by name in this codebase:**

- Alevras, Metkar, Yamamoto, Kumar, Friedhoff, Park, Takeori, LaDue, Davis,
  Galda (IBM Quantum + Moderna), *"mRNA secondary structure prediction using
  utility-scale quantum computers"*, 2024. [arXiv:2405.20328](https://arxiv.org/abs/2405.20328)
  — CVaR-VQE on IBM Eagle/Heron, up to 60 nt / 80 qubits, validated against
  the classical solver CPLEX.
- Kumar, Alevras, Metkar, Welling, Cade, Niesen, Friedhoff, Park, Shivpuje,
  LaDue, Davis, Galda (IBM Quantum + Moderna + Fermioniq), *"Towards
  secondary structure prediction of longer mRNA sequences using a
  quantum-centric optimization scheme"*, 2025.
  [arXiv:2505.05782](https://arxiv.org/abs/2505.05782) — CVaR-VQE +
  gauge transforms + local search, and a separate IQP-circuit scheme; up to
  156 qubits / 60 nt on IBM hardware.
- Zaborniak, et al., *"A QUBO model of the RNA folding problem optimized by
  variational hybrid quantum annealing"*, 2022.
  [arXiv:2208.04367](https://arxiv.org/abs/2208.04367) — the stem-based
  QUBO formulation this project's `qubo_builder.py` is modeled after;
  D-Wave hybrid annealing.
- Friedhoff, Metkar, Davis, Kumar, Galda (IBM Quantum + Moderna), *"Pauli
  Correlation Encoding for mRNA Secondary Structure Prediction:
  Problem-Aware Decoding for Dense-Constraint QUBOs"*, 2026.
  [arXiv:2605.20163](https://arxiv.org/abs/2605.20163) — compresses
  hundreds of QUBO variables onto a handful of qubits (e.g. 694-745
  variables into 23 qubits for ~104-nt sequences) via Pauli correlators;
  the most recent and most qubit-efficient of the four, and the one worth
  reading closely if your team attempts a genuinely novel encoding for the
  "optional advanced tasks."

**Tools:**
- ViennaRNA Python bindings: <https://viennarna-python.readthedocs.io/en/master/>
- Dot-bracket notation reference: <https://www.tbi.univie.ac.at/RNA/ViennaRNA/refman/io/rna_structures.html#dot-bracket-notation>
- Challenge intro deck (Galda): <https://alexgalda.github.io/quantum_mRNA_optimization/>

**Note on citing these in your report:** the four papers above form a
coherent, connected line of work by essentially the same Moderna/IBM team —
the challenge's own intro deck is built from the first of them. Citing all
four (not just the one the deck shows) and being specific about which
technique each contributes is a low-effort, high-signal way to demonstrate
you've actually read the literature rather than skimmed the challenge
document, which is exactly what "technical merit" and "communication
quality" are scoring for.
