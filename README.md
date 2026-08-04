# mRNA Secondary Structure Prediction via Quantum Optimization
### WISER Global Quantum+AI Program 2026 — WISER × Moderna challenge

A QUBO formulation of RNA secondary-structure prediction, solved classically
(as an exact correctness check), via QAOA (Qiskit + Qiskit Aer), and
benchmarked against ViennaRNA's real thermodynamic MFE structures — including
a working demonstration on the literal example sequence given in the
challenge document, reduced from 313 to 26 qubits via base-pair-probability
filtering.

---

## Team & contributions

Solo participant: **[Uma Maheswar Reddy V]**, covering all suggested team roles —
Project Lead (primary contact), Research Lead, Technical/Development Lead,
Data Analysis Lead, and Documentation Lead.

## Use of generative AI

This project's codebase (QUBO formulation, classical solvers, QAOA
implementation, scaling analysis, and experiment design) was developed with
substantial assistance from Claude (Anthropic) — including literature
review of the cited papers, code scaffolding, and drafting of this README.
Every classical component (QUBO construction, Nussinov DP, brute-force
solver, scaling analysis) was executed and cross-validated before being
trusted. The QAOA and ViennaRNA-dependent components were written without
direct execution access in this environment and were independently run and
verified by [Uma Maheswar Reddy V] before being used for real results — starting with
reproducing the exact `(((...)))` structure on a 9-nt toy example as a
correctness checkpoint (see §3) before trusting anything at larger scale.
Several real bugs were caught this way during development, not by
inspection: a QUBO-vs-Nussinov cross-check that initially disagreed, a
diagnostic timing script that silently mis-divided by a scipy-adjusted
iteration count (caught by a solver warning and independently confirmed),
and a plotting filter that silently dropped exactly the two sequences the
experiment existed to examine. One factual error introduced by an earlier
AI assistant (a cited qubit count of 354 instead of the correct 156, from
Kumar et al. 2025) was caught by checking the primary source directly.
Every technical claim in this report is grounded in a script in this repo
and a CSV in `results/` that a reviewer can rerun.

---

## 1. Background review (challenge deliverable #1)

**The biological problem.** mRNA is single-stranded, but it folds back on
itself, and complementary bases (A–U, G–C, and the weaker G–U "wobble" pair)
zip up into a pattern of stems, hairpin loops, bulges, and multi-branch
junctions — the *secondary structure*. That pattern affects mRNA stability,
translation efficiency, and manufacturability, which is why a company
building mRNA therapeutics needs to predict and steer it computationally
rather than only observing it after synthesis.

**The computational problem.** The number of possible foldings for a
sequence of length *n* grows exponentially (roughly 2.3ⁿ, per the challenge's
own intro deck). Classical tools (ViennaRNA's `RNAfold`) don't enumerate that
space; they use dynamic programming over thermodynamic nearest-neighbor
rules to find the Minimum Free Energy (MFE) structure in roughly O(n³) time
— but only because they assume non-crossing ("pseudoknot-free") structures.
Allowing crossing pairs (real, biologically important pseudoknots) makes
exact prediction NP-hard, which is the actual opening for quantum
optimization here.

**The quantum angle.** RNA folding can be written as a QUBO: one binary
variable per candidate base pair, rewards for energetically favorable pairs,
penalties for structural conflicts. This is a legitimate, currently active
research question, not a forced fit — Moderna and IBM Quantum have already
published real results on this exact problem on real IBM hardware (see
References). This project reproduces a small-to-medium-scale version of
that line of research, including one result on the challenge's own example
sequence.

---

## 2. The QUBO formulation this codebase implements

One binary variable `x_k` per candidate base pair `(i,j)`: chemically valid
(A-U, G-C, or G-U) and geometrically valid (`j - i > min_loop`).

Minimize:

```
H(x) = − pair_reward   · Σ_k x_k                              [reward: form pairs]
       − stack_bonus    · Σ_{(k,l) stacked}     x_k · x_l       [reward: prefer helices]
       + penalty_overlap    · Σ_{(k,l) share an endpoint} x_k · x_l   [constraint: 1 partner max]
       + penalty_pseudoknot · Σ_{(k,l) crossing}          x_k · x_l   [constraint: no pseudoknots]
```

Same family as "model 1" in Zaborniak et al. 2022. Deliberately simplified
— §7 below quantifies exactly where that simplification costs accuracy, and
by how much, rather than leaving it as a caveat.

`qubo_builder.build_qubo()` also accepts `restrict_to_pairs`, letting a QUBO
be built over a pre-filtered candidate set instead of the full naive one —
this is what makes §4.5 possible.

---

## 3. Correctness, established before anything else

On the 9-nt toy hairpin `GGGAAACCC`: an independent classical DP (Nussinov),
an exact brute-force QUBO solve, and QAOA all agree on `(((...)))` — and it
matches ViennaRNA's real thermodynamic MFE exactly. This was the first thing
verified, deliberately, before trusting any larger-scale result: it
separates "is the formulation correct" from "did the optimizer do a good
job," which are different questions easy to conflate without an independent
check.

---

## 4. Results

### 4.1 Qubit scaling and the memory wall

Naive encoding (1 qubit per candidate pair) needs **313 qubits** for the
44-nt challenge example sequence — far more than the 10-156 qubits published
work reports at similar lengths (Alevras et al. 2024; Kumar et al. 2025).
Exact statevector simulation cost is `2^n × 16 bytes`; at 31 qubits that's
already 34GB, more than qBraid's Large tier (25GB) provides. This isn't a
configurable limit — it's a real ceiling of exact simulation on a single
machine (`results/qubit_memory_wall.png`), and it directly motivated §4.5.

### 4.2 QAOA reliability (`results/exp1_reliability.csv/png`)

5 seeds × 11 sequences, comparing QAOA's best-found energy against the exact
QUBO ground state where computable (≤20 qubits):

| qubit range | success rate |
|---|---|
| ≤16 qubits (7 sequences) | **100%** |
| 21 qubits (ground state unverifiable) | 100% (self-consistent) |
| 23 qubits (2 sequences) | **40% and 80%** |

Reliability holds perfectly to ~16 qubits, then visibly degrades approaching
the ~24-qubit working ceiling — a measured scaling result, not a
theoretical estimate.

### 4.3 Does circuit depth help? Does CVaR alpha matter? (`exp2_depth`, `exp4_cvar_alpha`)

Tested specifically on the two hard sequences from §4.2 (`stem_bulge_15nt`,
`random_15nt_seedB`), since testing only the easy sequences (100% success
regardless of settings) can't show whether either parameter matters.

- **Depth**: mixed result, honestly reported rather than oversold.
  `stem_bulge_15nt` improved monotonically with more reps (best energy
  −4→−5→−6); `random_15nt_seedB` got worse (−6→−4→−2). This is `n_seeds=1`
  data — a real effect or seed variance can't be distinguished with one
  seed per depth level. Flagged as an open question, not resolved here.
- **CVaR alpha**: cleaner result. On the easy sequence, all four alpha
  values (0.1-0.75) converge to the ground state by iteration ~15-20. On
  the hard sequence, **none fully converge within 25 iterations**, but
  low alpha (0.1, 0.25) trends toward substantially better final values
  than high alpha (0.5, 0.75, which stays essentially flat). Low alpha's
  advantage doesn't just persist on harder problems — it grows.

### 4.4 Full sweep against real ViennaRNA MFE (`experiments_20260802_160711.csv`)

17 sequences, ViennaRNA confirmed working. Two clear, opposite patterns:

- **5 of 5 designed hairpin sequences** — including one at 23 qubits — QAOA
  reproduced ViennaRNA's exact real MFE structure. `f1 = 1.0`,
  `exact_match = True`, all five.
- **9 of 10 random sequences**: ViennaRNA predicts **no folding at all**
  (MFE energy = 0.0, structure = all dots) — but the simplified QUBO always
  finds pairs, because nothing in the objective penalizes the entropic
  cost of closing a loop; every additional pair is unconditionally
  rewarded. This is a systematic, now well-evidenced limitation of the
  pair-maximizing objective, not an edge case.

  Two "QAOA mismatch" rows in this sweep (`random_12nt_seedA`,
  `random_18nt_seedB`) are not real failures on inspection — both found a
  *different* structure with the *identical* QUBO energy as the true
  optimum (degenerate ground states). Comparing energies rather than
  structure strings gives 9/9 correct on the comparable sequences, not 7/9.

### 4.5 The capstone result: the challenge's own example sequence (`filtered_qubo_44nt_result.csv`)

Base-pair-probability filtering (McCaskill partition function via
ViennaRNA) reduces `challenge_example_44nt` from 313 candidate pairs to
**26** — a 91.7% reduction — small enough to actually run QAOA on. Result:

- QAOA found a 14-pair structure; ViennaRNA's real MFE also has 14 pairs.
- **11 of 14 pairs match exactly** — the entire outer helical scaffold,
  correct down to the nucleotide.
- `f1 = 0.786`, `base_pair_distance = 6`.
- The 3-pair disagreement is not scattered noise — it's localized to one
  region, with a specific, identifiable cause: ViennaRNA closes a clean,
  unbulged 3-pair stack plus one outer pair; QAOA instead extends the
  stack further inward through two single-nucleotide bulges. Real
  thermodynamics penalizes bulges explicitly (Turner nearest-neighbor
  parameters); this QUBO's `stack_bonus` rewards perfect adjacency but
  never penalizes a bulge, so the optimizer correctly solved the *wrong
  cost function* in this one region. This is the §2 simplification made
  concrete and precisely diagnosed, on the challenge's own sequence.

---

## 5. Project structure

```
wiser-moderna-mrna-quantum-folding/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── example_sequences.py       <- 17 sequences: toy, curated, random, challenge-provided
├── results/
│   ├── exp1_reliability.csv/.png
│   ├── exp2_depth.csv/.png
│   ├── exp3_penalties.csv/.png
│   ├── exp4_cvar_alpha.csv/.png
│   ├── experiments_20260802_160711.csv   <- authoritative full sweep, ViennaRNA-verified
│   ├── filtered_qubo_44nt_result.csv     <- the capstone result, §4.5
│   ├── qubit_memory_wall.png
│   └── scaling_analysis.png
└── src/
    ├── sequence_utils.py           <- complementarity, candidate pairs, dot-bracket <-> pairs
    ├── qubo_builder.py             <- QUBO construction, incl. restrict_to_pairs
    ├── classical_solvers.py        <- Nussinov DP + exact brute-force QUBO solver
    ├── classical_reference.py      <- ViennaRNA wrapper, bpp filtering, comparison metrics
    ├── quantum_solver_qaoa.py      <- from-scratch CVaR-QAOA (Qiskit + Aer), memory-aware
    ├── scaling_analysis.py         <- qubit-count / circuit-depth vs. sequence length
    ├── benchmark_runner.py         <- one-sequence end-to-end pipeline
    ├── run_experiments.py          <- full sweep across all sequences -> timestamped CSV
    ├── experiments.py              <- Experiments 1-4 (reliability, depth, penalties, alpha)
    └── filtered_qubo_experiment.py <- §4.5: filtered QUBO on the challenge sequence
```

## 6. Setup and running

```bash
pip install -r requirements.txt
```

```bash
cd src
python3 benchmark_runner.py                          # single-sequence pipeline demo
python3 run_experiments.py --qaoa --max-qubits-qaoa 24 --ram-gb 4   # full sweep
python3 experiments.py                                # Experiments 1-4
python3 filtered_qubo_experiment.py --qaoa --maxiter 25 --ram-gb 4  # §4.5 capstone
```

`--ram-gb` should match your compute tier (Small=4, Medium=8, Large=25) —
`quantum_solver_qaoa.py` uses it to refuse circuits that would need more
memory than available, with a clear message, before wasting time on one
that can't finish.

---

## 7. Limitations and future work

1. **Bulge-blind reward model** (§4.5): no penalty for interrupting a
   stack. Fix: incorporate real Turner nearest-neighbor loop parameters
   into the linear/quadratic coefficients instead of uniform rewards.
2. **Over-predicts structure on unstructured sequences** (§4.4): the
   objective has no mechanism to prefer zero pairs, ever. Fix: same as
   above — real thermodynamics' entropic loop penalty is what's missing.
3. **Depth's effect on hard instances is unresolved** (§4.3): single-seed
   data suggested opposite trends on two different hard sequences;
   distinguishing signal from seed variance needs multiple seeds per
   depth level, not attempted here due to time constraints (23-qubit
   circuits cost ~10-40s per COBYLA iteration; a full multi-seed sweep
   was budgeted against report-writing time instead).
4. **Pseudoknots excluded** from the baseline model, as stated in §2 —
   not attempted here; Zaborniak et al. 2022 describes a graded-penalty
   approach that could reintroduce them.
5. **Qubit-efficient encoding** (Pauli Correlation Encoding, Friedhoff et
   al. 2026) would extend well past the current ~26-qubit practical
   ceiling; base-pair-probability filtering (§4.5) is a first step in
   the same direction but a simpler one.

---

## References

- Alevras, Metkar, Yamamoto, Kumar, Friedhoff, Park, Takeori, LaDue, Davis,
  Galda (IBM Quantum + Moderna), *"mRNA secondary structure prediction using
  utility-scale quantum computers"*, 2024. [arXiv:2405.20328](https://arxiv.org/abs/2405.20328)
- Kumar, Alevras, Metkar, Welling, Cade, Niesen, Friedhoff, Park, Shivpuje,
  LaDue, Davis, Galda (IBM Quantum + Moderna + Fermioniq), *"Towards
  secondary structure prediction of longer mRNA sequences using a
  quantum-centric optimization scheme"*, 2025. [arXiv:2505.05782](https://arxiv.org/abs/2505.05782)
- Zaborniak et al., *"A QUBO model of the RNA folding problem optimized by
  variational hybrid quantum annealing"*, 2022. [arXiv:2208.04367](https://arxiv.org/abs/2208.04367)
- Friedhoff, Metkar, Davis, Kumar, Galda (IBM Quantum + Moderna), *"Pauli
  Correlation Encoding for mRNA Secondary Structure Prediction"*, 2026.
  [arXiv:2605.20163](https://arxiv.org/abs/2605.20163)
- ViennaRNA: <https://viennarna-python.readthedocs.io/en/master/>
- Challenge intro deck (Galda): <https://alexgalda.github.io/quantum_mRNA_optimization/>