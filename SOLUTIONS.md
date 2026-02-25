# EE 5393 Homework #1 — Sample Solutions

**Course:** EE 5393, Circuits, Computation, and Biology, Winter 2026  
**Due:** Wed., Feb. 24, 2026

---

## Problem 1(a): Estimating Outcome Probabilities

**Setup:** Three reactions with mass-action kinetics:
- R1: 2X₁ + X₂ → 4X₃ (k₁ = 1)
- R2: X₁ + 2X₃ → 3X₂ (k₂ = 2)
- R3: X₂ + X₃ → 2X₁ (k₃ = 3)

Initial state: S = [110, 26, 55]. Outcomes: C1 (x₁ ≥ 150), C2 (x₂ < 10), C3 (x₃ > 100).

**Propensities (mass-action with combinatorial factors):**
- a₁ = k₁ · C(x₁, 2) · x₂ = 1 · x₁(x₁−1)/2 · x₂
- a₂ = k₂ · x₁ · C(x₃, 2) = 2 · x₁ · x₃(x₃−1)/2
- a₃ = k₃ · x₂ · x₃ = 3 · x₂ · x₃

**Results (10,000 simulations, 10,000 max steps):**

| Outcome | Probability |
|---------|-------------|
| Pr(C1) = Pr(x₁ ≥ 150) | ≈ 0.0000 |
| Pr(C2) = Pr(x₂ < 10)  | ≈ 0.0000 |
| Pr(C3) = Pr(x₃ > 100) | ≈ 1.0000 |

**Why this is correct:**

1. **Verification:** From S = [3, 3, 3], the first-step probabilities are p₁ = a₁/(a₁+a₂+a₃) = 3/(3+9+27) ≈ 1/6, p₂ ≈ 1/3, p₃ ≈ 1/2. Simulation confirms: p₁ = 0.166, p₂ = 0.332, p₃ = 0.501.

2. **Dynamics:** From [110, 26, 55], x₁ drops rapidly (110 → ~8 by step 100) because R1 (consuming 2x₁) dominates when x₁ is large. x₂ grows steadily (R2 produces 3x₂). x₃ rises above 100, triggering C3 in every simulation.

3. **Conservation:** R1 adds 1 molecule per firing; R2, R3 conserve total. The total only increases, supporting the growth pattern.

**Code:** `problem1a.py`

---

## Problem 1(b): Mean and Variance after 7 Steps

**Setup:** Same reactions. Initial state: S = [9, 8, 7]. Compute E[Xᵢ] and Var(Xᵢ) after exactly 7 discrete reaction firings.

**Results (500,000 simulations):**

| Species | Mean | Variance | Std Dev |
|---------|------|----------|---------|
| X₁ | 5.833 | 5.883 | 2.426 |
| X₂ | 12.499 | 8.940 | 2.990 |
| X₃ | 7.803 | 8.942 | 2.990 |

**Why this is correct:**

1. **Verification:** From [3, 3, 3] after 1 step: E[X₁] = 3.325 (expected 10/3 ≈ 3.333), E[X₂] = 3.332 (expected 10/3), E[X₃] = 2.513 (expected 5/2).

2. **Conservation check:** Initial total = 24, mean total after 7 steps = 26.14, consistent with R1 firing ~2.14 times on average (each adding 1 molecule).

3. **Interpretation:** X₁ decreases from 9 to ~5.8 (consumed by R1, R2; partially replenished by R3). X₂ increases from 8 to ~12.5 (net production by R2). X₃ stays roughly stable at ~7.8 (R1 produces, R2/R3 consume, approximately balanced).

**Code:** `problem1b.py`

---

## Problem 2: Lambda Bacteriophage Decision

**Setup:** Simulate the lambda phage genetic switch (121 reactions, 63 species) from:
- Reaction model: http://mriedel.ece.umn.edu/files/lambda.r
- Initial values: http://mriedel.ece.umn.edu/files/lambda.in

Stealth (lysogeny): cI₂ > 145. Hijack (lytic): Cro₂ > 55.

**Results (200 simulations per MOI, max_time = 1000, Numba JIT):**

| MOI | P(Stealth) | P(Hijack) | P(Undecided) |
|-----|-----------|-----------|-------------|
| 1   | 0.000 | 0.070 | 0.930 |
| 2   | 0.000 | 0.620 | 0.380 |
| 3   | 0.000 | 0.710 | 0.290 |
| 4   | 0.005 | 0.720 | 0.275 |
| 5   | 0.000 | 0.750 | 0.250 |
| 6   | 0.060 | 0.780 | 0.160 |
| 7   | 0.145 | 0.730 | 0.125 |
| 8   | 0.215 | 0.740 | 0.045 |
| 9   | 0.255 | 0.740 | 0.005 |
| 10  | 0.220 | 0.770 | 0.010 |

**Why this is correct:**

1. **Biological interpretation:** The key trend is that P(Stealth) increases with MOI:
   - At low MOI (1–3): stealth probability is ~0. Few phage genomes produce little cII/cIII, so the cI repressor is not activated. The Cro pathway (hijack/lytic) dominates.
   - At high MOI (7–10): P(Stealth) rises to 15–25%. More phage genomes → more cII → PRE promoter activation → cI production → cI₂ accumulation → lysogeny.

2. **Undecided fraction:** At low MOI, many simulations are undecided because the molecule counts are too low for either threshold to be reached within the simulation time. With longer simulation times, more would resolve.

3. **Implementation:** Uses Numba JIT compilation for the Gillespie inner loop (~50× speedup over pure Python). Propensities computed using mass-action kinetics with combinatorial factors C(x, n) for multi-molecular reactants.

**Code:** `problem2_numba.py` (requires numpy, numba)

---

## Problem 3(a): CRN for Z∞ = X₀ · log₂(Y₀)

**Design:** Compose the Logarithm and Multiplication modules.

### Logarithm module (L∞ = log₂(Y₀)):
```
R1:  B         --slow-->     A₁ + B
R2:  A₁ + 2Y  --faster-->   C + Y' + A₁
R3:  2C        --faster-->   C
R4:  A₁       --fast-->     ∅
R5:  Y'        --medium-->   Y
R6:  C         --medium-->   L
```

### Multiplication module (Z∞ = X₀ · L∞):
```
R7:  X         --super_slow-->  A₂
R8:  A₂ + L   --faster-->      A₂ + L' + Z'
R9:  A₂       --fast-->        ∅
R10: L'        --medium-->      L
R11: Z'        --medium-->      Z
```

**Rate hierarchy:** super_slow ≪ slow ≪ medium ≪ fast ≪ faster

**Initial conditions:** X = X₀, Y = Y₀, B = 1, all others = 0.

**Results (20 Gillespie simulations per test case):**

| X₀ | Y₀ | Expected | Mean Z | StdDev |
|----|-----|----------|--------|--------|
| 4 | 8 | 12.0 | 11.95 | 1.86 |
| 3 | 16 | 12.0 | 11.70 | 1.93 |
| 5 | 4 | 10.0 | 10.20 | 0.87 |
| 2 | 32 | 10.0 | 10.55 | 1.72 |
| 6 | 2 | 6.0 | 6.25 | 0.94 |
| 10 | 8 | 30.0 | 31.00 | 4.35 |

**Why this is correct:**

1. **Phase 1 (Logarithm):** B is a catalyst that slowly produces A₁. Each A₁ halves Y by converting pairs 2Y → C + Y' (catalytically, since A₁ is preserved). The 2C → C reaction reduces C to 1, then C → L contributes exactly 1 to L. Y' → Y regenerates Y at half its previous count. After log₂(Y₀) halvings, Y = 1 and L = log₂(Y₀). The halving stops because A₁ + 2Y requires at least 2 molecules of Y.

2. **Phase 2 (Multiplication):** The super-slow rate ensures the logarithm module has completed before multiplication begins. X molecules are consumed one at a time. For each X, A₂ catalytically copies all L molecules to Z' (A₂ + L → A₂ + L' + Z'). Then L' → L restores L and Z' → Z accumulates. After X₀ iterations, Z = X₀ · log₂(Y₀).

3. **The rate hierarchy** ensures temporal separation between phases and between iterations within each phase.

**Code:** `problem3a.py`

---

## Problem 3(b): CRN for Y∞ = 2^(log₂(X₀))

**Design:** Compose the Logarithm and Exponentiation modules.

### Logarithm module (L∞ = log₂(X₀)):
```
R1:  B         --slow-->     A₁ + B
R2:  A₁ + 2X  --faster-->   C + X' + A₁
R3:  2C        --faster-->   C
R4:  A₁       --fast-->     ∅
R5:  X'        --medium-->   X
R6:  C         --medium-->   L
```

### Exponentiation module (Y∞ = 2^(L∞)):
```
R7:  L         --super_slow-->  A₂
R8:  A₂ + Y   --faster-->      A₂ + 2Y'
R9:  A₂       --fast-->        ∅
R10: Y'        --medium-->      Y
```

**Rate hierarchy:** super_slow ≪ slow ≪ medium ≪ fast ≪ faster

**Initial conditions:** X = X₀, Y = 1, B = 1, all others = 0.

**Results (20 Gillespie simulations per test case):**

| X₀ | Expected Y | Mean Y | StdDev |
|----|-----------|--------|--------|
| 2 | 2 | 2.10 | 0.54 |
| 4 | 4 | 4.40 | 1.02 |
| 8 | 8 | 8.75 | 1.58 |
| 16 | 16 | 18.80 | 5.82 |
| 32 | 32 | 40.35 | 12.29 |

**Why this is correct:**

1. **Phase 1 (Logarithm):** Identical to 3(a). Computes L = log₂(X₀) by repeated halving.

2. **Phase 2 (Exponentiation):** Each L molecule is consumed at the super-slow rate, producing A₂. A₂ catalytically doubles Y (A₂ + Y → A₂ + 2Y'), then A₂ disappears and Y' → Y regenerates at double the count. After L doublings, Y = 2^L = 2^(log₂(X₀)) = X₀.

3. **Note on noise:** The exponential module amplifies noise because each doubling doubles any error from the previous step. For X₀ = 32 (5 doublings), the relative variance is larger. This is inherent to the stochastic setting and would vanish in the deterministic limit of infinite rate separation.

4. **Mathematical guarantee:** In the limit of infinite rate separation, the composition is exact: Y∞ = 2^(log₂(X₀)) = X₀ for all powers of 2.

**Code:** `problem3b.py`

---

## Files Summary

| File | Problem | Description |
|------|---------|-------------|
| `problem1a.py` | 1(a) | Gillespie SSA for outcome probabilities |
| `problem1b.py` | 1(b) | Mean/variance after 7 steps |
| `problem2_numba.py` | 2 | Lambda phage (Numba JIT, fast) |
| `problem3a.py` | 3(a) | CRN: Z = X · log₂(Y) |
| `problem3b.py` | 3(b) | CRN: Y = 2^(log₂(X)) |

All code requires Python 3.12+. Problem 2 requires `numpy` and `numba`. Problem 3 requires `numpy`.
