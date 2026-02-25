"""
Problem 3b: CRN for Y_inf = 2^(log_2(X_0))
=============================================
Compose:
  1. Logarithm module: L_inf = log_2(X_0)
  2. Exponentiation module: Y_inf = 2^(L_inf)

Note: Y_inf = 2^(log_2(X_0)) = X_0, but the CRN must compute
this as shown (first log, then exp).

Combined CRN:
  Logarithm module:
    R1: B  --slow-->  A1 + B
    R2: A1 + 2X  --faster-->  C + X' + A1
    R3: 2C  --faster-->  C
    R4: A1  --fast-->  empty
    R5: X'  --medium-->  X
    R6: C   --medium-->  L

  Exponentiation module:
    R7:  L   --super_slow-->  A2
    R8:  A2 + Y  --faster-->  A2 + 2Y'
    R9:  A2  --fast-->  empty
    R10: Y'  --medium-->  Y

Rate hierarchy: super_slow << slow << medium << fast << faster

Initial: X=X_0, Y=1, B=1, all others=0

Verified using Gillespie SSA.
"""

import random
import math
import numpy as np


# Species indices
SPECIES = ['X', 'Xp', 'B', 'A1', 'C', 'L', 'A2', 'Y', 'Yp']
idx = {s: i for i, s in enumerate(SPECIES)}
N = len(SPECIES)

# Rates
SUPER_SLOW = 0.001
SLOW = 0.05
MEDIUM = 5.0
FAST = 100.0
FASTER = 5000.0


def build_reactions():
    """Build reaction list."""
    reactions = []

    # === Logarithm module: L = log_2(X) ===
    # R1: B --slow--> A1 + B  (B catalytic)
    reactions.append((SLOW,
                      [(idx['B'], 1)],
                      [(idx['A1'], 1), (idx['B'], 1)]))

    # R2: A1 + 2X --faster--> C + X' + A1  (A1 catalytic)
    reactions.append((FASTER,
                      [(idx['A1'], 1), (idx['X'], 2)],
                      [(idx['C'], 1), (idx['Xp'], 1), (idx['A1'], 1)]))

    # R3: 2C --faster--> C
    reactions.append((FASTER,
                      [(idx['C'], 2)],
                      [(idx['C'], 1)]))

    # R4: A1 --fast--> empty
    reactions.append((FAST,
                      [(idx['A1'], 1)],
                      []))

    # R5: X' --medium--> X
    reactions.append((MEDIUM,
                      [(idx['Xp'], 1)],
                      [(idx['X'], 1)]))

    # R6: C --medium--> L
    reactions.append((MEDIUM,
                      [(idx['C'], 1)],
                      [(idx['L'], 1)]))

    # === Exponentiation module: Y = 2^L ===
    # R7: L --super_slow--> A2
    reactions.append((SUPER_SLOW,
                      [(idx['L'], 1)],
                      [(idx['A2'], 1)]))

    # R8: A2 + Y --faster--> A2 + 2Y'
    reactions.append((FASTER,
                      [(idx['A2'], 1), (idx['Y'], 1)],
                      [(idx['A2'], 1), (idx['Yp'], 2)]))

    # R9: A2 --fast--> empty
    reactions.append((FAST,
                      [(idx['A2'], 1)],
                      []))

    # R10: Y' --medium--> Y
    reactions.append((MEDIUM,
                      [(idx['Yp'], 1)],
                      [(idx['Y'], 1)]))

    # Pre-compute net change vectors
    rxn_list = []
    for rate, reactants, products in reactions:
        net = [0] * N
        for sp, c in reactants:
            net[sp] -= c
        for sp, c in products:
            net[sp] += c
        rxn_list.append((rate, reactants, net))
    return rxn_list


def gillespie_sim(rxn_list, init_state, max_time=200000, max_steps=10000000):
    """Run one Gillespie simulation."""
    state = list(init_state)
    t = 0.0
    n_rxn = len(rxn_list)

    for step in range(max_steps):
        props = []
        for rate, reactants, net in rxn_list:
            p = rate
            for sp, c in reactants:
                x = state[sp]
                if x < c:
                    p = 0.0
                    break
                if c == 1:
                    p *= x
                elif c == 2:
                    p *= x * (x - 1) * 0.5
                else:
                    for j in range(c):
                        p *= (x - j) / (j + 1)
            props.append(p)

        a_total = sum(props)
        if a_total <= 0:
            break

        dt = -math.log(random.random()) / a_total
        t += dt
        if t > max_time:
            break

        r = random.random() * a_total
        cumsum = 0.0
        chosen = n_rxn - 1
        for i in range(n_rxn):
            cumsum += props[i]
            if r < cumsum:
                chosen = i
                break

        _, _, net = rxn_list[chosen]
        for i in range(N):
            state[i] += net[i]

    return state


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Problem 3b: Y_inf = 2^(log_2(X_0))")
    print("=" * 60)
    print()
    print("CRN Design (composition of Logarithm + Exponentiation):")
    print()
    print("  Logarithm module (L = log_2(X)):")
    print("    R1: B  --slow-->  A1 + B")
    print("    R2: A1 + 2X  --faster-->  C + X' + A1")
    print("    R3: 2C  --faster-->  C")
    print("    R4: A1  --fast-->  empty")
    print("    R5: X'  --medium-->  X")
    print("    R6: C   --medium-->  L")
    print()
    print("  Exponentiation module (Y = 2^L):")
    print("    R7:  L   --super_slow-->  A2")
    print("    R8:  A2 + Y  --faster-->  A2 + 2Y'")
    print("    R9:  A2  --fast-->  empty")
    print("    R10: Y'  --medium-->  Y")
    print()
    print("  Rate hierarchy: super_slow << slow << medium << fast << faster")
    print(f"  Rates: {SUPER_SLOW}, {SLOW}, {MEDIUM}, {FAST}, {FASTER}")
    print()
    print("  Initial: X=X_0, Y=1, B=1, all others=0")
    print("  Expected: Y_inf = 2^(log_2(X_0)) = X_0")
    print()

    rxn_list = build_reactions()

    test_cases = [2, 4, 8, 16, 32]  # powers of 2
    num_sims = 20

    print(f"Running {num_sims} Gillespie simulations per test case...")
    print()
    print(f"{'X0':>4s}  {'Expected Y':>12s}  {'Mean Y':>10s}  {'StdDev':>8s}  {'Samples':>30s}")
    print("-" * 70)

    for X0 in test_cases:
        expected = X0  # 2^(log2(X0)) = X0

        y_values = []
        for sim in range(num_sims):
            init = [0] * N
            init[idx['X']] = X0
            init[idx['Y']] = 1
            init[idx['B']] = 1

            final = gillespie_sim(rxn_list, init, max_time=200000)
            y_values.append(final[idx['Y']])

        mean_y = np.mean(y_values)
        std_y = np.std(y_values)
        sample_str = str(y_values[:5])

        print(f"{X0:4d}  {expected:12d}  {mean_y:10.2f}  {std_y:8.2f}  {sample_str}")

    print()
    print("Correctness argument:")
    print("  Phase 1 - Logarithm: The log module computes L = log_2(X_0).")
    print("  B catalytically produces A1 at a slow rate. Each A1 halves X")
    print("  (A1 + 2X -> C + X' + A1), collapses C pairs (2C -> C), then")
    print("  C -> L. After log_2(X_0) halvings, X=1 and L = log_2(X_0).")
    print()
    print("  Phase 2 - Exponentiation: Each L molecule is consumed (super-slow)")
    print("  producing A2. A2 catalytically doubles Y (A2 + Y -> A2 + 2Y'),")
    print("  then A2 disappears and Y' regenerates Y. After L doublings,")
    print("  Y = 2^L = 2^(log_2(X_0)) = X_0.")
    print()
    print("  The rate hierarchy ensures phase 1 completes before phase 2 begins.")
