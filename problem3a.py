"""
Problem 3a: CRN for Z_inf = X_0 * log_2(Y_0)
==================================================
Compose:
  1. Logarithm module: L_inf = log_2(Y_0)
  2. Multiplication module: Z_inf = X_0 * L_inf

Combined CRN (all reactions):
  Logarithm module:
    R1: B  --k1-->  A1 + B         (k1 = slow)
    R2: A1 + 2Y  --k2-->  C + Y' + A1  (k2 = faster)
    R3: 2C  --k3-->  C             (k3 = faster)
    R4: A1  --k4-->  empty         (k4 = fast)
    R5: Y'  --k5-->  Y             (k5 = medium)
    R6: C   --k6-->  L             (k6 = medium)

  Multiplication module:
    R7:  X   --k7-->   A2          (k7 = super_slow)
    R8:  A2 + L  --k8-->  A2 + L' + Z'  (k8 = faster)
    R9:  A2  --k9-->   empty       (k9 = fast)
    R10: L'  --k10-->  L           (k10 = medium)
    R11: Z'  --k11-->  Z           (k11 = medium)

Rate hierarchy: super_slow << slow << medium << fast << faster

Verified using Gillespie SSA.
"""

import random
import math
import numpy as np


# Species indices
SPECIES = ['X', 'Y', 'B', 'A1', 'C', 'Yp', 'L', 'A2', 'Lp', 'Zp', 'Z']
idx = {s: i for i, s in enumerate(SPECIES)}
N = len(SPECIES)

# Rates
SUPER_SLOW = 0.001
SLOW = 0.05
MEDIUM = 5.0
FAST = 100.0
FASTER = 5000.0


def build_reactions():
    """Build reaction list: each is (rate, reactants, net_change).
    reactants are lists of (species_idx, count).
    """
    reactions = []

    # R1: B --slow--> A1 + B  (B catalytic, net: produce A1)
    reactions.append((SLOW,
                      [(idx['B'], 1)],
                      [(idx['A1'], 1), (idx['B'], 1)]))

    # R2: A1 + 2Y --faster--> C + Y' + A1  (A1 catalytic)
    reactions.append((FASTER,
                      [(idx['A1'], 1), (idx['Y'], 2)],
                      [(idx['C'], 1), (idx['Yp'], 1), (idx['A1'], 1)]))

    # R3: 2C --faster--> C
    reactions.append((FASTER,
                      [(idx['C'], 2)],
                      [(idx['C'], 1)]))

    # R4: A1 --fast--> empty
    reactions.append((FAST,
                      [(idx['A1'], 1)],
                      []))

    # R5: Y' --medium--> Y
    reactions.append((MEDIUM,
                      [(idx['Yp'], 1)],
                      [(idx['Y'], 1)]))

    # R6: C --medium--> L
    reactions.append((MEDIUM,
                      [(idx['C'], 1)],
                      [(idx['L'], 1)]))

    # R7: X --super_slow--> A2
    reactions.append((SUPER_SLOW,
                      [(idx['X'], 1)],
                      [(idx['A2'], 1)]))

    # R8: A2 + L --faster--> A2 + L' + Z'
    reactions.append((FASTER,
                      [(idx['A2'], 1), (idx['L'], 1)],
                      [(idx['A2'], 1), (idx['Lp'], 1), (idx['Zp'], 1)]))

    # R9: A2 --fast--> empty
    reactions.append((FAST,
                      [(idx['A2'], 1)],
                      []))

    # R10: L' --medium--> L
    reactions.append((MEDIUM,
                      [(idx['Lp'], 1)],
                      [(idx['L'], 1)]))

    # R11: Z' --medium--> Z
    reactions.append((MEDIUM,
                      [(idx['Zp'], 1)],
                      [(idx['Z'], 1)]))

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
        # Compute propensities
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

        # Time step
        dt = -math.log(random.random()) / a_total
        t += dt
        if t > max_time:
            break

        # Choose reaction
        r = random.random() * a_total
        cumsum = 0.0
        chosen = n_rxn - 1
        for i in range(n_rxn):
            cumsum += props[i]
            if r < cumsum:
                chosen = i
                break

        # Apply
        _, _, net = rxn_list[chosen]
        for i in range(N):
            state[i] += net[i]

    return state


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("Problem 3a: Z_inf = X_0 * log_2(Y_0)")
    print("=" * 60)
    print()
    print("CRN Design (composition of Logarithm + Multiplication):")
    print()
    print("  Logarithm module (L = log_2(Y)):")
    print("    R1: B  --slow-->  A1 + B")
    print("    R2: A1 + 2Y  --faster-->  C + Y' + A1")
    print("    R3: 2C  --faster-->  C")
    print("    R4: A1  --fast-->  empty")
    print("    R5: Y'  --medium-->  Y")
    print("    R6: C   --medium-->  L")
    print()
    print("  Multiplication module (Z = X * L):")
    print("    R7:  X   --super_slow-->  A2")
    print("    R8:  A2 + L  --faster-->  A2 + L' + Z'")
    print("    R9:  A2  --fast-->  empty")
    print("    R10: L'  --medium-->  L")
    print("    R11: Z'  --medium-->  Z")
    print()
    print("  Rate hierarchy: super_slow << slow << medium << fast << faster")
    print(f"  Rates: {SUPER_SLOW}, {SLOW}, {MEDIUM}, {FAST}, {FASTER}")
    print()
    print("  Initial: X=X_0, Y=Y_0, B=1, all others=0")
    print()

    rxn_list = build_reactions()

    test_cases = [
        (4, 8),    # 4 * log2(8) = 4 * 3 = 12
        (3, 16),   # 3 * log2(16) = 3 * 4 = 12
        (5, 4),    # 5 * log2(4) = 5 * 2 = 10
        (2, 32),   # 2 * log2(32) = 2 * 5 = 10
        (6, 2),    # 6 * log2(2) = 6 * 1 = 6
        (10, 8),   # 10 * log2(8) = 10 * 3 = 30
    ]

    num_sims = 20

    print(f"Running {num_sims} Gillespie simulations per test case...")
    print()
    print(f"{'X0':>4s}  {'Y0':>4s}  {'Expected':>10s}  {'Mean Z':>10s}  {'StdDev':>8s}  {'Samples':>30s}")
    print("-" * 70)

    for X0, Y0 in test_cases:
        expected = X0 * math.log2(Y0)

        z_values = []
        for sim in range(num_sims):
            init = [0] * N
            init[idx['X']] = X0
            init[idx['Y']] = Y0
            init[idx['B']] = 1

            final = gillespie_sim(rxn_list, init, max_time=200000)
            z_values.append(final[idx['Z']])

        mean_z = np.mean(z_values)
        std_z = np.std(z_values)
        sample_str = str(z_values[:5])

        print(f"{X0:4d}  {Y0:4d}  {expected:10.1f}  {mean_z:10.2f}  {std_z:8.2f}  {sample_str}")

    print()
    print("Correctness argument:")
    print("  The logarithm module computes L = log_2(Y_0): B catalytically")
    print("  produces A1 (slowly). Each A1 halves Y via A1+2Y -> C+Y'+A1")
    print("  (the faster rate ensures all Y pairs are consumed per iteration).")
    print("  Pairs of C collapse (2C->C), then C->L produces exactly 1 L per")
    print("  halving. Y'->Y restores Y at half its count. After log_2(Y_0)")
    print("  iterations, Y=1 and L=log_2(Y_0).")
    print()
    print("  The multiplication module then runs at a super-slow rate to")
    print("  consume X one at a time. For each X molecule, A2 catalytically")
    print("  copies all L to Z' (A2+L -> A2+L'+Z'). Then L'->L restores L.")
    print("  After X_0 iterations, Z = X_0 * L = X_0 * log_2(Y_0).")
