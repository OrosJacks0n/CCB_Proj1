"""
Problem 1(b): Mean and Variance of X1, X2, X3 after 7 steps
=============================================================
Given reactions:
  R1: 2X1 + X2 -> 4X3,  k1 = 1
  R2: X1 + 2X3 -> 3X2,  k2 = 2
  R3: X2 + X3  -> 2X1,  k3 = 3

Starting state: S = [9, 8, 7]
Compute the mean and variance for X1, X2, X3 after 7 steps.

Method: Monte Carlo via Gillespie's SSA (discrete steps only, no time).
Each "step" is one reaction firing event.
"""

import random
import math
from collections import defaultdict


def gillespie_step_discrete(state):
    """
    Perform one discrete step: choose which reaction fires based on propensities.
    
    Propensities:
      a1 = x1*(x1-1)/2 * x2       (R1: 2X1 + X2 -> 4X3)
      a2 = x1 * x3*(x3-1)         (R2: X1 + 2X3 -> 3X2)
      a3 = 3 * x2 * x3            (R3: X2 + X3  -> 2X1)
    
    State changes:
      R1: [-2, -1, +4]
      R2: [-1, +3, -2]
      R3: [+2, -1, -1]
    """
    x1, x2, x3 = state
    
    a1 = (x1 * (x1 - 1) / 2) * x2
    a2 = x1 * x3 * (x3 - 1)
    a3 = 3 * x2 * x3
    
    a_total = a1 + a2 + a3
    
    if a_total <= 0:
        return list(state)  # No reaction can fire
    
    r = random.random() * a_total
    
    if r < a1:
        return [x1 - 2, x2 - 1, x3 + 4]
    elif r < a1 + a2:
        return [x1 - 1, x2 + 3, x3 - 2]
    else:
        return [x1 + 2, x2 - 1, x3 - 1]


def simulate_n_steps(initial_state, n_steps):
    """Run exactly n_steps of Gillespie's algorithm (discrete steps)."""
    state = list(initial_state)
    for _ in range(n_steps):
        state = gillespie_step_discrete(state)
    return state


def compute_mean_variance(initial_state, n_steps, num_simulations):
    """
    Run many simulations and compute mean and variance for each species
    after n_steps discrete steps.
    """
    # Collect final values for each species
    x1_vals = []
    x2_vals = []
    x3_vals = []
    
    for i in range(num_simulations):
        final_state = simulate_n_steps(initial_state, n_steps)
        x1_vals.append(final_state[0])
        x2_vals.append(final_state[1])
        x3_vals.append(final_state[2])
        
        if (i + 1) % 50000 == 0:
            print(f"  Completed {i+1}/{num_simulations} simulations...")
    
    # Compute mean and variance for each
    def mean(vals):
        return sum(vals) / len(vals)
    
    def variance(vals):
        m = mean(vals)
        return sum((v - m) ** 2 for v in vals) / len(vals)
    
    results = {
        'X1': {'mean': mean(x1_vals), 'variance': variance(x1_vals)},
        'X2': {'mean': mean(x2_vals), 'variance': variance(x2_vals)},
        'X3': {'mean': mean(x3_vals), 'variance': variance(x3_vals)},
    }
    
    return results


if __name__ == "__main__":
    random.seed(42)
    
    print("=" * 60)
    print("Problem 1(b): Mean and Variance after 7 Steps")
    print("=" * 60)
    print()
    print("Reactions:")
    print("  R1: 2X1 + X2 -> 4X3,  k1 = 1")
    print("  R2: X1 + 2X3 -> 3X2,  k2 = 2")
    print("  R3: X2 + X3  -> 2X1,  k3 = 3")
    print()
    print("Initial state: S = [9, 8, 7]")
    print("Number of steps: 7")
    print()
    
    initial_state = [9, 8, 7]
    n_steps = 7
    num_sims = 500000  # More simulations for better accuracy
    
    print(f"Running {num_sims} simulations...")
    print()
    
    results = compute_mean_variance(initial_state, n_steps, num_sims)
    
    print()
    print("Results after 7 steps:")
    print(f"  X1: Mean = {results['X1']['mean']:.4f}, Variance = {results['X1']['variance']:.4f}")
    print(f"  X2: Mean = {results['X2']['mean']:.4f}, Variance = {results['X2']['variance']:.4f}")
    print(f"  X3: Mean = {results['X3']['mean']:.4f}, Variance = {results['X3']['variance']:.4f}")
    print()
    
    # Also show the standard deviations
    print("Standard deviations:")
    print(f"  X1: StdDev = {math.sqrt(results['X1']['variance']):.4f}")
    print(f"  X2: StdDev = {math.sqrt(results['X2']['variance']):.4f}")
    print(f"  X3: StdDev = {math.sqrt(results['X3']['variance']):.4f}")
    print()
    
    # Verification: After 1 step from [3,3,3], expected values should be:
    # E[X1] = 1*(1/6) + 2*(1/3) + 5*(1/2) = 1/6 + 2/3 + 5/2 = 1/6 + 4/6 + 15/6 = 20/6 = 10/3
    print("-" * 60)
    print("Verification: 1 step from [3,3,3]")
    verify_results = compute_mean_variance([3, 3, 3], 1, 200000)
    expected_x1 = 1/6 + 2*1/3 + 5*1/2  # = 10/3 ~ 3.3333
    expected_x2 = 2/6 + 6*1/3 + 2*1/2  # = 1/3 + 2 + 1 = 10/3 ~ 3.3333
    expected_x3 = 7/6 + 1*1/3 + 2*1/2  # = 7/6 + 2/6 + 6/6 = 15/6 = 5/2 = 2.5
    print(f"  X1: Mean = {verify_results['X1']['mean']:.4f} (expected {expected_x1:.4f})")
    print(f"  X2: Mean = {verify_results['X2']['mean']:.4f} (expected {expected_x2:.4f})")
    print(f"  X3: Mean = {verify_results['X3']['mean']:.4f} (expected {expected_x3:.4f})")
    
    # Conservation check: For each sim, total molecules should be
    # initial: 9+8+7 = 24. R1 adds 1, R2 and R3 preserve total.
    print()
    print("Conservation check (total molecules):")
    print(f"  Initial: {sum(initial_state)}")
    total_mean = results['X1']['mean'] + results['X2']['mean'] + results['X3']['mean']
    print(f"  After 7 steps (mean total): {total_mean:.4f}")
    print(f"  Note: R1 adds 1 molecule each time it fires; R2 & R3 preserve total")
