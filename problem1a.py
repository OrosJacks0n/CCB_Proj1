"""
Problem 1(a): Analyzing Chemical Reaction Networks - Outcome Probabilities
===========================================================================
Given reactions:
  R1: 2X1 + X2 -> 4X3,  k1 = 1
  R2: X1 + 2X3 -> 3X2,  k2 = 2
  R3: X2 + X3  -> 2X1,  k3 = 3

Starting state: S = [110, 26, 55]
Compute Pr(C1), Pr(C2), Pr(C3) where:
  C1: x1 >= 150
  C2: x2 < 10
  C3: x3 > 100

Method: Gillespie's Stochastic Simulation Algorithm (SSA)
We run many independent simulations and check which outcome is reached first.
"""

import random
import math

def gillespie_step(state):
    """
    Perform one step of Gillespie's SSA for the given reaction network.
    
    Propensities (from the problem statement):
      a1 = k1 * C(x1,2) * x2 = 1 * x1*(x1-1)/2 * x2
      a2 = k2 * x1 * C(x3,2) = 2 * x1 * x3*(x3-1)/2 = x1 * x3*(x3-1)
      a3 = k3 * x2 * x3       = 3 * x2 * x3
    
    State changes:
      R1 fires: [-2, -1, +4]
      R2 fires: [-1, +3, -2]
      R3 fires: [+2, -1, -1]
    
    Returns: (new_state, dt) or (None, 0) if no reaction can fire.
    """
    x1, x2, x3 = state
    
    # Compute propensities
    a1 = (x1 * (x1 - 1) / 2) * x2       # R1: 2X1 + X2 -> 4X3
    a2 = x1 * x3 * (x3 - 1)              # R2: X1 + 2X3 -> 3X2
    a3 = 3 * x2 * x3                      # R3: X2 + X3  -> 2X1
    
    a_total = a1 + a2 + a3
    
    if a_total <= 0:
        return None, 0  # No reactions can fire
    
    # Time until next reaction (exponentially distributed)
    dt = -math.log(random.random()) / a_total
    
    # Choose which reaction fires
    r = random.random() * a_total
    
    if r < a1:
        # R1 fires: 2X1 + X2 -> 4X3
        new_state = [x1 - 2, x2 - 1, x3 + 4]
    elif r < a1 + a2:
        # R2 fires: X1 + 2X3 -> 3X2
        new_state = [x1 - 1, x2 + 3, x3 - 2]
    else:
        # R3 fires: X2 + X3 -> 2X1
        new_state = [x1 + 2, x2 - 1, x3 - 1]
    
    return new_state, dt


def check_outcomes(state):
    """Check if any outcome condition is met."""
    x1, x2, x3 = state
    outcomes = []
    if x1 >= 150:
        outcomes.append('C1')
    if x2 < 10:
        outcomes.append('C2')
    if x3 > 100:
        outcomes.append('C3')
    return outcomes


def simulate_once(initial_state, max_steps=10000):
    """
    Run one Gillespie simulation starting from initial_state.
    Returns the set of outcomes reached (first time any outcome is hit).
    We track which outcomes are reached at any point during the simulation.
    """
    state = list(initial_state)
    outcomes_reached = set()
    
    for step in range(max_steps):
        # Check outcomes at current state
        outcomes = check_outcomes(state)
        for o in outcomes:
            if o not in outcomes_reached:
                outcomes_reached.add(o)
        
        # If all outcomes have been checked or determined, we could stop early
        # But let's continue to see the full picture
        
        result = gillespie_step(state)
        if result[0] is None:
            break  # No reactions can fire
        state = result[0]
        
        # Safety check: no negative values
        if any(s < 0 for s in state):
            break
    
    # Check final state too
    outcomes = check_outcomes(state)
    for o in outcomes:
        outcomes_reached.add(o)
    
    return outcomes_reached


def estimate_probabilities(initial_state, num_simulations=50000, max_steps=5000):
    """
    Estimate Pr(C1), Pr(C2), Pr(C3) by running many simulations.
    For each simulation, we check whether each outcome is ever reached.
    """
    counts = {'C1': 0, 'C2': 0, 'C3': 0}
    
    for i in range(num_simulations):
        outcomes = simulate_once(initial_state, max_steps)
        for o in outcomes:
            counts[o] += 1
        
        if (i + 1) % 5000 == 0:
            print(f"  Completed {i+1}/{num_simulations} simulations...")
    
    probs = {k: v / num_simulations for k, v in counts.items()}
    return probs


if __name__ == "__main__":
    random.seed(42)
    
    print("=" * 60)
    print("Problem 1(a): Estimating Outcome Probabilities")
    print("=" * 60)
    print()
    print("Reactions:")
    print("  R1: 2X1 + X2 -> 4X3,  k1 = 1")
    print("  R2: X1 + 2X3 -> 3X2,  k2 = 2")
    print("  R3: X2 + X3  -> 2X1,  k3 = 3")
    print()
    print("Initial state: S = [110, 26, 55]")
    print("Outcomes:")
    print("  C1: x1 >= 150")
    print("  C2: x2 < 10")
    print("  C3: x3 > 100")
    print()
    
    initial_state = [110, 26, 55]
    num_sims = 10000
    max_steps = 10000
    
    print(f"Running {num_sims} Gillespie simulations (max {max_steps} steps each)...")
    print()
    
    probs = estimate_probabilities(initial_state, num_sims, max_steps)
    
    print()
    print("Results:")
    print(f"  Pr(C1) = Pr(x1 >= 150) ~= {probs['C1']:.4f}")
    print(f"  Pr(C2) = Pr(x2 < 10)   ~= {probs['C2']:.4f}")
    print(f"  Pr(C3) = Pr(x3 > 100)  ~= {probs['C3']:.4f}")
    print()
    
    # Verification: Also check with initial state [3,3,3] to verify algorithm
    print("-" * 60)
    print("Verification: Starting from S = [3, 3, 3], one step:")
    state = [3, 3, 3]
    counts_verify = {0: 0, 1: 0, 2: 0}
    n_verify = 100000
    for _ in range(n_verify):
        x1, x2, x3 = state
        a1 = (x1 * (x1 - 1) / 2) * x2
        a2 = x1 * x3 * (x3 - 1)
        a3 = 3 * x2 * x3
        a_total = a1 + a2 + a3
        r = random.random() * a_total
        if r < a1:
            counts_verify[0] += 1
        elif r < a1 + a2:
            counts_verify[1] += 1
        else:
            counts_verify[2] += 1
    
    print(f"  p1 (expected 1/6 ≈ 0.1667): {counts_verify[0]/n_verify:.4f}")
    print(f"  p2 (expected 1/3 ≈ 0.3333): {counts_verify[1]/n_verify:.4f}")
    print(f"  p3 (expected 1/2 ≈ 0.5000): {counts_verify[2]/n_verify:.4f}")
