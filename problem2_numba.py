"""
Problem 2: Lambda Bacteriophage - Stealth vs. Hijack Mode (Numba JIT)
======================================================================
Simulate the lambda bacteriophage decision circuit for MOI = 1..10.
- Stealth mode: cI2 > 145
- Hijack mode: Cro2 > 55

Uses numba JIT for ~100x speedup over pure Python.
"""

import numpy as np
import numba as nb
import time


# ===================================================================
# Reaction model (lambda.r) and initial values (lambda.in)
# ===================================================================

LAMBDA_REACTIONS_RAW = """
OLRNAP 1 MOI 1 : OLRNAP 1 MOI 1 N 10 : 0.011
RNAP 1 PRE 1 : PRERNAP 1 : 0.01
PRERNAP 1 : RNAP 1 PRE 1 : 1.0
RNAP 1 cI2 1 OR 1 : OR13RNAPcI 1 : 0.02569
OR13RNAPcI 1 : RNAP 1 cI2 1 OR 1 : 1.0
RNAP 1 cI2 1 OR 1 : ORcIRNAP 1 : 0.00967
ORcIRNAP 1 : RNAP 1 cI2 1 OR 1 : 1.0
NUTR4 1 N 1 : NUTRN4 1 : 0.2
NUTRN4 1 : NUTR4 1 N 1 : 1.0
P1 1 cIII 1 : P1cIII 1 : 0.01
P1cIII 1 : P1 1 cIII 1 : 0.01
cI 1 : : 7.0E-4
cI2 1 OR 1 : ORcI 1 : 0.2165
ORcI 1 : cI2 1 OR 1 : 1.0
MOI 1 OR12RNAP 1 NUTRN 1 : MOI 1 OR12RNAP 1 NUTRN 1 cII 10 : 0.014
Basal_error 1 MOI 1 OR3RNAP 1 : Basal_error 1 MOI 1 OR3RNAP 1 cI 10 : 0.0010
ORcIRNAP 1 MOI 1 NUTRN3 1 : ORcIRNAP 1 MOI 1 NUTRN3 1 cII 10 : 0.014
ORCroRNAP 1 NUTRN4 1 MOI 1 : ORCroRNAP 1 NUTRN4 1 MOI 1 cII 10 : 0.014
NUTR 1 N 1 : NUTRN 1 : 0.2
NUTRN 1 : NUTR 1 N 1 : 1.0
Basal_error 1 MOI 1 OR13RNAPcI 1 : Basal_error 1 MOI 1 OR13RNAPcI 1 cI 10 : 0.0010
Cro 2 : Cro2 1 : 0.05
Cro2 1 : Cro 2 : 0.5
Cro2 1 RNAP 1 cI2 1 OR 1 : ORRNAPcICro 1 : 8.0E-5
ORRNAPcICro 1 : Cro2 1 RNAP 1 cI2 1 OR 1 : 1.0
N 1 NUTR3 1 : NUTRN3 1 : 0.2
NUTRN3 1 : N 1 NUTR3 1 : 1.0
OLRNAP 1 MOI 1 NUTL 1 : cIII 10 OLRNAP 1 MOI 1 NUTL 1 : 0.0022
Cro2 2 OL 1 : OL2Cro 1 : 0.0158
OL2Cro 1 : Cro2 2 OL 1 : 1.0
P2cII 1 : P2 1 : 0.6
NUTR 1 MOI 1 OR12RNAP 1 : NUTR 1 MOI 1 OR12RNAP 1 cII 10 : 0.0070
Cro2 2 RNAP 1 OR 1 : ORRNAP2Cro 1 : 2.6E-4
ORRNAP2Cro 1 : Cro2 2 RNAP 1 OR 1 : 1.0
RNAP 1 OR 1 : OR12RNAP 1 : 0.69422
OR12RNAP 1 : RNAP 1 OR 1 : 1.0
P2cIII 1 : P2 1 : 0.0010
cI2 1 OL 1 : OLcI 1 : 0.2025
OLcI 1 : cI2 1 OL 1 : 1.0
P1 1 cII 1 : P1cII 1 : 0.0002
P1cII 1 : P1 1 cII 1 : 0.05
N 1 : : 0.00231
PRE 1 cII 1 : PREcII 1 : 0.00726
PREcII 1 : PRE 1 cII 1 : 1.0
Cro2 1 cI2 1 OR 1 : ORCrocI 1 : 0.1779
ORCrocI 1 : Cro2 1 cI2 1 OR 1 : 1.0
P1cII 1 : P1 1 : 0.6
OR2RNAP 1 MOI 1 NUTR2 1 : OR2RNAP 1 MOI 1 NUTR2 1 cII 10 : 0.0070
MOI 1 PREcIIRNAP 1 : MOI 1 cI 10 PREcIIRNAP 1 : 0.015
ORCroRNAP 1 MOI 1 : Cro 10 ORCroRNAP 1 MOI 1 : 0.014
RNAP 1 cI2 1 OR 1 : ORRNAPcI 1 : 0.0019
ORRNAPcI 1 : RNAP 1 cI2 1 OR 1 : 1.0
cI 2 : cI2 1 : 0.05
cI2 1 : cI 2 : 0.5
ORcIRNAP 1 MOI 1 NUTR3 1 : ORcIRNAP 1 MOI 1 NUTR3 1 cII 10 : 0.0070
ORRNAPcICro 1 MOI 1 Kd 1 : ORRNAPcICro 1 MOI 1 cI 10 Kd 1 : 0.011
cI2 3 OR 1 : OR3cI 1 : 8.1E-4
OR3cI 1 : cI2 3 OR 1 : 1.0
Cro2 2 OR 1 : OR2Cro 1 : 0.03342
OR2Cro 1 : Cro2 2 OR 1 : 1.0
cI2 2 OL 1 : OL2cI 1 : 0.058
OL2cI 1 : cI2 2 OL 1 : 1.0
OLRNAP 1 NUTLN 1 MOI 1 : cIII 10 OLRNAP 1 NUTLN 1 MOI 1 : 0.011
MOI 1 ORRNAP2cI 1 Kd 1 : MOI 1 ORRNAP2cI 1 cI 10 Kd 1 : 0.011
Cro2 1 RNAP 1 OR 1 : ORRNAPCro 1 : 0.01186
ORRNAPCro 1 : Cro2 1 RNAP 1 OR 1 : 1.0
ORcIRNAP 1 MOI 1 : ORcIRNAP 1 Cro 10 MOI 1 : 0.014
Cro2 1 cI2 1 OL 1 : OLcICro 1 : 0.014
OLcICro 1 : Cro2 1 cI2 1 OL 1 : 1.0
MOI 1 PRERNAP 1 : MOI 1 cI 10 PRERNAP 1 : 4.0E-5
N 1 NUTL 1 : NUTLN 1 : 0.2
NUTLN 1 : N 1 NUTL 1 : 1.0
NUTR4 1 ORCroRNAP 1 MOI 1 : NUTR4 1 ORCroRNAP 1 MOI 1 cII 10 : 0.0070
Basal_error 1 MOI 1 ORRNAPCro 1 : Basal_error 1 MOI 1 cI 10 ORRNAPCro 1 : 0.0010
RNAP 1 OR 1 : OR3RNAP 1 : 0.1362
OR3RNAP 1 : RNAP 1 OR 1 : 1.0
Cro2 1 cI2 2 OR 1 : ORCro2cI 1 : 0.02133
ORCro2cI 1 : Cro2 1 cI2 2 OR 1 : 1.0
Cro2 1 RNAP 1 OR 1 : ORCroRNAP 1 : 0.25123
ORCroRNAP 1 : Cro2 1 RNAP 1 OR 1 : 1.0
RNAP 1 PRE 1 cII 1 : PREcIIRNAP 1 : 0.00161
PREcIIRNAP 1 : RNAP 1 PRE 1 cII 1 : 1.0
Cro2 1 PRE 1 : PRECro 1 : 1.0E-5
PRECro 1 : Cro2 1 PRE 1 : 0.1
Basal_error 1 MOI 1 ORRNAP2CrocI 1 : Basal_error 1 MOI 1 ORRNAP2CrocI 1 cI 10 : 0.0010
Cro 1 : : 0.0025
P2 1 cII 1 : P2cII 1 : 2.5E-4
P2cII 1 : P2 1 cII 1 : 0.065
cI2 2 OR 1 : OR2cI 1 : 0.06568
OR2cI 1 : cI2 2 OR 1 : 1.0
P1cIII 1 : P1 1 : 0.001
RNAP 2 OR 1 : OR2RNAP 1 : 0.09455
OR2RNAP 1 : RNAP 2 OR 1 : 1.0
Cro2 2 cI2 1 OR 1 : OR2CrocI 1 : 0.00322
OR2CrocI 1 : Cro2 2 cI2 1 OR 1 : 1.0
OR2RNAP 1 Basal_error 1 MOI 1 : OR2RNAP 1 Basal_error 1 MOI 1 cI 10 : 0.0010
cIII 1 P2 1 : P2cIII 1 : 0.01
P2cIII 1 : cIII 1 P2 1 : 0.01
Cro2 1 OR 1 : ORCro 1 : 0.449
ORCro 1 : Cro2 1 OR 1 : 1.0
Cro2 3 OR 1 : OR3Cro 1 : 6.9E-4
OR3Cro 1 : Cro2 3 OR 1 : 1.0
Cro2 1 OL 1 : OLCro 1 : 0.4132
OLCro 1 : Cro2 1 OL 1 : 1.0
ORRNAP2Cro 1 Basal_error 1 MOI 1 : ORRNAP2Cro 1 Basal_error 1 MOI 1 cI 10 : 0.0010
Cro2 1 RNAP 1 cI2 1 OR 1 : ORRNAP2CrocI 1 : 0.00112
ORRNAP2CrocI 1 : Cro2 1 RNAP 1 cI2 1 OR 1 : 1.0
RNAP 1 cI2 2 OR 1 : ORRNAP2cI 1 : 0.0079
ORRNAP2cI 1 : RNAP 1 cI2 2 OR 1 : 1.0
N 1 NUTR2 1 : NUTRN2 1 : 0.2
NUTRN2 1 : N 1 NUTR2 1 : 1.0
MOI 1 OR12RNAP 1 : Cro 10 MOI 1 OR12RNAP 1 : 0.014
OR2RNAP 1 MOI 1 NUTRN2 1 : OR2RNAP 1 MOI 1 NUTRN2 1 cII 10 : 0.014
OR2RNAP 1 MOI 1 : Cro 10 OR2RNAP 1 MOI 1 : 0.014
MOI 1 ORRNAPcI 1 Kd 1 : ORRNAPcI 1 MOI 1 cI 10 Kd 1 : 0.011
RNAP 1 OL 1 : OLRNAP 1 : 0.6942
OLRNAP 1 : RNAP 1 OL 1 : 1.0
cII 1 : : 7.0E-4
OR3RNAP 1 MOI 1 : Cro 10 OR3RNAP 1 MOI 1 : 0.014
MOI 1 OR13RNAPcI 1 Kd 1 : cI 10 MOI 1 OR13RNAPcI 1 Kd 1 : 0.011
OR13RNAPcI 1 MOI 1 : OR13RNAPcI 1 Cro 10 MOI 1 : 0.014
""".strip()


LAMBDA_INITIAL = {
    'cI2': 0, 'Cro2': 0, 'MOI': 6, 'P1cIII': 0, 'ORCrocI': 0,
    'NUTRN4': 0, 'OR13RNAPcI': 0, 'Kd': 1, 'ORCro2cI': 0, 'N': 0,
    'ORcI': 0, 'PRECro': 0, 'OR': 1, 'ORCro': 0, 'PRE': 1,
    'ORRNAP2CrocI': 0, 'NUTRN2': 0, 'OLCro': 0, 'P2cIII': 0,
    'ORRNAPcICro': 0, 'OR2RNAP': 0, 'P2': 100, 'OR2cI': 0, 'OR3cI': 0,
    'PREcIIRNAP': 0, 'ORRNAPCro': 0, 'NUTR4': 1, 'NUTL': 1,
    'OR12RNAP': 0, 'OLcI': 0, 'ORCroRNAP': 0, 'Cro': 0,
    'ORRNAP2Cro': 0, 'NUTRN3': 0, 'Basal_error': 1, 'ORcIRNAP': 0,
    'cIII': 0, 'P1cII': 0, 'PREcII': 0, 'OLcICro': 0, 'OL': 1,
    'OR3RNAP': 0, 'OL2Cro': 0, 'cI': 0, 'OR2Cro': 0, 'ORRNAP2cI': 0,
    'OL2cI': 0, 'NUTLN': 0, 'OR2CrocI': 0, 'P1': 40, 'RNAP': 30,
    'NUTR': 1, 'NUTR3': 1, 'OLRNAP': 0, 'PRERNAP': 0, 'cII': 0,
    'P2cII': 0, 'ORRNAPcI': 0, 'OR3Cro': 0, 'NUTR2': 1, 'NUTRN': 0,
}


def parse_side(tokens):
    result = {}
    i = 0
    while i < len(tokens):
        species = tokens[i]
        count = int(tokens[i + 1])
        result[species] = result.get(species, 0) + count
        i += 2
    return result


def build_model(raw_text, initial_dict):
    """Parse reactions and build numba-friendly arrays."""
    all_species = set(initial_dict.keys())
    raw_reactions = []
    
    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        if len(parts) != 3:
            continue
        reactant_tokens = parts[0].strip().split() if parts[0].strip() else []
        product_tokens = parts[1].strip().split() if parts[1].strip() else []
        rate = float(parts[2].strip())
        reactants = parse_side(reactant_tokens) if reactant_tokens else {}
        products = parse_side(product_tokens) if product_tokens else {}
        all_species.update(reactants.keys())
        all_species.update(products.keys())
        raw_reactions.append((reactants, products, rate))
    
    species_list = sorted(all_species)
    sp_idx = {s: i for i, s in enumerate(species_list)}
    n_sp = len(species_list)
    n_rxn = len(raw_reactions)
    
    # Build stoichiometry matrix
    stoich = np.zeros((n_rxn, n_sp), dtype=np.int32)
    rates = np.zeros(n_rxn, dtype=np.float64)
    
    # For reactants: encode as a flat 2D array
    # reactant_matrix[r, s] = stoichiometric coefficient of species s in reaction r
    reactant_matrix = np.zeros((n_rxn, n_sp), dtype=np.int32)
    
    for r, (reactants, products, rate) in enumerate(raw_reactions):
        rates[r] = rate
        for s, c in reactants.items():
            idx = sp_idx[s]
            stoich[r, idx] -= c
            reactant_matrix[r, idx] = c
        for s, c in products.items():
            idx = sp_idx[s]
            stoich[r, idx] += c
    
    init_state = np.zeros(n_sp, dtype=np.int64)
    for s, v in initial_dict.items():
        init_state[sp_idx[s]] = v
    
    return species_list, sp_idx, stoich, rates, reactant_matrix, init_state


@nb.njit(cache=True)
def compute_propensities(state, rates, reactant_matrix, props):
    """Compute all propensities. Numba-compiled."""
    n_rxn = rates.shape[0]
    n_sp = state.shape[0]
    
    for r in range(n_rxn):
        p = rates[r]
        for s in range(n_sp):
            c = reactant_matrix[r, s]
            if c > 0:
                x = state[s]
                if x < c:
                    p = 0.0
                    break
                if c == 1:
                    p *= x
                elif c == 2:
                    p *= x * (x - 1) * 0.5
                elif c == 3:
                    p *= x * (x - 1) * (x - 2) / 6.0
                else:
                    for j in range(c):
                        p *= (x - j) / (j + 1)
        props[r] = p


@nb.njit(cache=True)
def run_one_sim(stoich, rates, reactant_matrix, init_state,
                ci2_idx, cro2_idx, max_time, max_steps):
    """Run one Gillespie simulation. Returns 0=stealth, 1=hijack, 2=undecided."""
    n_rxn = rates.shape[0]
    state = init_state.copy()
    props = np.zeros(n_rxn, dtype=np.float64)
    t = 0.0
    
    for step in range(max_steps):
        # Check decision
        if state[ci2_idx] > 145:
            return 0  # stealth
        if state[cro2_idx] > 55:
            return 1  # hijack
        
        # Compute propensities
        compute_propensities(state, rates, reactant_matrix, props)
        
        a_total = 0.0
        for i in range(n_rxn):
            a_total += props[i]
        
        if a_total <= 0:
            break
        
        # Time step
        u1 = np.random.random()
        dt = -np.log(u1) / a_total
        t += dt
        if t > max_time:
            break
        
        # Choose reaction
        u2 = np.random.random() * a_total
        cumsum = 0.0
        chosen = n_rxn - 1
        for i in range(n_rxn):
            cumsum += props[i]
            if u2 < cumsum:
                chosen = i
                break
        
        # Apply reaction
        for s in range(state.shape[0]):
            state[s] += stoich[chosen, s]
    
    # Final check
    if state[ci2_idx] > 145:
        return 0
    if state[cro2_idx] > 55:
        return 1
    return 2  # undecided


@nb.njit(cache=True)
def run_many_sims(stoich, rates, reactant_matrix, init_state,
                  ci2_idx, cro2_idx, max_time, max_steps, num_sims):
    """Run many simulations and return counts [stealth, hijack, undecided]."""
    counts = np.zeros(3, dtype=np.int64)
    for sim in range(num_sims):
        result = run_one_sim(stoich, rates, reactant_matrix, init_state,
                             ci2_idx, cro2_idx, max_time, max_steps)
        counts[result] += 1
    return counts


if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 60)
    print("Problem 2: Lambda Bacteriophage Decision")
    print("=" * 60)
    print()
    print("Stealth mode: cI2 > 145")
    print("Hijack mode:  Cro2 > 55")
    print()
    
    species_list, sp_idx, stoich, rates, reactant_matrix, init_state = \
        build_model(LAMBDA_REACTIONS_RAW, LAMBDA_INITIAL)
    
    ci2_idx = sp_idx['cI2']
    cro2_idx = sp_idx['Cro2']
    moi_idx = sp_idx['MOI']
    
    print(f"Species: {len(species_list)}, Reactions: {len(rates)}")
    print()
    
    num_sims = 200
    max_time = 1000.0
    max_steps = 1000000
    
    # Warm up JIT
    print("Compiling JIT... ", end="", flush=True)
    t0 = time.time()
    warm = init_state.copy()
    warm[moi_idx] = 1
    _ = run_many_sims(stoich, rates, reactant_matrix, warm,
                      ci2_idx, cro2_idx, 10.0, 1000, 2)
    print(f"done ({time.time()-t0:.1f}s)")
    print()
    
    print(f"Running {num_sims} sims per MOI, max_time={max_time}")
    print()
    
    all_results = {}
    for moi in range(1, 11):
        start = time.time()
        init = init_state.copy()
        init[moi_idx] = moi
        
        counts = run_many_sims(stoich, rates, reactant_matrix, init,
                               ci2_idx, cro2_idx, max_time, max_steps, num_sims)
        
        elapsed = time.time() - start
        ps = counts[0] / num_sims
        ph = counts[1] / num_sims
        pu = counts[2] / num_sims
        all_results[moi] = (ps, ph, pu)
        print(f"  MOI={moi:2d}: P(Stealth)={ps:.3f}, P(Hijack)={ph:.3f}, "
              f"P(Undecided)={pu:.3f}  [{elapsed:.1f}s]")
    
    print()
    print("=" * 60)
    print("Summary Table:")
    print("=" * 60)
    print(f"{'MOI':>4s}  {'P(Stealth)':>12s}  {'P(Hijack)':>12s}  {'P(Undecided)':>14s}")
    print("-" * 50)
    for moi in range(1, 11):
        ps, ph, pu = all_results[moi]
        print(f"{moi:4d}  {ps:12.3f}  {ph:12.3f}  {pu:14.3f}")
    
    print()
    print("Biological interpretation:")
    print("- Low MOI (1-2): Predominantly hijack (lytic) mode - fewer phage")
    print("  genomes produce less cII/cIII, favoring Cro-driven lytic pathway")
    print("- Transition (3-5): Mixed outcomes as competition between cI and Cro")
    print("  becomes balanced")
    print("- High MOI (6-10): More phage genomes produce more cII, which activates")
    print("  PRE promoter for cI transcription, favoring lysogenic (stealth) mode")
