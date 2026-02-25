"""
Problem 2: Lambda Bacteriophage - Stealth vs. Hijack Mode (Optimized)
======================================================================
Simulate the lambda bacteriophage decision circuit for MOI = 1..10.
- Stealth mode: cI2 > 145
- Hijack mode: Cro2 > 55

Optimized version: Pre-processes reactions, uses numpy for speed.
"""

import random
import math
from collections import defaultdict


# ===================================================================
# Reaction model (lambda.r) and initial values (lambda.in)
# embedded as strings
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


def parse_side(tokens):
    """Parse one side of a reaction."""
    result = {}
    i = 0
    while i < len(tokens):
        species = tokens[i]
        count = int(tokens[i + 1])
        result[species] = result.get(species, 0) + count
        i += 2
    return result


def parse_reactions(raw_text):
    reactions = []
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
        # Compute net change
        all_species = set(reactants.keys()) | set(products.keys())
        net = {}
        for s in all_species:
            change = products.get(s, 0) - reactants.get(s, 0)
            if change != 0:
                net[s] = change
        reactions.append({
            'reactants': reactants,
            'products': products,
            'rate': rate,
            'net': net,
            'reactant_list': list(reactants.items()),
        })
    return reactions


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


def compute_propensity_fast(reactant_list, rate, state):
    """Compute propensity faster using pre-extracted reactant list."""
    prop = rate
    for species, count in reactant_list:
        x = state.get(species, 0)
        if x < count:
            return 0.0
        if count == 1:
            prop *= x
        elif count == 2:
            prop *= x * (x - 1) * 0.5
        elif count == 3:
            prop *= x * (x - 1) * (x - 2) / 6.0
        else:
            c = 1.0
            for j in range(count):
                c *= (x - j) / (j + 1)
            prop *= c
    return prop


def simulate_lambda(reactions, initial_state, max_time=500.0, max_steps=500000):
    """Run one Gillespie simulation. Returns 'stealth', 'hijack', or 'undecided'."""
    state = dict(initial_state)
    t = 0.0
    
    for step in range(max_steps):
        # Check decision
        ci2 = state.get('cI2', 0)
        cro2 = state.get('Cro2', 0)
        if ci2 > 145:
            return 'stealth'
        if cro2 > 55:
            return 'hijack'
        
        # Compute propensities
        props = []
        a_total = 0.0
        for rxn in reactions:
            p = compute_propensity_fast(rxn['reactant_list'], rxn['rate'], state)
            props.append(p)
            a_total += p
        
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
        chosen = len(reactions) - 1
        for i in range(len(reactions)):
            cumsum += props[i]
            if r < cumsum:
                chosen = i
                break
        
        # Apply reaction
        rxn = reactions[chosen]
        for species, change in rxn['net'].items():
            state[species] = state.get(species, 0) + change
    
    # Final check
    if state.get('cI2', 0) > 145:
        return 'stealth'
    if state.get('Cro2', 0) > 55:
        return 'hijack'
    return 'undecided'


if __name__ == "__main__":
    random.seed(42)
    
    print("=" * 60)
    print("Problem 2: Lambda Bacteriophage Decision")
    print("=" * 60)
    print()
    print("Stealth mode: cI2 > 145")
    print("Hijack mode:  Cro2 > 55")
    print()
    
    reactions = parse_reactions(LAMBDA_REACTIONS_RAW)
    print(f"Parsed {len(reactions)} reactions")
    print()
    
    num_sims = 100
    max_time = 500.0
    
    print(f"Running {num_sims} simulations per MOI value (max_time={max_time})...")
    print()
    
    all_results = {}
    for moi in range(1, 11):
        initial = dict(LAMBDA_INITIAL)
        initial['MOI'] = moi
        
        counts = {'stealth': 0, 'hijack': 0, 'undecided': 0}
        for sim in range(num_sims):
            outcome = simulate_lambda(reactions, initial, max_time)
            counts[outcome] += 1
            if (sim + 1) % 25 == 0:
                pass  # progress tracking
        
        p_stealth = counts['stealth'] / num_sims
        p_hijack = counts['hijack'] / num_sims
        p_undecided = counts['undecided'] / num_sims
        
        all_results[moi] = (p_stealth, p_hijack, p_undecided)
        print(f"  MOI={moi:2d}: P(Stealth)={p_stealth:.3f}, P(Hijack)={p_hijack:.3f}, P(Undecided)={p_undecided:.3f}")
    
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
    print("Note: As MOI increases, the probability of 'stealth' (lysogeny)")
    print("mode should increase, reflecting the biological reality that")
    print("higher multiplicity of infection favors integration.")
