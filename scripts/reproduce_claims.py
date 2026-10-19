"""Print every headline number in the README from the stored result files (no training).

    python scripts/reproduce_claims.py          # read stored JSON
    python scripts/reproduce_claims.py --recompute-v4   # re-run the PREREG4 analysis from the stored .npz first
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: json.load(open(os.path.join(ROOT, *p)))

if "--recompute-v4" in sys.argv:
    subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "06_generalisation", "analyze_v4.py")],
                   check=True, stdout=subprocess.DEVNULL)

ok = lambda b: "PASS" if b else "FAIL"
print("== v1 (Kaggle T4, PREREG1) ==")
S = J("experiments", "02_kaggle_v1", "results", "SUMMARY.json")
C1 = S["F"]["C1"]
print(f"C1 top-3 hits on held-out test cells (of {C1['of']}, chance {C1['chance']:.0f}): "
      f"init-only {C1['hits']['init']}, data-only {C1['hits']['data']}, both {C1['hits']['both']}  -> {ok(C1['PASS'])}")
print(f"   bootstrap CI both-init {C1['ci_both_minus_init']}, both-data {C1['ci_both_minus_data']}")
C = S["C"]
print(f"step-500 proxy recall of final keys: {C['containment_step500']:.3f}  (proxy_valid={C['proxy_valid']})")
G = S["G"]
print(f"C6 internal vs behavioural grok-time prediction: C={G['internal_test_C']:.3f} vs {G['behavioural_test_C']:.3f} -> {ok(G['PASS'])}")

print("\n== v2 (Kaggle T4, PREREG2): what casts the data vote ==")
V = J("experiments", "03_kaggle_v2", "results", "V2_results.json")
for h, name in [("H2", "relabel a->a+t (shift)"), ("H1", "shuffled labels"), ("H3", "subtraction task"), ("H4", "MLP-only arch")]:
    x = V[h]
    print(f"{h} {name:24s} agreement with true vote {x['agreement']:.3f} (ceiling {x['ceiling']:.3f}, p {x['p']:.4f})")
print(f"H5 individual-pair datamodel: {ok(V['H5']['PASS'])} (held-out R^2 ~0.03; vote is collective)")

print("\n== v3 (CPU, PREREG3): closed-form D(k) ==")
V3 = J("experiments", "04_cpu_v3_formula", "results", "V3_results.json")
print(f"P1 within-split Spearman(D, data vote) = {V3['P1']['rho']:.3f}, p = {V3['P1']['p']:.1e} -> {ok(V3['P1']['PASS'])}")
P2 = V3["P2"]
print(f"P2 top-3 hits (of {P2['of']}, chance {P2['chance']:.0f}): OV-path init {P2['hits_attention_only']}, "
      f"D only {P2['hits_D_only']}, combined {P2['hits_combined']}; CI diff {P2['ci_diff']} -> {ok(P2['PASS'])}")

TF = J("experiments", "04_cpu_v3_formula", "results", "training_free_v1.json")
t, f = TF["test_factorial"], TF["final_circuit"]
print(f"[exploratory] training-free score on v1 test: {t['hits_training_free']} hits vs OV-path only {t['hits_attention_only']} "
      f"(chance {t['chance']:.0f}; C4 with trained votes {t['C4_hits_needing_trained_votes']})")
print(f"[exploratory] top-3 precision against the final circuit of {f['runs']} fully trained runs: "
      f"{f['top3_precision_training_free']:.2f} (OV-path only {f['top3_precision_attention_only']:.2f}, chance {f['chance']:.2f})")

print("\n== v4 (CPU, PREREG4): other primes and multiplication ==")
V4 = J("results", "v4", "V4_results.json")
for tag, r in V4.items():
    if not isinstance(r, dict):
        print(f"{tag}: {r}"); continue
    print(f"{tag:6s} G1 rho={r['spearman']:.3f} p={r['p']:.1e} {ok(r['G1_PASS'])} | "
          f"G2 spread {r['D_relative_spread']:.3f} vs 1/sqrt(n) {r['theory']:.3f} (ratio {r['ratio']:.2f}) {ok(r['G2_PASS'])} | "
          f"vote reliability {r['vote_reliability']:.2f}")

print("\n== v5 (PREREG5): steering by editing the split ==")
V5 = J("experiments", "07_steering", "results", "V5_results.json")
print(f"S1 target in step-200 top-3 by strength: {[round(x, 3) for x in V5['S1']['rate_by_level']]}  p={V5['S1']['p_signflip']:.4f} -> {ok(V5['S1']['PASS'])}")
print(f"S2 within-target Spearman(strength, rate) {V5['S2']['mean_within_target_spearman']:.3f} p={V5['S2']['p']:.1e} -> {ok(V5['S2']['PASS'])}")
s3 = V5["S3"]; print(f"S3 target in final circuit: steered {s3['steered_key']}/{s3['n_each']} vs original {s3['base_key']}/{s3['n_each']}, Fisher p={s3['p_fisher']:.1e} -> {ok(s3['PASS'])}")
print("S4", V5["S4"])
print("\n== v6 (PREREG6): sqrt(1-f) law ==")
V6 = J("experiments", "09_fraction_law", "results", "V6_results.json")
print(f"R = {V6['R']:.3f} (prediction {V6['prediction_sqrt_law']:.3f}, window +-30%) -> {ok(V6['F1_PASS'])}")
