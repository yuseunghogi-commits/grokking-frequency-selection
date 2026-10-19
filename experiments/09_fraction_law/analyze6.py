"""Pre-registered analysis for PREREG6 (written before the runs)."""
import os, sys, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ["GROK_OUT"] = os.path.join(HERE, "results"); sys.path.insert(0, os.path.join(ROOT, "experiments", "02_kaggle_v1"))
import gpu_grok as G
SPLITS = list(range(7600, 7608))
def ratio(Z):                                    # Z [S, I, K] z-scored log spectra -> sqrt(data var / init var), noise-corrected
    V = Z.mean(1); V = V - V.mean(0, keepdims=True)
    In = Z.mean(0); In = In - In.mean(0, keepdims=True)
    R = Z - Z.mean(1, keepdims=True) - Z.mean(0, keepdims=True) + Z.mean((0, 1), keepdims=True)
    s2 = V.var() - R.var() / Z.shape[1]; i2 = In.var() - R.var() / Z.shape[0]
    return float(np.sqrt(max(s2, 0) / i2)), float(s2), float(i2)
Z = {f: np.stack([G.zscore_log(np.load(os.path.join(HERE, "results", "runs", f"f{f}_s{s}.npz"))["embed60"]) for s in SPLITS])
     for f in (0.3, 0.85)}
r3, r85 = ratio(Z[0.3]), ratio(Z[0.85]); R = r85[0] / r3[0]
rng = np.random.default_rng(0); boot = []
for _ in range(2000):
    si = rng.integers(0, 8, 8); ii = rng.integers(0, 12, 12)
    a = ratio(Z[0.3][si][:, ii])[0]; b = ratio(Z[0.85][si][:, ii])[0]
    if a > 0: boot.append(b / a)
pred = float(np.sqrt(0.15 / 0.7))
res = dict(prereg6_sha256=hashlib.sha256(open(os.path.join(ROOT, "prereg", "PREREG6.txt"), "rb").read()).hexdigest(),
           ratio_f03=r3, ratio_f085=r85, R=R, R_boot_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           prediction_sqrt_law=pred, prediction_linear_law=0.15 / 0.7, F1_PASS=bool(0.7 * pred <= R <= 1.3 * pred))
json.dump(res, open(os.path.join(HERE, "results", "V6_results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
