"""Pre-registered analysis for PREREG4 (G1, G2). Written before the runs finished."""
import sys, os, json, glob, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from freqsel.tasks import Task
from freqsel.predictors import label_sharing_D
from freqsel.stats import zscore_log, perm_p_within, split_half_reliability
RES = os.path.join(ROOT, "results", "v4")
SPLITS = list(range(8000, 8008))
out = {}
for name, p in [("add", 59), ("add", 97), ("mul", 59), ("mul", 97)]:
    T = Task(name, p); tag = f"{name}{p}"
    fs = [os.path.join(RES, f"{tag}_s{s}.npz") for s in SPLITS]
    if not all(os.path.exists(f) for f in fs): out[tag] = "incomplete"; continue
    Z = np.stack([zscore_log(np.load(f)["embed100"]) for f in fs])          # [S, inits, nf]
    V = Z.mean(1); V -= V.mean(0, keepdims=True)
    Draw = np.stack([label_sharing_D(T, s) for s in SPLITS]); D = Draw - Draw.mean(0, keepdims=True)
    rho, pval = perm_p_within(D, V, reps=10000)
    spread = float(np.mean([np.exp(d).std() / np.exp(d).mean() for d in Draw]))
    theory = 1 / np.sqrt(T.n)
    out[tag] = dict(spearman=rho, p=pval, G1_PASS=bool(rho > 0.3 and pval < 0.01),
                    D_relative_spread=spread, theory=theory, ratio=spread / theory,
                    G2_PASS=bool(abs(spread / theory - 1) <= 0.25),
                    vote_reliability=split_half_reliability(Z), n_freqs=T.nf)
json.dump(out, open(os.path.join(RES, "V4_results.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
