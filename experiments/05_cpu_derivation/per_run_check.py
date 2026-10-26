"""Exploratory derivation check (writes results/per_run_check.json): does each single run's early embedding update dE
follow its own split's D(k)? 4 splits x 4 inits (same inits on every split). The split-specific part of each run's
log-spectrum is obtained by subtracting the mean over splits for the same init (and D is centred over splits likewise).
Compared against the run's own split and against the other splits, for AdamW and for plain SGD (lr 0.1)."""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("GROK_OUT", os.path.join(_HERE, "results")); sys.path.insert(0, os.path.join(_ROOT, "experiments", "02_kaggle_v1")); sys.path.insert(0, _HERE)
import numpy as np, torch
from scipy.stats import spearmanr
from deltaE import D_shared, spec, run
torch.set_num_threads(2)
SPLITS, INITS = (7000, 7001, 7002, 7003), [7100, 7101, 7102, 7103]
D = np.stack([D_shared(s) for s in SPLITS]); Dc = D - D.mean(0, keepdims=True)
res = {}
for opt in ("adam", "sgd"):
    for t in (10, 30):
        S = np.zeros((4, 4, 56))
        for si, s in enumerate(SPLITS):
            out = run(INITS, s, steps=(t,), sgd=(opt == "sgd"))
            for ii, dE in enumerate(out[t]): S[si, ii] = spec(dE)
        Sc = S - S.mean(0, keepdims=True)
        own = [spearmanr(Sc[si, ii], Dc[si]).correlation for si in range(4) for ii in range(4)]
        other = [spearmanr(Sc[si, ii], Dc[sj]).correlation for si in range(4) for ii in range(4) for sj in range(4) if sj != si]
        res[f"{opt}_step{t}"] = dict(own_mean=float(np.mean(own)), own_min=float(np.min(own)), own_max=float(np.max(own)),
                                     other_mean=float(np.mean(other)), n_runs=16)
        print(opt, t, res[f"{opt}_step{t}"], flush=True)
json.dump(res, open(os.path.join(_HERE, "results", "per_run_check.json"), "w"), indent=1)
