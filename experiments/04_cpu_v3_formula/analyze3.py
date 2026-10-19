"""Pre-registered v3 analysis (written before the runs finished)."""
import os, sys, json, glob, hashlib
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("GROK_OUT", os.path.join(_HERE, "results"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "02_kaggle_v1"))
import numpy as np
import gpu_grok as G
P = 113; w = 2 * np.pi * np.arange(1, 57) / P
SPLITS = list(range(5000, 5016)); INITS = list(range(6000, 6016))

def D_shared(split):
    tr, _ = G.split_indices(split); a = (tr // P).numpy(); b = (tr % P).numpy(); c = (a + b) % P
    out = np.zeros(56)
    for side in (a, b):
        S = np.zeros((56, P), complex); np.add.at(S.T, c, np.exp(1j * np.outer(w, side)).T)
        out += (np.abs(S) ** 2).sum(1)
    return np.log(out)

E = np.full((16, 16, 56), np.nan); A0 = np.full((16, 56), np.nan)
for f in glob.glob(str(G.OUT / "runs" / "*.npz")):
    d = np.load(f); s = SPLITS.index(int(d["split"]))
    for j, i in enumerate(d["inits"]):
        E[INITS.index(int(i)), s] = d["embed200"][j]; A0[INITS.index(int(i))] = np.log(d["path0"][j])
assert not np.isnan(E).any(), "incomplete runs"
Z = G.zscore_log(E)                                   # [I,S,56]
V = Z.mean(0); V = V - V.mean(0, keepdims=True)       # [S,56]
D = np.stack([D_shared(s) for s in SPLITS]); Dc = D - D.mean(0, keepdims=True)
rho, p = G.perm_p_within(Dc, V, reps=10000)
res = dict(prereg3_sha256=hashlib.sha256(open(os.path.join(_ROOT, "prereg", "PREREG3.txt"), "rb").read()).hexdigest())
# reliability of the vote (split-half of inits), descriptive
h1 = Z[0::2].mean(0); h2 = Z[1::2].mean(0); h1 -= h1.mean(0); h2 -= h2.mean(0)
r = G.within_split_spearman(h1, h2); res["vote_reliability"] = 2 * r / (1 + r)
res["P1"] = dict(rho=rho, p=p, PASS=bool(rho > 0.5 and p < 1e-3))
W3 = np.argsort(-E, -1)[..., :3]
score = G.zz(A0)[:, None, :] + 1.0 * G.zz(D)[None]
Hc = G.topk_hits_cells(score, W3); Ha = G.topk_hits_cells(np.broadcast_to(G.zz(A0)[:, None, :], E.shape), W3)
Hd = G.topk_hits_cells(np.broadcast_to(G.zz(D)[None], E.shape), W3)
ci = G.paired_boot_IS(Hc, Ha)
_, pc = G.perm_p_hits(A0, G.zz(D), 1.0, W3, reps=2000)
n = 16 * 16
res["P2"] = dict(hits_combined=int(Hc.sum()), hits_attention_only=int(Ha.sum()), hits_D_only=int(Hd.sum()),
                 of=3 * n, chance=n * 9 / 56, ci_diff=ci, p_vs_chance=pc, PASS=bool(ci[0] > 0 and pc < 1e-3))
json.dump(res, open(G.OUT / "V3_results.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
