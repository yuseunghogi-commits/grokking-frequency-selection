"""Pre-registered analysis for PREREG7 (written before the runs). Usage: python analyze7.py <results_dir>"""
import os, sys, json, glob, hashlib, itertools, numpy as np
from scipy.stats import spearmanr, fisher_exact
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
sys.path.insert(0, os.path.join(R, "experiments", "07_steering")); from steer import D_of, split_perm
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "results")
I = json.load(open(os.path.join(R, "experiments", "07_steering", "splits_info.json"))); T, BASE = I["targets"], I["base"]
def zlog(x): l = np.log(x); return (l - l.mean(-1, keepdims=True)) / l.std(-1, keepdims=True)
res = dict(prereg7_sha256=hashlib.sha256(open(os.path.join(R, "prereg", "PREREG7.txt"), "rb").read()).hexdigest())
# ---- stage A
emb = {}; ro = {}
for k in T:
    for g in (0.0, 1.0):
        fs = sorted(glob.glob(os.path.join(RES, "A", f"k{k}_g{g}_c*.npz")))
        d = [np.load(f) for f in fs]
        emb[(k, g)] = np.concatenate([x["embed200"] for x in d]); inits = np.concatenate([x["inits"] for x in d])
        if g == 0.0: ro[k] = {n: np.concatenate([x[f"ro_{n}"] for x in d]) for n in ("OV1", "OV2", "ALL", "E", "U")}
hit = np.array([[(np.argsort(-emb[(k, g)], -1)[:, :3] == k - 1).any(1).mean() for g in (0.0, 1.0)] for k in T])
diff = hit[:, 1] - hit[:, 0]; flips = np.array(list(itertools.product([1, -1], repeat=len(T))))
p1 = float(((flips * diff).mean(1) >= diff.mean() - 1e-12).mean())
res["T1"] = dict(rate_original=float(hit[:, 0].mean()), rate_steered=float(hit[:, 1].mean()), by_target=hit.tolist(),
                 p_signflip=p1, PASS=bool(hit[:, 1].mean() >= 0.5 and p1 < 0.01))
# T2: data vote on the 8 original splits vs D (within-split Spearman, centred over splits; permutation within split)
Z = np.stack([zlog(emb[(k, 0.0)]) for k in T])                     # [S=8, I=12, 56]
V = Z.mean(1); V = V - V.mean(0, keepdims=True)
D = np.stack([np.log(D_of(split_perm(BASE[str(k)])[0])) for k in T]); D = D - D.mean(0, keepdims=True)
def wss(X, Y): return float(np.mean([spearmanr(x, y).correlation for x, y in zip(X, Y)]))
obs = wss(D, V); rng = np.random.default_rng(0)
null = np.array([wss(np.array([rng.permutation(x) for x in D]), V) for _ in range(10000)])
p2 = float(((null >= obs - 1e-12).sum() + 1) / (len(null) + 1))
res["T2"] = dict(spearman=obs, p=p2, PASS=bool(obs > 0.3 and p2 < 0.01))
# T4: init vote (mean over the 8 original splits, centred over inits) vs init readouts (same inits on every split)
IV = Z.mean(0); IV = IV - IV.mean(0, keepdims=True)                # [12, 56]
R0 = {n: np.log(ro[T[0]][n]) for n in ("OV1", "OV2", "ALL", "E", "U")}
rho = {n: [spearmanr(R0[n][i], IV[i]).correlation for i in range(12)] for n in R0}
R0["ALL*U"] = R0["ALL"] + R0["U"]; R0["OV1*U"] = R0["OV1"] + R0["U"]
for n in ("ALL*U", "OV1*U"): rho[n] = [spearmanr(R0[n][i], IV[i]).correlation for i in range(12)]
d4 = np.array(rho["ALL"]) - np.array(rho["OV1"]); boot = [d4[rng.integers(0, 12, 12)].mean() for _ in range(10000)]
ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
res["T4"] = dict(mean_rho={n: float(np.mean(v)) for n, v in rho.items()}, diff_ALL_minus_OV1=float(d4.mean()), ci95=ci,
                 PASS=bool(ci[0] > 0 and np.mean(rho["ALL"]) > 0.3))
# ---- stage B
rows = sum([json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "B", "*.json")))], [])
if len(rows) == 32:
    st = [r for r in rows if r["level"] > 0]; ba = [r for r in rows if r["level"] == 0]
    if np.mean([r["grok"] is not None for r in ba]) < 0.5:
        res["T3"] = dict(status="not testable: fewer than half of the original-split runs grokked")
    else:
        key = lambda r: r["k"] in r["logit_keys"]
        a1, a0 = sum(map(key, st)), sum(map(key, ba)); g1 = [r for r in st if r["grok"] is not None]
        p3 = fisher_exact([[a1, len(st) - a1], [a0, len(ba) - a0]], alternative="greater")[1]
        res["T3"] = dict(steered=a1, original=a0, n_each=16, p_fisher=float(p3),
                         steered_grokked_with_key=sum(map(key, g1)), steered_grokked=len(g1),
                         PASS=bool(p3 < 0.01 and sum(map(key, g1)) >= 0.5 * max(len(g1), 1)),
                         secondary_neuron_rule=dict(steered=sum(r["k"] in r["neuron_keys"] for r in st),
                                                    original=sum(r["k"] in r["neuron_keys"] for r in ba)))
    res["T5"] = {lab: dict(grok_rate=float(np.mean([r["grok"] is not None for r in R_])),
                           median_grok=float(np.median([r["grok"] for r in R_ if r["grok"] is not None] or [np.nan])),
                           mean_test_acc=float(np.mean([r["test_acc"] for r in R_])))
                 for lab, R_ in (("steered", st), ("original", ba))}
    res["runs"] = [{k: r[k] for k in ("k", "level", "init", "grok", "logit_keys", "neuron_keys", "test_acc")} for r in rows]
else:
    res["T3"] = f"incomplete ({len(rows)}/32)"
json.dump(res, open(os.path.join(RES, "V7_results.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1, default=float))
