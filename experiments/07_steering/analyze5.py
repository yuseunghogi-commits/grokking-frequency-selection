"""Pre-registered analysis for PREREG5 (S1-S4). Written before the runs."""
import os, sys, json, glob, hashlib, itertools, numpy as np
from scipy.stats import spearmanr, fisher_exact
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
I = json.load(open(os.path.join(HERE, "splits_info.json"))); T, L = I["targets"], I["levels"]
RA = os.path.join(HERE, "results", "A"); RB = os.path.join(HERE, "results", "B")
res = dict(prereg5_sha256=hashlib.sha256(open(os.path.join(ROOT, "prereg", "PREREG5.txt"), "rb").read()).hexdigest())
hit = np.full((len(T), len(L), 12), np.nan); prank = np.full((len(T), 12), np.nan)
for a, k in enumerate(T):
    for b, g in enumerate(L):
        for c0 in range(0, 12, 4):
            f = os.path.join(RA, f"k{k}_g{g}_c{c0}.npz")
            if not os.path.exists(f): continue
            d = np.load(f); top = np.argsort(-d["embed200"], -1)[:, :3]
            hit[a, b, c0:c0 + 4] = (top == k - 1).any(1)
            if g == 0: prank[a, c0:c0 + 4] = (d["path0"] > d["path0"][:, [k - 1]]).sum(1)
if not np.isnan(hit).any():
    rate = hit.mean(2)                                                      # [targets, levels]
    diff = rate[:, -1] - rate[:, 0]
    flips = np.array(list(itertools.product([1, -1], repeat=len(T))))
    p1 = float(((flips * diff).mean(1) >= diff.mean() - 1e-12).mean())
    res["S1"] = dict(rate_by_level=rate.mean(0).tolist(), rate_by_target_level=rate.tolist(), mean_diff=float(diff.mean()),
                     p_signflip=p1, PASS=bool(rate[:, -1].mean() >= 0.5 and p1 < 0.01))
    def msp(R): return float(np.nanmean([spearmanr(L, r).correlation if np.std(r) > 0 else 0.0 for r in R]))
    obs = msp(rate); rng = np.random.default_rng(0)
    null = np.array([msp(np.array([rng.permutation(r) for r in rate])) for _ in range(10000)])
    p2 = float(((null >= obs - 1e-12).sum() + 1) / (len(null) + 1))
    res["S2"] = dict(mean_within_target_spearman=obs, p=p2, PASS=bool(obs > 0.5 and p2 < 0.01))
    # exploratory: success vs the init's own preference for k* (rank of k* in OV-path spectrum at init, 0 = favourite)
    hs = hit[:, -1].ravel(); pr = prank.ravel()
    res["exploratory_vs_init_rank"] = dict(hit_rate_init_rank_top_half=float(hs[pr < 28].mean()),
                                           hit_rate_init_rank_bottom_half=float(hs[pr >= 28].mean()))
else:
    res["S1"] = res["S2"] = "incomplete"
rows = sum([json.load(open(f)) for f in sorted(glob.glob(os.path.join(RB, "*.json")))], [])
if len(rows) == 32:
    def key(r): return r["k"] in r["final"]["keys"]
    st = [r for r in rows if r["level"] > 0]; ba = [r for r in rows if r["level"] == 0]
    a1, a0 = sum(map(key, st)), sum(map(key, ba))
    p3 = fisher_exact([[a1, len(st) - a1], [a0, len(ba) - a0]], alternative="greater")[1]
    res["S3"] = dict(steered_key=a1, base_key=a0, n_each=len(st), p_fisher=float(p3),
                     PASS=bool(p3 < 0.01 and a1 >= 0.5 * len(st)),
                     runs=[dict(k=r["k"], level=r["level"], init=r["init"], grok=r["grok"], keys=r["final"]["keys"],
                                test_acc=r["final"]["test_acc"]) for r in rows])
    res["S4"] = {lab: dict(grok_rate=float(np.mean([r["grok"] is not None for r in R])),
                           median_grok=float(np.median([r["grok"] for r in R if r["grok"] is not None] or [np.nan])),
                           mean_final_test_acc=float(np.mean([r["final"]["test_acc"] for r in R])))
                 for lab, R in (("steered", st), ("base", ba))}
else:
    res["S3"] = f"incomplete ({len(rows)}/32)"
json.dump(res, open(os.path.join(HERE, "results", "V5_results.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "S3" or isinstance(v, str)}, indent=1, default=float))
if isinstance(res.get("S3"), dict): print({k: v for k, v in res["S3"].items() if k != "runs"})
