"""Analysis of an init x split factorial (fact/<name>/i*_s*.npz).
usage: python3 fact_analysis.py <name>   -> prints and writes fact/<name>_analysis.json
Winner set W(i,s) = top-3 embedding frequencies at step 500.
Votes (leave-one-out, no fitted parameters):
  init vote  a[i,s,k] = mean over splits s' != s of z(embed500[i,s',k])
  data vote  b[i,s,k] = mean over inits  i' != i of z(embed500[i',s,k])
z() standardizes the 56 log-shares within one run.
"""
import sys, glob, re, json, numpy as np
from math import comb
from scipy.stats import spearmanr
name = sys.argv[1]
files = glob.glob(f"fact/{name}/i*_s*.npz")
cells = {tuple(map(int, re.findall(r"i(\d+)_s(\d+)", f)[0])): f for f in files}
I = sorted({i for i, _ in cells}); S = sorted({s for _, s in cells})
nI, nS = len(I), len(S)
E = np.full((nI, nS, 56), np.nan); A0 = np.full((nI, nS, 56), np.nan)
for (i, s), f in cells.items():
    d = np.load(f); st = list(d["steps"])
    E[I.index(i), S.index(s)] = d["embed"][st.index(500)]
    A0[I.index(i), S.index(s)] = d["attn"][st.index(0)]
def z(x): l = np.log(x); return (l - l.mean(-1, keepdims=True)) / l.std(-1, keepdims=True)
Z = z(E)
W = np.argsort(-E, -1)[..., :3]                                 # winners (0-indexed k)
win = np.zeros_like(E, dtype=bool)
for a in range(nI):
    for b in range(nS): win[a, b, W[a, b]] = True

def hits(score):                                               # top-3 of score vs winners, summed over cells
    top = np.argsort(-score, -1)[..., :3]
    return int(sum(np.isin(top[a, b], W[a, b]).sum() for a in range(nI) for b in range(nS)))

N = nI * nS
h1 = np.array([comb(3, j) * comb(53, 3 - j) / comb(56, 3) for j in range(4)])
null = np.array([1.0])
for _ in range(N): null = np.convolve(null, h1)
def p_chance(h): return float(null[h:].sum())

# leave-one-out votes
init_vote = np.zeros_like(Z); data_vote = np.zeros_like(Z)
for a in range(nI):
    for b in range(nS):
        init_vote[a, b] = np.mean([Z[a, bb] for bb in range(nS) if bb != b], 0)
        data_vote[a, b] = np.mean([Z[aa, b] for aa in range(nI) if aa != a], 0)
def zz(x): return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)
attn0 = zz(np.log(A0))
res = dict(name=name, n_inits=nI, n_splits=nS, cells=N, chance_hits=round(N * 9 / 56, 2))
for label, sc in [("init_vote", init_vote), ("data_vote", data_vote),
                  ("init+data", zz(init_vote) + zz(data_vote)),
                  ("attn_path_at_init", attn0),
                  ("attn+data", attn0 + zz(data_vote))]:
    h = hits(sc); res[label] = dict(hits=h, of=3 * N, p_vs_chance=p_chance(h))

# paired bootstrap over inits and splits for (init+data) - max(init, data)
rng = np.random.default_rng(0)
def cellhits(score):
    top = np.argsort(-score, -1)[..., :3]
    return np.array([[np.isin(top[a, b], W[a, b]).sum() for b in range(nS)] for a in range(nI)])
H = {k: cellhits(v) for k, v in [("init", init_vote), ("data", data_vote), ("both", zz(init_vote) + zz(data_vote)),
                                 ("attn", attn0), ("attn+data", attn0 + zz(data_vote))]}
def boot(a_, b_):
    d = []
    for _ in range(5000):
        ia = rng.integers(0, nI, nI); sb = rng.integers(0, nS, nS)
        d.append(H[a_][np.ix_(ia, sb)].sum() - H[b_][np.ix_(ia, sb)].sum())
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
res["ci_both_minus_init"] = boot("both", "init"); res["ci_both_minus_data"] = boot("both", "data")
res["ci_attn+data_minus_attn"] = boot("attn+data", "attn")

# overlap structure
def ov(a1, b1, a2, b2): return len(set(W[a1, b1]) & set(W[a2, b2]))
same_i = [ov(a, b, a, c) for a in range(nI) for b in range(nS) for c in range(b + 1, nS)]
same_s = [ov(a, b, c, b) for b in range(nS) for a in range(nI) for c in range(a + 1, nI)]
diff = [ov(a, b, c, d) for a in range(nI) for b in range(nS) for c in range(nI) for d in range(nS) if a < c and b != d]
res["overlap"] = dict(same_init_diff_split=float(np.mean(same_i)), same_split_diff_init=float(np.mean(same_s)),
                      diff_both=float(np.mean(diff)), chance=9 / 56)
# how concentrated are split favourites?
cnt = win.sum(0)                                              # [nS,56] wins per split per k
res["max_wins_per_split"] = cnt.max(-1).tolist()
json.dump(res, open(f"fact/{name}_analysis.json", "w"), indent=1)
np.savez(f"fact/{name}_arrays.npz", Z=Z, win=win, data_vote=data_vote, init_vote=init_vote, attn0=attn0, I=I, S=S)
print(json.dumps(res, indent=1))
