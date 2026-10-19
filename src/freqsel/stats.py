import numpy as np
from scipy.stats import rankdata

def zscore_log(x):
    l = np.log(np.maximum(np.asarray(x, float), 1e-30)); return (l - l.mean(-1, keepdims=True)) / l.std(-1, keepdims=True)
def zz(x): return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)
def within_split_spearman(X, Y):
    return float(np.mean([np.corrcoef(rankdata(x), rankdata(y))[0, 1] for x, y in zip(X, Y)]))
def perm_p_within(X, Y, reps=10000, seed=0):
    rng = np.random.default_rng(seed); obs = within_split_spearman(X, Y)
    ra = np.stack([rankdata(x) for x in X]); rb = np.stack([rankdata(y) for y in Y])
    ra = (ra - ra.mean(1, keepdims=True)) / ra.std(1, keepdims=True); rb = (rb - rb.mean(1, keepdims=True)) / rb.std(1, keepdims=True)
    base = np.tile(np.arange(X.shape[1]), (X.shape[0], 1)); cnt = 0
    for _ in range(reps):
        if (np.take_along_axis(ra, rng.permuted(base, axis=1), 1) * rb).mean() >= obs - 1e-12: cnt += 1
    return obs, (cnt + 1) / (reps + 1)
def topk_hits(score, win3):
    top = np.argsort(-score, -1)[..., :3]
    return (top[..., :, None] == win3[..., None, :]).any(-1).sum(-1)
def split_half_reliability(Z):
    """Z [splits, inits, nf] z-scored spectra -> Spearman-Brown corrected reliability of the centred data vote."""
    h1 = Z[:, 0::2].mean(1); h2 = Z[:, 1::2].mean(1); h1 -= h1.mean(0); h2 -= h2.mean(0)
    r = within_split_spearman(h1, h2); return 2 * r / (1 + r)
