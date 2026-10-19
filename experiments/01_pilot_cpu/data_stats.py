"""Closed-form statistics of a training split, per frequency k (no training involved).
usage: python3 data_stats.py <split0> <n>  -> fact/datastats_<split0>_<n>.npz  (array [n,56] per statistic)
D1 sum coherence      |sum_T e^{i w (a+b)}|          (= Fourier of the label histogram, since y=a+b mod p)
D2 diff coherence     |sum_T e^{i w (a-b)}|
D3 marginal coherence |sum_T e^{i w a}|^2 + |sum_T e^{i w b}|^2
D4 diagonal leakage   sum_{j != k} |<psi_k, psi_j>_T|^2 , psi_k = e^{i w_k (a+b)}
D5 Gram: lambda_min and condition number of the 8 degree-2 features at k on T
D5r residual: fraction of cos/sin(w_k(a+b)) NOT explained by the degree-2 features of all other frequencies on T
"""
import sys, numpy as np
from model import dataset, P
p = P
s0, n = int(sys.argv[1]), int(sys.argv[2])
ks = np.arange(1, 57); w = 2 * np.pi * ks / p
out = {k: np.zeros((n, 56)) for k in ["D1", "D2", "D3", "D4", "D5min", "D5cond", "D5r"]}
for j, s in enumerate(range(s0, s0 + n)):
    x, y, tr, te = dataset(seed=s)
    a = x[tr, 0].numpy().astype(float); b = x[tr, 1].numpy().astype(float)
    N = len(a)
    E = lambda ph: np.exp(1j * np.outer(w, ph))                    # [56,N]
    Es, Ed, Ea, Eb = E(a + b), E(a - b), E(a), E(b)
    out["D1"][j] = np.abs(Es.sum(1)); out["D2"][j] = np.abs(Ed.sum(1))
    out["D3"][j] = np.abs(Ea.sum(1)) ** 2 + np.abs(Eb.sum(1)) ** 2
    # leakage: C(m) = sum e^{i 2pi m (a+b)/p} for all m mod p
    m = np.arange(p); Cm = np.exp(1j * 2 * np.pi * np.outer(m, a + b) / p).sum(1)
    for t, k in enumerate(ks):
        js = [jj for jj in list(range(1, 57)) + [-q for q in range(1, 57)] if jj != k]
        out["D4"][j, t] = sum(abs(Cm[(k - jj) % p]) ** 2 for jj in js)
    ca, sa, cb, sb = np.cos(np.outer(w, a)), np.sin(np.outer(w, a)), np.cos(np.outer(w, b)), np.sin(np.outer(w, b))
    feats = np.stack([ca, sa, cb, sb, ca * cb, ca * sb, sa * cb, sa * sb], 1)   # [56,8,N]
    for t in range(56):
        G = feats[t] @ feats[t].T / N; ev = np.linalg.eigvalsh(G)
        out["D5min"][j, t] = ev[0]; out["D5cond"][j, t] = ev[-1] / ev[0]
    allf = feats.reshape(56 * 8, N).T                                           # [N, 448]
    for t in range(56):
        others = np.delete(allf, slice(8 * t, 8 * t + 8), axis=1)
        tgt = np.stack([np.cos(w[t] * (a + b)), np.sin(w[t] * (a + b))], 1)
        coef, *_ = np.linalg.lstsq(others, tgt, rcond=None)
        r = tgt - others @ coef
        out["D5r"][j, t] = (r ** 2).sum() / (tgt ** 2).sum()
    print("split", s, "done", flush=True)
np.savez(f"fact/datastats_{s0}_{n}.npz", **out)
