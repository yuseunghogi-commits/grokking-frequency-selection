"""Construct training splits that steer the data vote toward a target frequency by maximising D(k_target).
D here is the two-sided label-sharing statistic used in v3 (operand a and operand b)."""
import numpy as np, torch
P = 113; NF = 56; W = 2 * np.pi * np.arange(1, NF + 1) / P

def split_perm(seed, frac=0.3):
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(P * P, generator=g).numpy()
    n = int(frac * P * P); return perm[:n], perm[n:]

def D_of(idx):
    a, b = idx // P, idx % P; c = (a + b) % P; out = np.zeros(NF)
    for s in (a, b):
        S = np.zeros((NF, P), complex); np.add.at(S.T, c, np.exp(1j * np.outer(W, s)).T); out += (np.abs(S) ** 2).sum(1)
    return out

def objective(logD, kt):            # kt is 0-based
    return logD[kt] - np.delete(logD, kt).mean()

def steer(seed, kt, target_gain, sign=+1, frac=0.3, max_prop=400000, rng_seed=0):
    """Greedy pair swaps (train <-> held-out) until sign*objective has risen by target_gain over the start split.
    Returns train indices and a log. Incremental: keep S_a, S_b [NF, P] complex sums per answer c."""
    tr, te = split_perm(seed, frac); tr = tr.copy(); te = te.copy(); rng = np.random.default_rng(rng_seed)
    def contrib(i):
        a, b = i // P, i % P; return (a + b) % P, np.exp(1j * W * a), np.exp(1j * W * b)
    Sa = np.zeros((NF, P), complex); Sb = np.zeros((NF, P), complex)
    for i in tr:
        c, ea, eb = contrib(i); Sa[:, c] += ea; Sb[:, c] += eb
    def logD(): return np.log((np.abs(Sa) ** 2).sum(1) + (np.abs(Sb) ** 2).sum(1))
    J0 = sign * objective(logD(), kt); J = J0; acc = 0
    Dsum = (np.abs(Sa) ** 2).sum(1) + (np.abs(Sb) ** 2).sum(1)
    for t in range(max_prop):
        if J - J0 >= target_gain: break
        u = rng.integers(len(tr)); v = rng.integers(len(te)); i, j = tr[u], te[v]
        ci, ea_i, eb_i = contrib(i); cj, ea_j, eb_j = contrib(j)
        cols = {ci, cj}; old = sum((np.abs(Sa[:, c]) ** 2 + np.abs(Sb[:, c]) ** 2) for c in cols)
        Sa[:, ci] -= ea_i; Sb[:, ci] -= eb_i; Sa[:, cj] += ea_j; Sb[:, cj] += eb_j
        new = sum((np.abs(Sa[:, c]) ** 2 + np.abs(Sb[:, c]) ** 2) for c in cols)
        Dn = Dsum - old + new; Jn = sign * objective(np.log(Dn), kt)
        if Jn > J: J, Dsum = Jn, Dn; tr[u], te[v] = j, i; acc += 1
        else: Sa[:, ci] += ea_i; Sb[:, ci] += eb_i; Sa[:, cj] -= ea_j; Sb[:, cj] -= eb_j
    return np.sort(tr), dict(J0=float(J0), J=float(J), gain=float(J - J0), accepted=acc, proposals=t + 1)
