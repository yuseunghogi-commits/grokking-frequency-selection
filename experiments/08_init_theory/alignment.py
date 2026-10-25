"""Exploratory: alignment of the initial embedding vs the early data-driven update with the output pathway (theory_init_side.md T3)."""
import os, sys, numpy as np, torch, torch.nn.functional as F
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(H)); R = os.path.join(ROOT, "experiments", "02_kaggle_v1"); os.environ["GROK_OUT"] = R + "/results"; sys.path.insert(0, R); import gpu_grok as G
torch.set_num_threads(1); P = 113
ph = torch.exp(-2j * np.pi * torch.outer(torch.arange(1, 57), torch.arange(P)).double() / P)   # [56, p]
def fc(M): return ph @ M.double().to(torch.complex128)                                       # [56, d]
def cosc(x, y): return (torch.abs((x.conj() * y).sum(-1)) / (x.abs().pow(2).sum(-1).sqrt() * y.abs().pow(2).sum(-1).sqrt())).numpy()
inits = [6000, 6001, 6002, 6003]; split = 5000; M = len(inits)
sts = [G.init_state(i) for i in inits]; W = {k: torch.stack([s[k] for s in sts]).requires_grad_(True) for k in sts[0]}
tr, _ = G.split_indices(split); X = G.full_grid()[tr][None].expand(M, -1, -1); a, b = G.grid_pairs(); Y = G.labels("add", a, b)[tr][None].expand(M, -1)
opt = G.StackedAdamW(W, 1e-3, torch.ones(M)); E0 = W["W_E"][:, :P].detach().clone(); snaps = {}
for t in range(31):
    if t in (5, 10, 20, 30): snaps[t] = W["W_E"][:, :P].detach().clone()
    if t == 30: break
    loss = F.cross_entropy(G.forward(W, X).reshape(-1, P), Y.reshape(-1), reduction="none").reshape(M, -1).mean(1).sum(); loss.backward(); opt.step(G.lr_at(t))
for m in range(M):
    U = sts[m]["W_U"].double(); OV = torch.einsum("hde,hef->df", sts[m]["W_V"], sts[m]["W_O"]).double()
    uk = fc(U.T)                                   # [56, d] output direction per frequency (classes)
    targets = {"W_U": uk, "OV.W_U": (OV.to(torch.complex128) @ uk.T).T}
    e0 = fc(E0[m])
    for name, y in targets.items():
        row = [f"init E0: {cosc(e0, y).mean():.3f}/{cosc(e0, y.conj()).mean():.3f}"]
        for t in (5, 10, 30):
            dE = fc(snaps[t][m] - E0[m]); row.append(f"dE@{t}: {cosc(dE, y).mean():.3f}/{cosc(dE, y.conj()).mean():.3f}")
        print(inits[m], f"{name:7s}", " | ".join(row))
print("chance ~", round(float(np.sqrt(np.pi / 4 / 128)), 3), "(mean |cos| of random complex vectors in C^128)")
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(ROOT, "experiments", "07_steering")); from steer import D_of, split_perm
D = np.log(D_of(split_perm(split)[0]))
print("per-frequency relations at step 10 (Spearman across 56 k):")
for m in range(M):
    OV = torch.einsum("hde,hef->df", sts[m]["W_V"], sts[m]["W_O"]).double(); uk = fc(sts[m]["W_U"].double().T)
    y = (OV.to(torch.complex128) @ uk.T).T; dE = fc(snaps[10][m] - E0[m])
    al = np.maximum(cosc(dE, y), cosc(dE, y.conj())); amp = np.log(dE.abs().pow(2).sum(-1).numpy())
    print(f"  init {inits[m]}: rho(|dE_k|^2, D) {spearmanr(amp, D).correlation:+.2f} | rho(align_k, D) {spearmanr(al, D).correlation:+.2f} | rho(align_k, |dE_k|^2) {spearmanr(al, amp).correlation:+.2f}")
