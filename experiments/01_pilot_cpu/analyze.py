"""Reverse-engineer the modular-addition transformer.
usage: python3 analyze.py <ckpt.pt>   -> prints findings, writes figs/ and results.json
"""
import torch, torch.nn.functional as F, numpy as np, json, sys, os
from model import OneLayer, dataset, P
torch.set_grad_enabled(False)
p = P

def fourier_basis(p):
    x = torch.arange(p).float()
    rows, names = [torch.ones(p)], ["const"]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * np.pi * k * x / p), torch.sin(2 * np.pi * k * x / p)]
        names += [f"cos{k}", f"sin{k}"]
    B = torch.stack(rows)
    return B / B.norm(dim=1, keepdim=True), names

B, names = fourier_basis(p)               # [p,p] orthonormal, row i = basis fn i

def load(path):
    m = OneLayer(); m.load_state_dict(torch.load(path)); m.eval(); return m

def embed_freq_norms(m):
    WE = m.W_E[:p]                       # [p,d]
    c = B @ WE                           # [p,d] coefficient per basis fn
    n = c.norm(dim=1) ** 2
    return torch.stack([n[2*k-1] + n[2*k] for k in range(1, p//2 + 1)]).sqrt()  # [56]

def key_freqs(m, min_share=0.03):
    """Key frequencies = those that are the dominant frequency of >= min_share of live MLP neurons."""
    G = neuron_grid(m)[-1]
    C = to_fourier2d(G); C[0, 0] = 0
    per = torch.stack([((C * keep_mask([k])[..., None])**2).sum((0, 1)) for k in range(1, p//2 + 1)])
    alive = G.reshape(-1, G.shape[-1]).std(0) > 1e-3
    dom = per.argmax(0)[alive] + 1
    return sorted(int(k) for k in dom.unique() if (dom == k).float().mean() >= min_share)

def neuron_grid(m):
    x, y, tr, te = dataset()
    cache = {}
    logits = m(x, cache=cache)
    return x, y, tr, te, logits, cache["neur"].reshape(p, p, -1)

def to_fourier2d(G):                     # G [p,p,N] -> coef [p,p,N]
    return torch.einsum("ia,jb,abn->ijn", B, B, G)

def from_fourier2d(C):
    return torch.einsum("ia,jb,ijn->abn", B, B, C)

def keep_mask(keys):
    idx = [0] + [i for k in keys for i in (2*k-1, 2*k)]
    M = torch.zeros(p, p)
    for i in idx:
        for j in idx:
            M[i, j] = 1
    return M

def loss_with_neurons(m, x, y, sel, grid_new):
    flat = grid_new.reshape(p * p, -1)
    lg = m(x, hook=lambda n: flat)
    return F.cross_entropy(lg[sel], y[sel]).item(), (lg[sel].argmax(-1) == y[sel]).float().mean().item()

def analyze(path, figs=True, quiet=False, keys=None):
    m = load(path)
    x, y, tr, te, logits, G = neuron_grid(m)
    keys = keys or key_freqs(m)
    out = dict(ckpt=path, key_freqs=keys)
    out["test_loss"] = F.cross_entropy(logits[te], y[te]).item()
    out["test_acc"] = (logits[te].argmax(-1) == y[te]).float().mean().item()
    fn = embed_freq_norms(m)
    out["embed_freq_norms"] = fn.tolist()
    out["embed_key_share"] = (fn[[k-1 for k in keys]]**2).sum().item() / (fn**2).sum().item()

    # --- neurons: variance explained by key-frequency 2D Fourier terms
    C = to_fourier2d(G)
    M = keep_mask(keys)[..., None]
    Cc = C.clone(); Cc[0, 0] = 0               # drop mean
    tot = (Cc**2).sum().item()
    out["neuron_var_explained_by_keys"] = ((Cc * M)**2).sum().item() / tot

    # each neuron's dominant frequency
    per_n = []
    for k in range(1, p//2 + 1):
        Mk = keep_mask([k])[..., None]
        per_n.append(((Cc * Mk)**2).sum((0, 1)))
    per_n = torch.stack(per_n)                         # [56, N]
    nvar = (Cc**2).sum((0, 1)) + 1e-12
    dom = per_n.argmax(0) + 1
    domfrac = per_n.max(0).values / nvar
    alive = G.reshape(-1, G.shape[-1]).std(0) > 1e-3
    out["n_alive_neurons"] = int(alive.sum())
    out["neuron_cluster_sizes"] = {int(k): int(((dom == k) & alive).sum()) for k in keys}
    out["frac_alive_neurons_in_key_cluster"] = float(sum(((dom == k) & alive).sum() for k in keys) / alive.sum())
    out["median_single_freq_var_frac"] = float(domfrac[alive].median())

    # --- trig-identity check: quadratic terms split into cos/sin(w(a+b)) vs cos/sin(w(a-b)) parts
    # cos(a+b)=cc-ss, sin(a+b)=sc+cs ; cos(a-b)=cc+ss, sin(a-b)=sc-cs   (c,s indices for a then b)
    e_sum = e_diff = e_lin = 0.0
    for k in keys:
        c, s = 2*k-1, 2*k
        cc, ss, cs, sc = Cc[c, c], Cc[s, s], Cc[c, s], Cc[s, c]
        e_sum += (((cc - ss)**2 + (sc + cs)**2) / 2).sum().item()
        e_diff += (((cc + ss)**2 + (sc - cs)**2) / 2).sum().item()
        e_lin += (Cc[c, 0]**2 + Cc[s, 0]**2 + Cc[0, c]**2 + Cc[0, s]**2).sum().item()
    out["key_energy_a_plus_b"] = e_sum / tot
    out["key_energy_a_minus_b"] = e_diff / tot
    out["key_energy_linear_a_or_b"] = e_lin / tot
    # --- logits: fraction of logit variance explained by cos(w_k(a+b-c)) for key k
    L = logits.reshape(p, p, p) - logits.reshape(p, p, p).mean(-1, keepdim=True)
    a = torch.arange(p)[:, None, None]; b = torch.arange(p)[None, :, None]; c = torch.arange(p)[None, None, :]
    feats = []
    for k in keys:
        w = 2 * np.pi * k / p
        feats += [torch.cos(w * (a + b - c)).float(), torch.sin(w * (a + b - c)).float()]
    Xf = torch.stack([f.reshape(-1) for f in feats], 1)
    sol = torch.linalg.lstsq(Xf, L.reshape(-1, 1)).solution
    resid = L.reshape(-1, 1) - Xf @ sol
    out["logit_var_explained_cos_a_plus_b_minus_c"] = 1 - (resid**2).sum().item() / (L**2).sum().item()
    out["logit_coefs_per_key"] = {int(k): float((sol[2*i]**2 + sol[2*i+1]**2).sqrt()) for i, k in enumerate(keys)}

    # --- causal ablations on MLP neurons (test set)
    out["loss_clean"] = out["test_loss"]
    out["restricted"] = loss_with_neurons(m, x, y, te, from_fourier2d(C * M))
    Cx = C.clone(); Cx = Cx * (1 - M); Cx[0, 0] = C[0, 0]
    out["excluded"] = loss_with_neurons(m, x, y, te, from_fourier2d(Cx))
    out["mean_ablate_mlp"] = loss_with_neurons(m, x, y, te, G.mean((0, 1), keepdim=True).expand_as(G))
    # control: remove the same number of random NON-key frequencies
    rng = np.random.default_rng(0)
    others = [k for k in range(1, p//2 + 1) if k not in keys]
    ctrl = []
    for t in range(5 if len(others) >= len(keys) else 0):
        rk = list(rng.choice(others, len(keys), replace=False))
        Mr = keep_mask([int(r) for r in rk])[..., None]
        Cr = C * (1 - Mr); Cr[0, 0] = C[0, 0]
        ctrl.append(loss_with_neurons(m, x, y, te, from_fourier2d(Cr))[0])
    out["excluded_random_control_loss"] = ctrl

    if not quiet:
        print(json.dumps({k: v for k, v in out.items() if k != "embed_freq_norms"}, indent=1))
    if figs:
        make_figs(m, G, fn, keys, dom, domfrac, alive, attn=None)
    return out

def make_figs(m, G, fn, keys, dom, domfrac, alive, attn):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figs", exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    ACC, MUT = "#c2410c", "#9ca3af"
    # embedding Fourier spectrum
    f, ax = plt.subplots(figsize=(8, 3.2))
    cols = [ACC if k in keys else MUT for k in range(1, len(fn) + 1)]
    ax.bar(range(1, len(fn) + 1), fn.numpy(), color=cols)
    for k in keys:
        ax.text(k, fn[k-1].item() * 1.02, str(k), ha="center", va="bottom", fontsize=10, color=ACC)
    ax.set_xlabel("frequency k  (period 113/k)"); ax.set_ylabel("embedding norm")
    f.tight_layout(); f.savefig("figs/embed_spectrum.png", dpi=160); plt.close(f)
    # example neurons: one per key freq
    ks = keys[:4]
    f, axs = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 3.2))
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, ks):
        cand = torch.where((dom == k) & alive)[0]
        n = cand[domfrac[cand].argmax()].item()
        ax.imshow(G[:, :, n].numpy(), cmap="magma", origin="lower")
        ax.set_title(f"neuron {n}  (k={k})", fontsize=10); ax.set_xlabel("b"); ax.set_ylabel("a")
        ax.set_xticks([0, 56, 112]); ax.set_yticks([0, 56, 112])
    f.tight_layout(); f.savefig("figs/neurons.png", dpi=160); plt.close(f)

if __name__ == "__main__":
    r = analyze(sys.argv[1])
    json.dump(r, open("results.json", "w"), indent=1)
