"""Two-layer version of the v1 transformer (no LayerNorm), batched over M models with stacked weights.

Layer 1: causal attention + MLP at all three positions (a, b, =). Layer 2: attention from the '=' position over all
positions + MLP at '='. Same widths, init scales and optimiser as the 1-layer model in gpu_grok.py.
Uses gpu_grok for the task, splits, Fourier basis and StackedAdamW, so splits and spectra are identical to v1-v5."""
import math, numpy as np, torch, torch.nn.functional as F
import gpu_grok as G
P, NF, DEV = G.P, G.NF, G.DEV
D_MODEL, H, D_MLP = 128, 4, 512

def init_state2(seed):
    torch.manual_seed(seed); d, h, dh = D_MODEL, H, D_MODEL // H
    W = {"W_E": torch.randn(P + 1, d) / d**0.5, "W_pos": torch.randn(3, d) / d**0.5}
    for l in (1, 2):
        W[f"W_Q{l}"] = torch.randn(h, d, dh) / d**0.5; W[f"W_K{l}"] = torch.randn(h, d, dh) / d**0.5
        W[f"W_V{l}"] = torch.randn(h, d, dh) / d**0.5; W[f"W_O{l}"] = torch.randn(h, dh, d) / d**0.5
        W[f"W_in{l}"] = torch.randn(d, D_MLP) / d**0.5; W[f"b_in{l}"] = torch.zeros(D_MLP)
        W[f"W_out{l}"] = torch.randn(D_MLP, d) / D_MLP**0.5; W[f"b_out{l}"] = torch.zeros(d)
    W["W_U"] = torch.randn(d, P) / d**0.5
    return W

_CAUSAL = torch.tril(torch.ones(3, 3, dtype=torch.bool))
def forward2(W, X, return_cache=False):
    """X [M,N,3] -> logits [M,N,P]."""
    M = X.shape[0]; mi = torch.arange(M, device=X.device)[:, None, None]; dh = D_MODEL // H
    r = W["W_E"][mi, X] + W["W_pos"][:, None]                                         # [M,N,3,d]
    q = torch.einsum("mnpd,mhde->mnhpe", r, W["W_Q1"]); k = torch.einsum("mnpd,mhde->mnhpe", r, W["W_K1"])
    v = torch.einsum("mnpd,mhde->mnhpe", r, W["W_V1"])
    s = torch.einsum("mnhqe,mnhke->mnhqk", q, k) / dh**0.5
    s = s.masked_fill(~_CAUSAL.to(X.device), float("-inf")).softmax(-1)
    r = r + torch.einsum("mnhqe,mhed->mnqd", torch.einsum("mnhqk,mnhke->mnhqe", s, v), W["W_O1"])
    n1 = F.relu(torch.einsum("mnpd,mdf->mnpf", r, W["W_in1"]) + W["b_in1"][:, None, None])
    r = r + torch.einsum("mnpf,mfd->mnpd", n1, W["W_out1"]) + W["b_out1"][:, None, None]
    q2 = torch.einsum("mnd,mhde->mnhe", r[:, :, -1], W["W_Q2"]); k2 = torch.einsum("mnpd,mhde->mnhpe", r, W["W_K2"])
    v2 = torch.einsum("mnpd,mhde->mnhpe", r, W["W_V2"])
    a2 = (torch.einsum("mnhe,mnhpe->mnhp", q2, k2) / dh**0.5).softmax(-1)
    x = r[:, :, -1] + torch.einsum("mnhe,mhed->mnd", torch.einsum("mnhp,mnhpe->mnhe", a2, v2), W["W_O2"])
    n2 = F.relu(torch.einsum("mnd,mdf->mnf", x, W["W_in2"]) + W["b_in2"][:, None])
    x = x + torch.einsum("mnf,mfd->mnd", n2, W["W_out2"]) + W["b_out2"][:, None]
    logits = torch.einsum("mnd,mdc->mnc", x, W["W_U"])
    return (logits, dict(n1=n1[:, :, -1], n2=n2)) if return_cache else logits

# ---- readouts ------------------------------------------------------------------------------------------------------
def OV(W, l): return torch.einsum("mhde,mhef->mdf", W[f"W_V{l}"], W[f"W_O{l}"])
def path_readouts(W):
    """Fourier share per frequency of candidate token maps at init. [M,56] each.
    OV1: layer-1 OV (the 1-layer analogue). ALL: E (OV1 + OV2 + OV1 OV2), the linear token->'=' maps through both
    attention layers (ignoring MLP1)."""
    E = W["W_E"][:, :P]; o1, o2 = OV(W, 1), OV(W, 2)
    return dict(OV1=G.per_freq_share(E @ o1), OV2=G.per_freq_share(E @ o2),
                ALL=G.per_freq_share(E @ (o1 + o2 + o1 @ o2)), E=G.per_freq_share(E))
def unembed_share(W):                                   # Fourier share of W_U over answer classes [M,56]
    return G.per_freq_share(W["W_U"].transpose(1, 2))

@torch.no_grad()
def logit_keys(Wm, thresh=0.05):
    """Key frequencies of a single model from its full-grid logits: energy share of the cos/sin k(a+b-c) components.
    Returns (keys 1-based with share >= thresh, share vector [56])."""
    L = forward2(Wm, G.full_grid()[None])[0].reshape(P, P, P).double().cpu().numpy()
    Fh = np.fft.fftn(L)                                  # index [ka, kb, kc]
    k = np.arange(1, NF + 1); e = np.abs(Fh[k, k, (-k) % P]) ** 2
    share = e / e.sum(); return [int(i + 1) for i in np.nonzero(share >= thresh)[0]], share.tolist()

@torch.no_grad()
def neuron_keys(Wm, min_share=0.03):
    """Neuron-cluster rule of v1 applied to layer-2 MLP neurons at '=' (secondary key definition)."""
    _, c = forward2(Wm, G.full_grid()[None], return_cache=True)
    Gn = c["n2"][0].reshape(P, P, -1)
    C = torch.einsum("ia,jb,abn->ijn", G.B, G.B, Gn); C[0, 0] = 0
    per = torch.einsum("kij,ijn->kn", G.freq_mask_2d(), C ** 2)
    alive = Gn.reshape(-1, Gn.shape[-1]).std(0) > 1e-3; dom = per.argmax(0) + 1; n = int(alive.sum())
    return [k for k in range(1, NF + 1) if n and float(((dom == k) & alive).sum()) / n >= min_share]

# ---- training ------------------------------------------------------------------------------------------------------
def train_batch2(specs, steps, record=(), eval_every=0, custom_train_idx=None, stop_after_grok=None, frac=0.3, verbose=False):
    """specs: list of dict(init=, split=). Returns dict(rec{step:{embed, path0?}}, grok, final, curves, last_step)."""
    M = len(specs); sts = [init_state2(s["init"]) for s in specs]
    W = {k: torch.stack([st[k] for st in sts]).to(DEV).requires_grad_(True) for k in sts[0]}
    a, b = G.grid_pairs(); y_all = G.labels("add", a, b).to(DEV); Gd = G.full_grid()
    trs, tes = [], []
    for j, s in enumerate(specs):
        tr, te = G.split_indices(s["split"], frac)
        if custom_train_idx is not None:
            tr = custom_train_idx[j]; mask = torch.ones(P * P, dtype=torch.bool); mask[tr] = False
            te = torch.nonzero(mask).flatten()
        trs.append(tr); tes.append(te)
    TR = torch.stack(trs).to(DEV); X = Gd[TR]; Y = y_all[TR]
    opt = G.StackedAdamW(W, 1e-3, torch.ones(M, device=DEV))
    rec, curves, grok = {}, [], [None] * M; record = set(record)
    with torch.no_grad():
        rd = {k: v.cpu().numpy() for k, v in path_readouts(W).items()}; rd["U"] = unembed_share(W).cpu().numpy()
    for step in range(steps + 1):
        logits = forward2(W, X)
        lossv = F.cross_entropy(logits.reshape(-1, P), Y.reshape(-1), reduction="none").reshape(M, -1).mean(1)
        if step in record:
            with torch.no_grad(): rec[step] = dict(embed=G.per_freq_share(W["W_E"][:, :P]).cpu().numpy())
        if eval_every and step % eval_every == 0:
            with torch.no_grad():
                row = dict(step=step, test_acc=[])
                for m0 in range(0, M, 8):
                    sl = slice(m0, min(M, m0 + 8))
                    lg = forward2({k: v[sl] for k, v in W.items()}, Gd.to(DEV)[None].expand(sl.stop - sl.start, -1, -1))
                    for j, m in enumerate(range(sl.start, sl.stop)):
                        te = tes[m].to(DEV); row["test_acc"].append(float((lg[j][te].argmax(-1) == y_all[te]).float().mean()))
                curves.append(row)
                if verbose: print('eval', step, [round(x, 3) for x in row['test_acc']], flush=True)
                for m in range(M):
                    if grok[m] is None and row["test_acc"][m] >= 0.9: grok[m] = step
        if stop_after_grok is not None and all(g is not None and step >= g + stop_after_grok for g in grok): break
        if step == steps: break
        lossv.sum().backward(); opt.step(G.lr_at(step))
    final = None
    if eval_every:
        final = []
        for m in range(M):
            Wm = {k: v[m:m + 1].detach() for k, v in W.items()}
            lk, sh = logit_keys(Wm)
            final.append(dict(logit_keys=lk, logit_share=sh, neuron_keys=neuron_keys(Wm),
                              test_acc=curves[-1]["test_acc"][m]))
    return dict(rec=rec, readouts0=rd, grok=grok, final=final, curves=curves, last_step=step)
