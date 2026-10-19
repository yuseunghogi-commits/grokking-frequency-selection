"""Train M independent OneLayer models at once (stacked weights, one einsum per op).
Each model has its own init and its own training split; full-batch AdamW, identical math to train.py.
"""
import torch, torch.nn.functional as F, numpy as np
from model import OneLayer, dataset, P
torch.set_grad_enabled(True)
p = P
NAMES = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "b_in", "W_out", "b_out", "W_U"]

def init_state(seed):
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in OneLayer().state_dict().items()}

def fourier_B():
    x = torch.arange(p).float(); rows = [torch.ones(p)]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * np.pi * k * x / p), torch.sin(2 * np.pi * k * x / p)]
    Bm = torch.stack(rows); return Bm / Bm.norm(dim=1, keepdim=True)
B = fourier_B()

def per_freq(v):                       # v [M,p,d] -> [M,56] share
    q = (torch.einsum("ia,mad->mid", B, v) ** 2).sum(-1)
    fn = q[:, 1::2] + q[:, 2::2]
    return fn / fn.sum(-1, keepdim=True)

def forward(W, X):                     # W: dict of [M,...]; X [M,N,3] long
    M = X.shape[0]
    r = torch.stack([W["W_E"][m][X[m]] for m in range(M)]) + W["W_pos"][:, None]   # [M,N,3,d]
    dh = W["W_Q"].shape[-1]
    q = torch.einsum("mnd,mhde->mnhe", r[:, :, -1], W["W_Q"])
    k = torch.einsum("mnpd,mhde->mnhpe", r, W["W_K"])
    v = torch.einsum("mnpd,mhde->mnhpe", r, W["W_V"])
    a = (torch.einsum("mnhe,mnhpe->mnhp", q, k) / dh ** 0.5).softmax(-1)
    z = torch.einsum("mnhp,mnhpe->mnhe", a, v)
    h = r[:, :, -1] + torch.einsum("mnhe,mhed->mnd", z, W["W_O"])
    n = F.relu(torch.einsum("mnd,mdf->mnf", h, W["W_in"]) + W["b_in"][:, None])
    h = h + torch.einsum("mnf,mfd->mnd", n, W["W_out"]) + W["b_out"][:, None]
    return torch.einsum("mnd,mdc->mnc", h, W["W_U"])

def train_batch(init_seeds, split_seeds, steps=500, record=(0, 50, 100, 200, 300, 500), wd=1.0, frac=0.3):
    M = len(init_seeds)
    sts = [init_state(s) for s in init_seeds]
    W = {k: torch.stack([st[k] for st in sts]).requires_grad_(True) for k in NAMES}
    Xs, Ys = [], []
    for sp in split_seeds:
        x, y, tr, te = dataset(frac=frac, seed=sp); Xs.append(x[tr]); Ys.append(y[tr])
    X, Y = torch.stack(Xs), torch.stack(Ys)
    opt = torch.optim.AdamW(list(W.values()), lr=1e-3, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1.0))
    rec = {}
    for step in range(steps + 1):
        logits = forward(W, X)
        # sum of per-model mean losses: gradients per model identical to separate training (Adam is per-element)
        loss = F.cross_entropy(logits.reshape(-1, p), Y.reshape(-1), reduction="none").reshape(M, -1).mean(1)
        if step in record:
            with torch.no_grad():
                rec[step] = dict(
                    embed=per_freq(W["W_E"][:, :p]).numpy(),
                    attn=per_freq(torch.einsum("mnd,mhde,mhef->mnf", W["W_E"][:, :p], W["W_V"], W["W_O"])).numpy(),
                    train_acc=(logits.argmax(-1) == Y).float().mean(1).numpy())
        opt.zero_grad(); loss.sum().backward(); opt.step(); sched.step()
    return rec
