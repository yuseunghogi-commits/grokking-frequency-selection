import math, torch, torch.nn.functional as F
from .models import init_state
from .spectra import freq_share, path_share

def forward(W, X, arch="attn"):
    M = X.shape[0]; mi = torch.arange(M, device=X.device)[:, None, None]
    if arch == "attn":
        r = W["W_E"][mi, X] + W["W_pos"][:, None]; dh = W["W_Q"].shape[-1]
        q = torch.einsum("mnd,mhde->mnhe", r[:, :, -1], W["W_Q"])
        k = torch.einsum("mnpd,mhde->mnhpe", r, W["W_K"]); v = torch.einsum("mnpd,mhde->mnhpe", r, W["W_V"])
        a = (torch.einsum("mnhe,mnhpe->mnhp", q, k) / dh**0.5).softmax(-1)
        mid = r[:, :, -1] + torch.einsum("mnhe,mhed->mnd", torch.einsum("mnhp,mnhpe->mnhe", a, v), W["W_O"])
        pre = torch.einsum("mnd,mdf->mnf", mid, W["W_in"]) + W["b_in"][:, None]
    else:
        mid = None; pre = torch.einsum("mnd,mdf->mnf", W["W_E"][mi, X[:, :, :2]].flatten(2), W["W_in"]) + W["b_in"][:, None]
    out = torch.einsum("mnf,mfd->mnd", F.relu(pre), W["W_out"]) + W["b_out"][:, None]
    if mid is not None: out = out + mid
    return torch.einsum("mnd,mdc->mnc", out, W["W_U"])

class StackedAdamW:
    """torch.optim.AdamW semantics with a per-model weight-decay vector."""
    def __init__(self, W, wd, betas=(0.9, 0.98), eps=1e-8):
        self.W, self.b1, self.b2, self.eps, self.wd, self.t = W, betas[0], betas[1], eps, wd, 0
        self.m = {k: torch.zeros_like(v) for k, v in W.items()}; self.v = {k: torch.zeros_like(v) for k, v in W.items()}
    @torch.no_grad()
    def step(self, lr):
        self.t += 1; bc1 = 1 - self.b1**self.t; bc2 = 1 - self.b2**self.t
        for k, p in self.W.items():
            g = p.grad; p.mul_(1 - lr * self.wd.view(-1, *[1] * (p.dim() - 1)))
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1); self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(self.m[k], (self.v[k].sqrt() / math.sqrt(bc2)).add_(self.eps), value=-lr / bc1); p.grad = None

def train_batch(task, inits, splits, steps, record=(), arch="attn", frac=0.3, wd=1.0, lr=1e-3, device="cpu"):
    """Train len(inits) independent models (model j: init seed inits[j], split seed splits[j]).
    Returns {step: {'embed': [M,nf], 'path': [M,nf]}} for steps in `record` (recorded before that step's update)."""
    sts = [init_state(s, task.p, arch) for s in inits]; M = len(inits)
    W = {k: torch.stack([s[k] for s in sts]).to(device).requires_grad_(True) for k in sts[0]}
    XA = task.inputs(); TR = torch.stack([task.split(s, frac)[0] for s in splits])
    X = XA[TR].to(device); Y = task.y[TR].to(device)
    opt = StackedAdamW(W, torch.full((M,), wd, device=device)); rec = {}
    for t in range(steps + 1):
        logits = forward(W, X, arch)
        if t in record:
            with torch.no_grad():
                rec[t] = dict(embed=freq_share(W["W_E"], task).cpu().numpy(), path=path_share(W, task, arch).cpu().numpy())
        if t == steps: break
        F.cross_entropy(logits.reshape(-1, task.p), Y.reshape(-1), reduction="none").reshape(M, -1).mean(1).sum().backward()
        opt.step(lr * min(t / 10, 1.0))
    return rec
