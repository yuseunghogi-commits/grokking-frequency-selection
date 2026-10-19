import torch, torch.nn as nn, torch.nn.functional as F

P = 113

class OneLayer(nn.Module):
    """1-layer attention + MLP transformer, no LayerNorm (so the circuit is readable)."""
    def __init__(self, p=P, d=128, h=4, d_mlp=512, n_ctx=3):
        super().__init__()
        self.p, self.d, self.h, self.dh = p, d, h, d // h
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / d**0.5)   # token p = '='
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d) / d**0.5)
        self.W_Q = nn.Parameter(torch.randn(h, d, self.dh) / d**0.5)
        self.W_K = nn.Parameter(torch.randn(h, d, self.dh) / d**0.5)
        self.W_V = nn.Parameter(torch.randn(h, d, self.dh) / d**0.5)
        self.W_O = nn.Parameter(torch.randn(h, self.dh, d) / d**0.5)
        self.W_in = nn.Parameter(torch.randn(d, d_mlp) / d**0.5)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d) / d_mlp**0.5)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = nn.Parameter(torch.randn(d, p) / d**0.5)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)).bool())

    def forward(self, x, cache=None, hook=None):
        # Single layer: only the '=' (last) position feeds the output, so compute
        # the query, MLP and unembed there only. Mathematically identical.
        r = self.W_E[x] + self.W_pos                      # [B,3,d]
        q = torch.einsum("bd,hde->bhe", r[:, -1], self.W_Q)
        k = torch.einsum("bpd,hde->bhpe", r, self.W_K)
        v = torch.einsum("bpd,hde->bhpe", r, self.W_V)
        s = torch.einsum("bhe,bhpe->bhp", q, k) / self.dh**0.5
        a = s.softmax(-1)                                  # [B,h,3] (last row is unmasked)
        z = torch.einsum("bhp,bhpe->bhe", a, v)
        r = r[:, -1] + torch.einsum("bhe,hed->bd", z, self.W_O)
        pre = r @ self.W_in + self.b_in
        neur = F.relu(pre)                                 # [B,d_mlp]
        if hook is not None:
            neur = hook(neur)
        r = r + neur @ self.W_out + self.b_out
        if cache is not None:
            cache.update(attn=a, neur=neur, resid_final=r)
        return r @ self.W_U

def dataset(p=P, frac=0.3, seed=0):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.full_like(a, p)], 1)
    y = (a + b) % p
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n = int(frac * p * p)
    return x, y, perm[:n], perm[n:]
