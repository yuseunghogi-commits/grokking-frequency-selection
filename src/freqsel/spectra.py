import math, torch

def fourier_basis(n, device="cpu"):
    """Orthonormal real Fourier basis of size n: rows [const, cos1, sin1, cos2, sin2, ...]."""
    x = torch.arange(n, dtype=torch.float64); rows = [torch.ones(n, dtype=torch.float64)]
    for k in range(1, (n - 1) // 2 + 1):
        rows += [torch.cos(2 * math.pi * k * x / n), torch.sin(2 * math.pi * k * x / n)]
    if n % 2 == 0: rows.append(torch.cos(math.pi * x))
    B = torch.stack(rows); return (B / B.norm(dim=1, keepdim=True)).float().to(device)

def freq_share(V, task):
    """V: [M, p+1, d] embedding-like matrix indexed by token. Returns [M, nf] share of Fourier power per frequency,
    computed over the task's tokens re-indexed by the group map (identity for add/sub, discrete log for mul)."""
    tok = task.tokens.to(V.device); idx = torch.as_tensor(task.log[tok.cpu().numpy()], device=V.device)
    Vg = torch.zeros(V.shape[0], task.n, V.shape[2], device=V.device); Vg[:, idx] = V[:, tok]
    B = fourier_basis(task.n, V.device)
    q = (torch.einsum("ia,mad->mid", B, Vg) ** 2).sum(-1)
    fn = q[:, 1:2 * task.nf + 1:2] + q[:, 2:2 * task.nf + 2:2]
    return fn / fn.sum(-1, keepdim=True)

def path_share(W, task, arch="attn"):
    if arch == "attn": V = torch.einsum("mnd,mhde,mhef->mnf", W["W_E"], W["W_V"], W["W_O"])
    else: V = W["W_E"] @ W["W_in"][:, :W["W_E"].shape[-1]]
    return freq_share(V, task)
