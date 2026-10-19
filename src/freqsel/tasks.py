"""Tasks on Z_p. Each task exposes: the set of input pairs, labels, and a 'group index' map so that Fourier
analysis is done in the basis where the task is addition (identity for add/sub, discrete log for mul)."""
import numpy as np, torch

def primitive_root(p):
    phi = p - 1; fac = [q for q in range(2, phi + 1) if phi % q == 0 and all(q % r for r in range(2, int(q**.5) + 1))]
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in fac): return g
    raise ValueError

class Task:
    def __init__(self, name="add", p=113):
        assert name in ("add", "sub", "mul")
        self.name, self.p = name, p
        if name == "mul":
            g = primitive_root(p); self.g = g
            self.log = np.full(p, -1); x = 1
            for e in range(p - 1): self.log[x] = e; x = x * g % p
            self.n = p - 1                                   # group order, Fourier size
            vals = torch.arange(1, p)
        else:
            self.log = np.arange(p); self.n = p; vals = torch.arange(p)
        a = vals.repeat_interleave(len(vals)); b = vals.repeat(len(vals))
        self.a, self.b = a, b
        self.y = {"add": (a + b) % p, "sub": (a - b) % p, "mul": (a * b) % p}[name]
        self.tokens = vals                                   # tokens whose embeddings are analysed
        self.nf = (self.n - 1) // 2 if self.n % 2 else self.n // 2 - 1   # frequencies 1..nf (complex pairs only)

    def split(self, seed, frac=0.3):
        """Same generator as the original experiments (torch randperm with Generator(seed))."""
        g = torch.Generator().manual_seed(seed); N = len(self.a)
        perm = torch.randperm(N, generator=g); n = int(frac * N)
        return perm[:n], perm[n:]

    def inputs(self):
        return torch.stack([self.a, self.b, torch.full_like(self.a, self.p)], 1)
