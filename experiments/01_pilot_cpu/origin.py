"""Where does the frequency choice start? Track per-frequency spectra of several quantities
every 5 steps for the first 150 steps and see which one locks onto the eventual winners first."""
import torch, torch.nn.functional as F, numpy as np, json
from model import OneLayer, dataset, P
from early import init_state, KEYS
from analyze import B
torch.set_grad_enabled(True)
p = P
x, y, tr, te = dataset()
X = x[tr]; S = ((X[:, 0] + X[:, 1]) % p).float(); c = torch.arange(p).float()
W = [2 * np.pi * k / p for k in range(1, p // 2 + 1)]
COS = torch.stack([torch.cos(w * (S[:, None] - c[None])) for w in W])   # [56,N,p]
SIN = torch.stack([torch.sin(w * (S[:, None] - c[None])) for w in W])

def per_freq(v):            # v [p, d] indexed by number -> [56] norm^2 per frequency
    q = (B @ v).norm(dim=1) ** 2
    return torch.stack([q[2*k-1] + q[2*k] for k in range(1, p // 2 + 1)])

def quantities(m):
    with torch.no_grad():
        f = m(X); f = f - f.mean(-1, keepdim=True)
        Rc = (COS * f).sum((1, 2)); Rs = (SIN * f).sum((1, 2))
        # input side: embedding, and embedding as seen by the MLP through the direct path
        return dict(embed=per_freq(m.W_E[:p]),
                    embed_to_mlp=per_freq(m.W_E[:p] @ m.W_in),
                    embed_via_attn=per_freq(torch.einsum("nd,hde,hef->nf", m.W_E[:p], m.W_V, m.W_O)),
                    unembed=per_freq(m.W_U.T),                      # frequency over the output c
                    func_correct=Rc,                                # signed: + means pointing at the right answer
                    func_mag=(Rc ** 2 + Rs ** 2).sqrt())

def hits(v, keys):
    order = sorted(range(56), key=lambda k: -float(v[k]))
    n = min(len(keys), 3)
    return sum(order.index(k - 1) < len(keys) for k in keys) / len(keys)

out = {}
for s in range(4):
    m = OneLayer(); m.load_state_dict(init_state(s))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(t / 10, 1.0))
    q0 = quantities(m); rec = []
    for step in range(151):
        if step % 5 == 0:
            q = quantities(m)
            row = dict(step=step)
            for name, v in q.items():
                row[name] = hits(v, KEYS[s])
                if name != "func_correct":   # growth since init (ratio), for magnitude-type quantities
                    row[name + "_growth"] = hits(v / (q0[name] + 1e-12), KEYS[s])
            rec.append(row)
        loss = F.cross_entropy(m(X), y[tr]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    out[s] = rec
    print("seed", s, "done", flush=True)
json.dump(out, open("seeds/origin.json", "w"))

names = [k for k in out[0][0] if k != "step"]
print("mean hit-rate over 4 seeds (fraction of winners in top-k):")
print("step  " + "  ".join(f"{n[:14]:>14s}" for n in names))
for i in range(0, 31, 2):
    print(f"{out[0][i]['step']:4d}  " + "  ".join(f"{np.mean([out[s][i][n] for s in out]):14.2f}" for n in names))
