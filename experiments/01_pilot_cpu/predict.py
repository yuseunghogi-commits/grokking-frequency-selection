"""Can we predict the winning frequencies from the initialization alone?

For each frequency k, define the network's output component along the *correct-answer pattern* at k:
    R_k(theta) = sum_{train (a,b)} sum_c  f_theta(a,b)[c] * cos(w_k (a+b-c))      (f = logits centered over c)
The one-hot target has equal weight on every k, so the target itself favours no frequency.
Predictors computed at init only:
  P1  align   : R_k at init (does the random net already point the right way at k?)
  P2  ntk     : ||dR_k/dtheta||^2  (gradient-descent growth rate of R_k, linearized / NTK view)
  P3  adam1   : change in R_k produced by Adam's first step, dtheta = -sign(dLoss/dtheta)
usage: python3 predict.py known      -> 4 seeds with known final winners
       python3 predict.py new A B    -> seeds A..B-1: winners proxied by top-k at step 500 (train 500 steps)
"""
import torch, torch.nn.functional as F, numpy as np, json, sys, time
from model import OneLayer, dataset, P
from early import init_state, train, KEYS
from analyze import B
torch.set_grad_enabled(True)
torch.set_grad_enabled(True)
p = P
x, y, tr, te = dataset()
X, S = x[tr], ((x[tr, 0] + x[tr, 1]) % p).float()
c = torch.arange(p).float()
W = [2 * np.pi * k / p for k in range(1, p // 2 + 1)]
PH = [torch.cos(w * (S[:, None] - c[None])) for w in W]          # [N,p] per k

def R_all(m):
    f = m(X); f = f - f.mean(-1, keepdim=True)
    return torch.stack([(f * ph).sum() for ph in PH])             # [56]

def predictors(state):
    m = OneLayer(); m.load_state_dict(state)
    params = [q for q in m.parameters()]
    R = R_all(m)
    align = R.detach()
    ntk = []
    for k in range(len(W)):
        g = torch.autograd.grad(R[k], params, retain_graph=True)
        ntk.append(sum((gi ** 2).sum() for gi in g).item())
    # Adam's first step ~ -lr * sign(grad of loss); first-order change in R_k
    m.zero_grad(); F.cross_entropy(m(X), y[tr]).backward()
    step = [-torch.sign(q.grad) for q in params]
    adam1 = []
    R2 = R_all(m)
    for k in range(len(W)):
        g = torch.autograd.grad(R2[k], params, retain_graph=True)
        adam1.append(sum((gi * si).sum() for gi, si in zip(g, step)).item())
    with torch.no_grad():   # P4 (pre-registered after origin.py): frequency content of the embedding as read through attention
        v = torch.einsum("nd,hde,hef->nf", m.W_E[:p], m.W_V, m.W_O)
        q = (B @ v).norm(dim=1) ** 2
        attn_path = [(q[2*k-1] + q[2*k]).item() for k in range(1, p // 2 + 1)]
    return dict(attn_path=attn_path, align=align.tolist(), ntk=ntk, adam1=adam1)

def hits(score, keys):
    order = sorted(range(56), key=lambda k: -score[k])[:len(keys)]
    return sum((k - 1) in order for k in keys), [sorted(range(56), key=lambda k: -score[k]).index(k - 1) + 1 for k in keys]

def proxy_keys(state, n=3):
    spec = train(state, 500, 500)[-1]["spec"]
    return [k + 1 for k in sorted(range(56), key=lambda k: -spec[k])[:n]]

if __name__ == "__main__":
    mode = sys.argv[1]
    seeds = range(4) if mode == "known" else range(int(sys.argv[2]), int(sys.argv[3]))
    out, tot, n_tot = [], {"attn_path": 0, "align": 0, "ntk": 0, "adam1": 0}, 0
    for s in seeds:
        t0 = time.time()
        st = init_state(s)
        keys = KEYS[s] if mode == "known" else proxy_keys(st)
        pr = predictors(st)
        row = dict(seed=s, keys=keys)
        for name, sc in pr.items():
            h, ranks = hits(sc, keys); row[name] = dict(hits=h, ranks=ranks); tot[name] += h
        n_tot += len(keys); out.append(row)
        print(json.dumps(row), f"{time.time()-t0:.0f}s", flush=True)
    chance = sum(len(r["keys"]) ** 2 / 56 for r in out)
    print("TOTAL", {k: f"{v}/{n_tot}" for k, v in tot.items()}, f"chance {chance:.2f}", flush=True)
    json.dump(dict(rows=out, total=tot, n=n_tot, chance=chance), open(f"seeds/predict_{mode}.json", "w"), indent=1)
