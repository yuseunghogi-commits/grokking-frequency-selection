"""Early-phase experiments: when are frequencies chosen, and which init weights decide them.
usage: python3 early.py timing | swap
"""
import torch, torch.nn.functional as F, json, sys, time
from model import OneLayer, dataset, P
from analyze import B
torch.set_grad_enabled(True)
p = P
KEYS = {int(s): r["final_keys"] for s, r in json.load(open("seeds/summary.json")).items()}
GROUPS = {"embed": ["W_E", "W_pos"], "attn": ["W_Q", "W_K", "W_V", "W_O"],
          "mlp": ["W_in", "b_in", "W_out", "b_out"], "unembed": ["W_U"]}
x, y, tr, te = dataset()

def init_state(seed):
    torch.manual_seed(seed)
    return {k: v.clone() for k, v in OneLayer().state_dict().items()}

def spectrum(m):
    with torch.no_grad():
        c = (B @ m.W_E[:p]).norm(dim=1) ** 2
        fn = torch.stack([c[2*k-1] + c[2*k] for k in range(1, p//2 + 1)])
        return (fn / fn.sum()).tolist()

def train(state, steps, every, wd=1.0):
    m = OneLayer(); m.load_state_dict(state)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1.0))
    rec = []
    for step in range(steps + 1):
        logits = m(x[tr]); loss = F.cross_entropy(logits, y[tr])
        if step % every == 0:
            rec.append(dict(step=step, spec=spectrum(m),
                            train_acc=(logits.argmax(-1) == y[tr]).float().mean().item()))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return rec

def share(spec, keys):
    return sum(spec[k - 1] for k in keys)

def topk_hits(spec, keys):
    order = sorted(range(56), key=lambda k: -spec[k])[:len(keys)]
    return sum((k - 1) in order for k in keys)

if len(sys.argv) > 1 and sys.argv[1] == "timing":
    out = {}
    for s in range(4):
        t0 = time.time()
        rec = train(init_state(s), 500, 10)
        out[s] = [dict(step=r["step"], train_acc=r["train_acc"], key_share=share(r["spec"], KEYS[s]),
                       hits=topk_hits(r["spec"], KEYS[s]), n=len(KEYS[s])) for r in rec]
        first = next((r["step"] for r in out[s] if r["hits"] >= min(3, r["n"])), None)
        print(f"seed {s} keys {KEYS[s]}: top-k holds >=3 keys from step {first}; "
              f"train acc 100% at step {next((r['step'] for r in out[s] if r['train_acc'] == 1.0), None)}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open("seeds/timing.json", "w"))

if len(sys.argv) > 1 and sys.argv[1] == "swap":
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    out = []
    for a, b in pairs:
        A, Bs = init_state(a), init_state(b)
        for g, names in GROUPS.items():
            st = {k: (Bs[k] if k in names else A[k]).clone() for k in A}
            spec = train(st, 500, 500)[-1]["spec"]
            r = dict(base=a, donor=b, group=g,
                     share_base_keys=share(spec, KEYS[a]), share_donor_keys=share(spec, KEYS[b]),
                     hits_base=topk_hits(spec, KEYS[a]), hits_donor=topk_hits(spec, KEYS[b]),
                     n_base=len(KEYS[a]), n_donor=len(KEYS[b]))
            out.append(r)
            print(json.dumps(r), flush=True)
    json.dump(out, open("seeds/swap.json", "w"), indent=1)

if len(sys.argv) > 1 and sys.argv[1] == "noise":
    out = []
    for s in range(4):
        A = init_state(s)
        for eps in (0.01, 0.05, 0.2):
            g = torch.Generator().manual_seed(1000 + s)
            st = {k: v + eps * v.std() * torch.randn(v.shape, generator=g) if v.is_floating_point() and v.numel() > 1 and v.std() > 0 else v.clone()
                  for k, v in A.items()}
            spec = train(st, 500, 500)[-1]["spec"]
            r = dict(seed=s, eps=eps, hits=topk_hits(spec, KEYS[s]), n=len(KEYS[s]),
                     share=share(spec, KEYS[s]))
            out.append(r); print(json.dumps(r), flush=True)
    json.dump(out, open("seeds/noise.json", "w"), indent=1)
