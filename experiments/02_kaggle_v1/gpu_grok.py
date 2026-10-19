# =====================================================================================
#  Reading what a network will learn before it shows  —  GPU program (Kaggle / Colab)
#  Self-contained. Stages save results to OUT and are skipped if already done (resume).
# =====================================================================================
import os, sys, json, time, math, hashlib, itertools
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

SMOKE = os.environ.get("GROK_SMOKE", "0") == "1"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = 113
NF = P // 2                                   # 56 frequencies, k = 1..56 (1-indexed in reports)
T_START = time.time()
TIME_LIMIT_H = float(os.environ.get("GROK_TIME_LIMIT_H", "11.0"))

def _out_dir():
    if os.environ.get("GROK_OUT"): return Path(os.environ["GROK_OUT"])
    if Path("/content/drive/MyDrive").exists(): return Path("/content/drive/MyDrive/grok_out")
    if Path("/kaggle/working").exists(): return Path("/kaggle/working/grok_out")
    return Path("grok_out")
OUT = _out_dir(); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "log.txt"

def log(*a):
    s = f"[{(time.time()-T_START)/60:7.1f} min] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(s + "\n")

def time_left_h(): return TIME_LIMIT_H - (time.time() - T_START) / 3600

# -------------------------------------------------------------------------------------
#  PRE-REGISTRATION  (fixed before any of these runs exist; hashed into every output)
# -------------------------------------------------------------------------------------
PREREG = r"""
PRE-REGISTRATION — written 2026-10-18 (KST) before any run of this notebook.
Winner set W(i,s): top-3 embedding Fourier frequencies at step 500 (base model, add mod 113, frac 0.3, wd 1.0).
Dev IDs: inits 0-31, splits 0-31. Test IDs: inits 1000-1031, splits 1000-1031. Independent-vote inits 5000-5063.
Votes for cell (i,s): init_vote = mean over other splits of z(log embed share @500) for init i;
  data_vote = mean over other inits of the same for split s; z() standardizes over the 56 frequencies within a run.
  data_vote_ind = mean over the 64 independent inits (5000-5063) on split s, step 200.
Hits = number of winners inside the predicted top-3, summed over cells. Chance-level p-values use a cluster-aware
  permutation null (2000 perms): frequency labels of the init-side score are shuffled with one permutation per init and
  those of the data-side score with one permutation per split (cells sharing an init or split are not independent).
  The hypergeometric expectation (cells x 9/56) is reported for reference only.

C1 (separability, test factorial): hits(z(init_vote)+z(data_vote)) exceeds hits(init_vote) AND hits(data_vote);
    PASS iff both paired-bootstrap 95% CIs (resampling inits and splits, 5000 reps) of the differences are > 0.
C2 (data statistic): on dev, choose among closed-form statistics D1,D2,D3,D4,D5min,D5cond,D5r (either sign) the one with the
    highest mean within-split Spearman correlation with the dev data_vote_ind. On test splits: PASS iff that
    (sign-fixed) mean within-split Spearman > 0 with within-split permutation p < 0.01 (10000 perms).
C3 (causal data test): pick the 2 dev splits with the largest max_k of dev data_vote (factorial, step 500); k* = argmax.
    Datamodel (1536 runs/split, 10% pairs swapped for test pairs, fresh inits, outcome z(k*) at step 200, ridge with CV)
    ranks training pairs by how much removing them lowers k*. Interventions on 64 fresh inits (9000-9063, paired):
    remove top-10% pairs (A), 5 random 10% (B), bottom-10% (C). PASS (per split) iff mean_B z(k*) - mean_A z(k*) > 0 with
    one-sided paired sign-flip permutation p < 0.01 after Holm over the 2 splits, and mean_C > mean_A.
C4 (combined predictor, test factorial): score = z(attn_path_at_init) + lam * z(data_vote_ind), lam chosen on dev from
    {0,0.25,0.5,1,2,4}. PASS iff hits(combined) - hits(attn_path) has paired-bootstrap 95% CI > 0 AND
    hits(combined) is above chance with permutation p < 0.001.  C4b: same with the C2 statistic in place of data_vote_ind (reported).
R1 (replication of pilot): attention-path-at-init alone on test cells: PASS iff above chance with permutation p < 0.001.
C5 (generality): for each config that groks (test acc>=0.9) in >= 12/16 full runs: PASS iff median commitment step
    (first recorded step after which the top-3 equals the step-500 top-3) <= 5% of median grok step.
C6 (forecasting grok time; grid wd{0.1,0.3,1,3} x frac{0.25,0.3,0.4}): on dev runs pick the best internal and the best
    behavioural forecast (metric x t0 in {500,1000,2000} x sign) by Harrell's C. On test runs: PASS iff internal C >= 0.75
    AND internal C - behavioural C >= 0.05 AND bootstrap 95% CI of that difference > 0.
Holm adjustment is reported across C1-C4,R1. Failures are reported as failures.
"""
PREREG_HASH = hashlib.sha256(PREREG.encode()).hexdigest()

# -------------------------------------------------------------------------------------
#  Data, models (pilot-identical initialisation), Fourier basis
# -------------------------------------------------------------------------------------
def grid_pairs(p=P):
    a = torch.arange(p).repeat_interleave(p); b = torch.arange(p).repeat(p)
    return a, b

def labels(task, a, b, p=P):
    return (a + b) % p if task == "add" else (a - b) % p

def split_indices(split_seed, frac=0.3, p=P):
    """Identical to pilot model.dataset(): randperm with torch.Generator(seed)."""
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(p * p, generator=g)
    n = int(frac * p * p)
    return perm[:n], perm[n:]

class OneLayer(nn.Module):           # copied from pilot (parameter creation order matters for init)
    def __init__(self, p=P, d=128, h=4, d_mlp=512, n_ctx=3):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / d**0.5)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d) / d**0.5)
        self.W_Q = nn.Parameter(torch.randn(h, d, d // h) / d**0.5)
        self.W_K = nn.Parameter(torch.randn(h, d, d // h) / d**0.5)
        self.W_V = nn.Parameter(torch.randn(h, d, d // h) / d**0.5)
        self.W_O = nn.Parameter(torch.randn(h, d // h, d) / d**0.5)
        self.W_in = nn.Parameter(torch.randn(d, d_mlp) / d**0.5)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d) / d_mlp**0.5)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = nn.Parameter(torch.randn(d, p) / d**0.5)

class MLPOnly(nn.Module):            # no attention: concat(E[a], E[b]) -> ReLU MLP -> unembed
    def __init__(self, p=P, d=128, d_mlp=512):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / d**0.5)
        self.W_in = nn.Parameter(torch.randn(2 * d, d_mlp) / (2 * d)**0.5)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d) / d_mlp**0.5)
        self.b_out = nn.Parameter(torch.zeros(d))
        self.W_U = nn.Parameter(torch.randn(d, p) / d**0.5)

def init_state(seed, arch="attn"):
    torch.manual_seed(seed)
    m = OneLayer() if arch == "attn" else MLPOnly()
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

def fourier_basis(p=P):
    x = torch.arange(p).double(); rows = [torch.ones(p, dtype=torch.float64)]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * math.pi * k * x / p), torch.sin(2 * math.pi * k * x / p)]
    Bm = torch.stack(rows); Bm = Bm / Bm.norm(dim=1, keepdim=True)
    return Bm.float()
B = fourier_basis().to(DEV)
_FREQ_MASK = None
def freq_mask_2d():
    """[56, p, p] mask selecting rows/cols {0, 2k-1, 2k} of the 2D Fourier coefficient matrix (excluding [0,0])."""
    global _FREQ_MASK
    if _FREQ_MASK is None:
        M = torch.zeros(NF, P, P, device=DEV)
        for k in range(1, NF + 1):
            idx = torch.tensor([0, 2 * k - 1, 2 * k], device=DEV)
            M[k - 1][idx[:, None], idx[None, :]] = 1
            M[k - 1][0, 0] = 0
        _FREQ_MASK = M
    return _FREQ_MASK

def per_freq_share(v):               # v [M, p, d] -> [M, 56] fraction of Fourier power per frequency
    q = (torch.einsum("ia,mad->mid", B, v) ** 2).sum(-1)
    fn = q[:, 1::2] + q[:, 2::2]
    return fn / fn.sum(-1, keepdim=True)

def path_spectrum(W, arch):
    if arch == "attn":
        v = torch.einsum("mnd,mhde,mhef->mnf", W["W_E"][:, :P], W["W_V"], W["W_O"])
    else:
        v = W["W_E"][:, :P] @ W["W_in"][:, :W["W_E"].shape[-1]]
    return per_freq_share(v)

# -------------------------------------------------------------------------------------
#  Batched forward (M independent models, stacked weights)
# -------------------------------------------------------------------------------------
def forward(W, X, arch="attn", neur_override=None, return_cache=False):
    """X [M,N,3] long on DEV. Returns logits [M,N,P]."""
    M = X.shape[0]
    mi = torch.arange(M, device=X.device)[:, None, None]
    if arch == "attn":
        r = W["W_E"][mi, X] + W["W_pos"][:, None]                       # [M,N,3,d]
        dh = W["W_Q"].shape[-1]
        q = torch.einsum("mnd,mhde->mnhe", r[:, :, -1], W["W_Q"])
        k = torch.einsum("mnpd,mhde->mnhpe", r, W["W_K"])
        v = torch.einsum("mnpd,mhde->mnhpe", r, W["W_V"])
        a = (torch.einsum("mnhe,mnhpe->mnhp", q, k) / dh ** 0.5).softmax(-1)
        z = torch.einsum("mnhp,mnhpe->mnhe", a, v)
        mid = r[:, :, -1] + torch.einsum("mnhe,mhed->mnd", z, W["W_O"])
        pre = torch.einsum("mnd,mdf->mnf", mid, W["W_in"]) + W["b_in"][:, None]
    else:
        e = W["W_E"][mi, X[:, :, :2]]                                     # [M,N,2,d]
        mid = None
        pre = torch.einsum("mnd,mdf->mnf", e.flatten(2), W["W_in"]) + W["b_in"][:, None]
    neur = F.relu(pre) if neur_override is None else neur_override
    out = torch.einsum("mnf,mfd->mnd", neur, W["W_out"]) + W["b_out"][:, None]
    if mid is not None: out = out + mid
    logits = torch.einsum("mnd,mdc->mnc", out, W["W_U"])
    if return_cache: return logits, dict(neur=neur, mid=mid)
    return logits

class StackedAdamW:
    """torch.optim.AdamW semantics (decoupled decay, bias correction, eps 1e-8) with a per-model weight decay vector."""
    def __init__(self, W, lr, wd, betas=(0.9, 0.98), eps=1e-8):
        self.W, self.lr, self.b1, self.b2, self.eps = W, lr, betas[0], betas[1], eps
        self.wd = wd                                                      # tensor [M]
        self.m = {k: torch.zeros_like(v) for k, v in W.items()}
        self.v = {k: torch.zeros_like(v) for k, v in W.items()}
        self.t = 0
    @torch.no_grad()
    def step(self, lr_now):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t; bc2 = 1 - self.b2 ** self.t
        for k, p in self.W.items():
            g = p.grad
            wd = self.wd.view(-1, *([1] * (p.dim() - 1)))
            p.mul_(1 - lr_now * wd)
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (self.v[k].sqrt() / math.sqrt(bc2)).add_(self.eps)
            p.addcdiv_(self.m[k], denom, value=-lr_now / bc1)
            p.grad = None

def lr_at(step, base=1e-3):          # pilot: LambdaLR(min(s/10,1)); the update at iteration t uses factor min(t/10,1)
    return base * min(step / 10, 1.0)

def zscore_log(share):               # numpy [..., 56] -> z of log share over the 56 frequencies
    l = np.log(np.maximum(np.asarray(share, dtype=np.float64), 1e-30))
    return (l - l.mean(-1, keepdims=True)) / l.std(-1, keepdims=True)

# -------------------------------------------------------------------------------------
#  Circuit analysis for one model (index m in a stacked dict)
# -------------------------------------------------------------------------------------
_GRID = None
def full_grid():
    global _GRID
    if _GRID is None:
        a, b = grid_pairs()
        _GRID = torch.stack([a, b, torch.full_like(a, P)], 1).to(DEV)
    return _GRID

def slice_model(W, m): return {k: v[m:m + 1] for k, v in W.items()}

@torch.no_grad()
def circuit_metrics(Wm, arch, task, tr_idx, te_idx, keys=None, min_share=0.03):
    """Wm: single-model stacked dict. Returns dict with keys, neuron key share, clean/restricted/excluded test loss."""
    X = full_grid()[None]
    a, b = grid_pairs(); y = labels(task, a, b).to(DEV)
    logits, cache = forward(Wm, X, arch, return_cache=True)
    G = cache["neur"][0].reshape(P, P, -1)                               # [p,p,N]
    C = torch.einsum("ia,jb,abn->ijn", B, B, G)
    C0 = C.clone(); C0[0, 0] = 0
    tot = (C0 ** 2).sum((0, 1)) + 1e-12
    per = torch.einsum("kij,ijn->kn", freq_mask_2d(), C0 ** 2)          # [56,N]
    alive = G.reshape(-1, G.shape[-1]).std(0) > 1e-3
    dom = per.argmax(0) + 1
    if keys is None:
        ks = []
        n_alive = int(alive.sum())
        for k in range(1, NF + 1):
            if n_alive and float(((dom == k) & alive).sum()) / n_alive >= min_share: ks.append(k)
        keys = ks
    Mk = torch.zeros(P, P, device=DEV)
    for k in keys: Mk += freq_mask_2d()[k - 1]
    Mk = Mk.clamp(max=1)[..., None]
    key_share = float(((C0 * Mk) ** 2).sum() / (C0 ** 2).sum()) if len(keys) else 0.0
    def run_with(Cmod):
        Gm = torch.einsum("ia,jb,ijn->abn", B, B, Cmod).reshape(1, P * P, -1)
        return forward(Wm, X, arch, neur_override=Gm)[0]
    def loss_acc(lg, idx):
        return float(F.cross_entropy(lg[idx], y[idx])), float((lg[idx].argmax(-1) == y[idx]).float().mean())
    clean = loss_acc(logits[0], te_idx)
    out = dict(keys=keys, neuron_key_share=key_share, test_loss=clean[0], test_acc=clean[1],
               train_loss=loss_acc(logits[0], tr_idx)[0])
    if len(keys):
        Cr = C * Mk; Cr[0, 0] = C[0, 0]
        Cx = C * (1 - Mk); Cx[0, 0] = C[0, 0]
        out["restricted_loss"], out["restricted_acc"] = loss_acc(run_with(Cr), te_idx)
        out["excluded_loss"], out["excluded_acc"] = loss_acc(run_with(Cx), te_idx)
    return out

# -------------------------------------------------------------------------------------
#  Generic batched trainer
# -------------------------------------------------------------------------------------
def train_batch(specs, steps, record=(), arch="attn", task="add", frac=0.3, eval_every=0,
                analyze_at=(), stop_after_grok=None, custom_train_idx=None):
    """specs: list of dict(init=, split=, wd=). All share arch/task/frac.
    custom_train_idx: optional list of LongTensor (one per spec) overriding the split's training indices (same length).
    Returns dict: rec{step:{embed,path,train_acc}}, curves (if eval_every), analyses{step:[per-model dict]}, grok_step."""
    M = len(specs)
    sts = [init_state(s["init"], arch) for s in specs]
    W = {k: torch.stack([st[k] for st in sts]).to(DEV).requires_grad_(True) for k in sts[0]}
    a, b = grid_pairs(); y_all = labels(task, a, b)
    G = full_grid()
    trs, tes = [], []
    for j, s in enumerate(specs):
        tr, te = split_indices(s["split"], frac)
        if custom_train_idx is not None:
            tr = custom_train_idx[j]
            mask = torch.ones(P * P, dtype=torch.bool); mask[tr] = False
            te = torch.nonzero(mask).flatten()
        trs.append(tr); tes.append(te)
    TR = torch.stack(trs).to(DEV)                                         # [M,N]
    X = G[TR]                                                             # [M,N,3]
    Y = y_all.to(DEV)[TR]
    wd = torch.tensor([s.get("wd", 1.0) for s in specs], device=DEV, dtype=torch.float32)
    opt = StackedAdamW(W, 1e-3, wd)
    rec, curves, analyses = {}, [], {}
    grok = [None] * M
    record = set(record); analyze_at = set(analyze_at)
    for step in range(steps + 1):
        logits = forward(W, X, arch)
        lossv = F.cross_entropy(logits.reshape(-1, P), Y.reshape(-1), reduction="none").reshape(M, -1).mean(1)
        if step in record:
            with torch.no_grad():
                rec[step] = dict(embed=per_freq_share(W["W_E"][:, :P]).cpu().numpy(),
                                 path=path_spectrum(W, arch).cpu().numpy(),
                                 train_acc=(logits.argmax(-1) == Y).float().mean(1).cpu().numpy(),
                                 train_loss=lossv.detach().cpu().numpy())
        if eval_every and step % eval_every == 0:
            with torch.no_grad():
                wn = sum((v ** 2).flatten(1).sum(1) for v in W.values())
                row = dict(step=step, train_loss=lossv.detach().cpu().numpy().tolist(),
                           wnorm=wn.cpu().numpy().tolist(), test_loss=[], test_acc=[])
                for m0 in range(0, M, 16):
                    sl = slice(m0, min(M, m0 + 16))
                    lg = forward({k: v[sl] for k, v in W.items()}, G[None].expand(sl.stop - sl.start, -1, -1), arch)
                    for j, m in enumerate(range(sl.start, sl.stop)):
                        te = tes[m].to(DEV)
                        row["test_loss"].append(float(F.cross_entropy(lg[j][te], y_all.to(DEV)[te])))
                        row["test_acc"].append(float((lg[j][te].argmax(-1) == y_all.to(DEV)[te]).float().mean()))
                curves.append(row)
                for m in range(M):
                    if grok[m] is None and row["test_acc"][m] >= 0.9: grok[m] = step
        if step in analyze_at:
            with torch.no_grad():
                analyses[step] = [circuit_metrics(slice_model(W, m), arch, task, trs[m].to(DEV), tes[m].to(DEV))
                                  for m in range(M)]
        if stop_after_grok is not None and all(g is not None and step >= g + stop_after_grok for g in grok) \
                and step >= max(analyze_at, default=0):
            break
        if step == steps: break
        lossv.sum().backward()
        opt.step(lr_at(step))
    final = None
    if "final" in analyze_at or eval_every:
        with torch.no_grad():
            final = [circuit_metrics(slice_model(W, m), arch, task, trs[m].to(DEV), tes[m].to(DEV)) for m in range(M)]
    return dict(rec=rec, curves=curves, analyses=analyses, grok=grok, final=final, last_step=step)

def batch_size_for(arch="attn"):
    if SMOKE: return 4
    if DEV.type == "cpu": return 4
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    return 64 if mem >= 14 else 32

# -------------------------------------------------------------------------------------
#  Stage helpers: chunked, resumable
# -------------------------------------------------------------------------------------
def run_chunked(stage, specs, steps, record, **kw):
    """Runs specs in chunks; saves each chunk to OUT/stage/chunk_XXXX.npz with spectra; returns concatenated arrays."""
    d = OUT / stage; d.mkdir(exist_ok=True)
    bs = kw.pop("bs", None) or batch_size_for(kw.get("arch", "attn"))
    rec_steps = sorted(record)
    chunks = [specs[i:i + bs] for i in range(0, len(specs), bs)]
    t0 = time.time(); done_new = 0
    for ci, ch in enumerate(chunks):
        f = d / f"chunk_{ci:04d}.npz"
        if f.exists(): continue
        if time_left_h() < 0.3: log(f"{stage}: time budget exhausted at chunk {ci}/{len(chunks)}"); break
        r = train_batch(ch, steps, record=rec_steps, **kw)
        np.savez(str(f) + ".tmp.npz",
                 init=np.array([s["init"] for s in ch]), split=np.array([s["split"] for s in ch]),
                 steps=np.array(rec_steps),
                 embed=np.stack([r["rec"][t]["embed"] for t in rec_steps], 1),
                 path=np.stack([r["rec"][t]["path"] for t in rec_steps], 1),
                 train_acc=np.stack([r["rec"][t]["train_acc"] for t in rec_steps], 1))
        os.replace(str(f) + ".tmp.npz", f)
        done_new += 1
        el = time.time() - t0
        log(f"{stage}: chunk {ci+1}/{len(chunks)}  ({el/done_new:.0f} s/chunk, ETA {el/done_new*(len(chunks)-ci-1)/60:.0f} min)")
    return load_chunks(stage)

def load_chunks(stage):
    fs = sorted((OUT / stage).glob("chunk_*.npz"))
    fs = [f for f in fs if not f.name.endswith(".tmp.npz")]
    if not fs: return None
    ds = [np.load(f) for f in fs]
    return dict(init=np.concatenate([d["init"] for d in ds]), split=np.concatenate([d["split"] for d in ds]),
                steps=ds[0]["steps"], embed=np.concatenate([d["embed"] for d in ds]),
                path=np.concatenate([d["path"] for d in ds]), train_acc=np.concatenate([d["train_acc"] for d in ds]))

def stage_done(name): return (OUT / f"{name}.done").exists()
def mark_done(name, info=None):
    (OUT / f"{name}.done").write_text(json.dumps(info or {}, indent=1, default=float))

# -------------------------------------------------------------------------------------
#  Statistics
# -------------------------------------------------------------------------------------
def hyper_null(n_cells, k=3):
    h = np.array([math.comb(k, j) * math.comb(NF - k, k - j) / math.comb(NF, k) for j in range(k + 1)])
    d = np.array([1.0])
    for _ in range(n_cells): d = np.convolve(d, h)
    return d
def p_above_chance(hits, n_cells): return float(hyper_null(n_cells)[hits:].sum())

def topk_hits_cells(score, win_top3):
    """score [I,S,56], win_top3 [I,S,3] -> hits matrix [I,S]."""
    top = np.argsort(-score, -1)[..., :3]
    return (top[..., :, None] == win_top3[..., None, :]).any(-1).sum(-1)

def perm_p_hits(init_score, data_score, lam, win_top3, reps=2000, seed=0):
    """init_score [I,56] (per init), data_score [S,56] or None, lam: weight. Cluster-aware permutation null."""
    rng = np.random.default_rng(seed)
    I, S = win_top3.shape[:2]
    def hits(a_, d_):
        sc = zz(a_)[:, None, :] + (lam * zz(d_)[None] if d_ is not None else 0)
        return int(topk_hits_cells(np.broadcast_to(sc, (I, S, NF)), win_top3).sum())
    obs = hits(init_score, data_score); cnt = 0
    baseI = np.tile(np.arange(NF), (I, 1)); baseS = np.tile(np.arange(NF), (S, 1))
    for _ in range(reps):
        a_ = np.take_along_axis(init_score, rng.permuted(baseI, axis=1), 1)
        d_ = np.take_along_axis(data_score, rng.permuted(baseS, axis=1), 1) if data_score is not None else None
        if hits(a_, d_) >= obs: cnt += 1
    return obs, (cnt + 1) / (reps + 1)

def paired_boot_IS(Ha, Hb, reps=5000, seed=0):
    rng = np.random.default_rng(seed); I, S = Ha.shape; d = []
    D = Ha - Hb
    for _ in range(reps):
        ii = rng.integers(0, I, I); ss = rng.integers(0, S, S)
        d.append(D[np.ix_(ii, ss)].sum())
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

def within_split_spearman(stat, target):
    """stat,target [S,56] -> mean over splits of Spearman rho across the 56 frequencies."""
    from scipy.stats import rankdata
    rs = []
    for s in range(stat.shape[0]):
        a = rankdata(stat[s]); b = rankdata(target[s])
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs))

def perm_p_within(stat, target, reps=10000, seed=0):
    from scipy.stats import rankdata
    rng = np.random.default_rng(seed)
    obs = within_split_spearman(stat, target)
    ra = np.stack([rankdata(x) for x in stat]); rb = np.stack([rankdata(x) for x in target])
    ra = (ra - ra.mean(1, keepdims=True)) / ra.std(1, keepdims=True)
    rb = (rb - rb.mean(1, keepdims=True)) / rb.std(1, keepdims=True)
    cnt = 0
    base = np.tile(np.arange(NF), (stat.shape[0], 1))
    for _ in range(reps):
        rp = np.take_along_axis(ra, rng.permuted(base, axis=1), 1)
        if (rp * rb).mean(1).mean() >= obs - 1e-12: cnt += 1
    return obs, (cnt + 1) / (reps + 1)

def harrell_c(score, time_, event):
    """Higher score = earlier event. Returns concordance over comparable pairs."""
    n = len(score); num = den = 0.0
    for i in range(n):
        if not event[i]: continue
        for j in range(n):
            if time_[i] < time_[j]:
                den += 1
                num += 1.0 if score[i] > score[j] else (0.5 if score[i] == score[j] else 0.0)
    return num / den if den else float("nan")

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()

# -------------------------------------------------------------------------------------
#  Configuration (full vs smoke)
# -------------------------------------------------------------------------------------
if SMOKE:
    N_FACT, N_IND, N_PROXY, N_DM, N_Q3SEED = 3, 3, 4, 8, 2
    STEPS_SHORT, STEPS_PROXY, STEPS_Q2, STEPS_Q3 = 12, 30, 30, 40
    REC_F = (0, 1, 5, 10, 12); REC_IND = (0, 1, 5, 12); WIN_STEP = 12; IND_STEP = 12; DM_STEP = 12
    T0S = (10, 20, 30); Q2_REC = (0, 5, 10, 12); Q2_N = (4, 2, 4)
else:
    N_FACT, N_IND, N_PROXY, N_DM, N_Q3SEED = 32, 64, 64, 1536, 8
    STEPS_SHORT, STEPS_PROXY, STEPS_Q2, STEPS_Q3 = 500, 10000, 15000, 25000
    REC_F = (0, 1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500); REC_IND = (0, 1, 5, 20, 50, 200)
    WIN_STEP = 500; IND_STEP = 200; DM_STEP = 200
    T0S = (500, 1000, 2000); Q2_REC = (0, 10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500); Q2_N = (16, 4, 16)

DEV_INITS = list(range(N_FACT)); DEV_SPLITS = list(range(N_FACT))
TEST_INITS = list(range(1000, 1000 + N_FACT)); TEST_SPLITS = list(range(1000, 1000 + N_FACT))
IND_INITS = list(range(5000, 5000 + N_IND))

# -------------------------------------------------------------------------------------
#  Stage A — sanity: batched == reference, pilot seed 0 groks
# -------------------------------------------------------------------------------------
def stage_A():
    if stage_done("A"): return
    log("Stage A: speed benchmark + equivalence check")
    bs = batch_size_for(); t0 = time.time()
    train_batch([dict(init=i, split=i) for i in range(bs)], 20 if not SMOKE else 3, record=())
    sps = (time.time() - t0) / (21 if not SMOKE else 4)
    est_steps = 2 * N_FACT**2 * STEPS_SHORT + (2 * N_FACT) * N_IND * max(REC_IND) + 2 * N_DM * DM_STEP + N_PROXY * STEPS_PROXY
    log(f"  batch {bs}: {sps:.3f} s/step -> Q1 stages ~{est_steps / bs * sps / 3600:.1f} h, forecasting stage up to "
        f"~{6 * 2 * N_Q3SEED * 4 / bs * STEPS_Q3 * sps / 3600 / 2:.1f}-{6 * 2 * N_Q3SEED * 4 / bs * STEPS_Q3 * sps / 3600:.1f} h")
    res_speed = dict(batch=bs, sec_per_step=sps)
    # reference single-model training with torch.optim.AdamW (pilot recipe)
    class Ref(OneLayer):
        def fwd(self, x):
            W = {k: v[None] for k, v in self.named_parameters()}
            return forward(W, x[None])[0]
    tr, te = split_indices(0); Xg = full_grid(); a, b = grid_pairs(); y = labels("add", a, b).to(DEV)
    res = {}
    torch.manual_seed(0); ref = Ref().to(DEV)
    opt = torch.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1.0))
    for step in range(31):
        loss = F.cross_entropy(ref.fwd(Xg[tr.to(DEV)]), y[tr.to(DEV)])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    r = train_batch([dict(init=0, split=0), dict(init=1, split=0)], 31, record=(31,))
    # note: train_batch records at step 31 *before* the 31st update, i.e. after 31 updates (steps 0..30)
    with torch.no_grad():
        ref_spec = per_freq_share(ref.W_E[None, :P]).cpu().numpy()[0]
    res["max_abs_diff_embed_spectrum"] = float(np.abs(ref_spec - r["rec"][31]["embed"][0]).max())
    log("  batched vs torch AdamW, 31 updates: max |d spectrum| =", res["max_abs_diff_embed_spectrum"])
    res["equivalence_ok"] = res["max_abs_diff_embed_spectrum"] < 2e-3
    if not res["equivalence_ok"]: log("  WARNING: batched trainer differs from torch AdamW reference more than 2e-3")
    # pilot seed 0 full run (cheap on GPU)
    steps = 40 if SMOKE else 12000
    r = train_batch([dict(init=0, split=0)], steps, eval_every=max(steps // 120, 1))
    res["seed0_grok_step"] = r["grok"][0]; res["seed0_final"] = r["final"][0]
    log("  seed 0: grok step", r["grok"][0], "keys", r["final"][0]["keys"], "test acc", r["final"][0]["test_acc"])
    res["speed"] = res_speed
    mark_done("A", res)

# -------------------------------------------------------------------------------------
#  Stage B — init x split factorials (dev, test)
# -------------------------------------------------------------------------------------
def stage_B():
    for name, I, S in [("B_dev", DEV_INITS, DEV_SPLITS), ("B_test", TEST_INITS, TEST_SPLITS)]:
        if stage_done(name): continue
        log(f"Stage {name}: {len(I)}x{len(S)} factorial, {STEPS_SHORT} steps")
        specs = [dict(init=i, split=s) for s in S for i in I]
        d = run_chunked(name, specs, STEPS_SHORT, REC_F)
        if d is not None and len(d["init"]) == len(specs): mark_done(name)

def factorial_arrays(name, step):
    d = load_chunks(name)
    I = sorted(set(d["init"].tolist())); S = sorted(set(d["split"].tolist()))
    st = list(d["steps"]).index(step)
    E = np.full((len(I), len(S), NF), np.nan); A0 = np.full_like(E, np.nan)
    for j in range(len(d["init"])):
        a, b = I.index(d["init"][j]), S.index(d["split"][j])
        E[a, b] = d["embed"][j, st]; A0[a, b] = d["path"][j, 0]
    return I, S, E, A0, d

# -------------------------------------------------------------------------------------
#  Stage C — proxy validation (full runs on dev cells)
# -------------------------------------------------------------------------------------
def stage_C():
    if stage_done("C"): return
    log("Stage C: proxy validation")
    rng = np.random.default_rng(0)
    cells = [(int(i), int(s)) for i, s in zip(rng.choice(DEV_INITS, N_PROXY), rng.choice(DEV_SPLITS, N_PROXY))]
    specs = [dict(init=i, split=s) for i, s in cells]
    out = []
    bs = batch_size_for()
    for c0 in range(0, len(specs), bs):
        f = OUT / f"C_chunk_{c0:04d}.json"
        if f.exists(): out += json.loads(f.read_text()); continue
        r = train_batch(specs[c0:c0 + bs], STEPS_PROXY, record=(200 if not SMOKE else 10, WIN_STEP),
                        eval_every=100 if not SMOKE else 5)
        ch = []
        for m, s in enumerate(specs[c0:c0 + bs]):
            t200 = 200 if not SMOKE else 10
            ch.append(dict(init=s["init"], split=s["split"], grok=r["grok"][m], final_keys=r["final"][m]["keys"],
                           final_test_acc=r["final"][m]["test_acc"],
                           top3_win=np.argsort(-r["rec"][WIN_STEP]["embed"][m])[:3].tolist(),
                           top3_200=np.argsort(-r["rec"][t200]["embed"][m])[:3].tolist()))
        f.write_text(json.dumps(ch)); out += ch
        log(f"  C chunk {c0//bs+1}/{math.ceil(len(specs)/bs)} done")
    def contain(key):
        n = sum(len(o["final_keys"]) for o in out)
        h = sum(sum((k - 1) in o[key] for k in o["final_keys"]) for o in out)
        return h, n
    h5, n = contain("top3_win"); h2, _ = contain("top3_200")
    info = dict(runs=len(out), grokked=sum(o["grok"] is not None for o in out),
                containment_step500=h5 / max(n, 1), containment_step200=h2 / max(n, 1),
                agree_200_500=float(np.mean([len(set(o["top3_win"]) & set(o["top3_200"])) / 3 for o in out])))
    info["proxy_valid"] = info["containment_step500"] >= 0.75
    log("  proxy:", info)
    mark_done("C", info)

# -------------------------------------------------------------------------------------
#  Stage D — independent data votes (64 fresh inits per split) + closed-form data statistics
# -------------------------------------------------------------------------------------
def stage_D():
    if not stage_done("D_ind"):
        log("Stage D: independent-init data votes")
        specs = [dict(init=i, split=s) for s in DEV_SPLITS + TEST_SPLITS for i in IND_INITS]
        d = run_chunked("D_ind", specs, max(REC_IND), REC_IND)
        if d is not None and len(d["init"]) == len(specs): mark_done("D_ind")
    if not stage_done("D_stats"):
        log("Stage D: closed-form data statistics")
        stats = {s: data_stats(s) for s in DEV_SPLITS + TEST_SPLITS}
        np.savez(OUT / "D_stats.npz", splits=np.array(list(stats)),
                 **{k: np.stack([stats[s][k] for s in stats]) for k in next(iter(stats.values()))})
        mark_done("D_stats")

@torch.no_grad()
def data_stats(split_seed, frac=0.3):
    tr, _ = split_indices(split_seed, frac); a, b = grid_pairs()
    a = a[tr].to(DEV).double(); b = b[tr].to(DEV).double(); N = len(a)
    w = 2 * math.pi * torch.arange(1, NF + 1, device=DEV, dtype=torch.float64) / P
    E = lambda ph: torch.exp(1j * w[:, None] * ph[None])
    out = {}
    out["D1"] = E(a + b).sum(1).abs(); out["D2"] = E(a - b).sum(1).abs()
    out["D3"] = E(a).sum(1).abs() ** 2 + E(b).sum(1).abs() ** 2
    m = torch.arange(P, device=DEV, dtype=torch.float64)
    Cm = torch.exp(1j * 2 * math.pi * m[:, None] * (a + b)[None] / P).sum(1)
    js = torch.tensor(list(range(1, NF + 1)) + [-q for q in range(1, NF + 1)], device=DEV)
    D4 = []
    for k in range(1, NF + 1):
        jj = js[js != k]
        D4.append((Cm[(k - jj) % P].abs() ** 2).sum())
    out["D4"] = torch.stack(D4)
    ca, sa, cb, sb = (torch.cos(w[:, None] * a), torch.sin(w[:, None] * a), torch.cos(w[:, None] * b), torch.sin(w[:, None] * b))
    feats = torch.stack([ca, sa, cb, sb, ca * cb, ca * sb, sa * cb, sa * sb], 1)       # [56,8,N]
    G = feats @ feats.transpose(1, 2) / N
    ev = torch.linalg.eigvalsh(G)
    out["D5min"] = ev[:, 0]; out["D5cond"] = ev[:, -1] / ev[:, 0]
    allf = feats.reshape(NF * 8, N).T
    D5r = []
    for t in range(NF):
        keep = torch.ones(NF * 8, dtype=torch.bool, device=DEV); keep[8 * t:8 * t + 8] = False
        others = allf[:, keep]
        tgt = torch.stack([torch.cos(w[t] * (a + b)), torch.sin(w[t] * (a + b))], 1)
        coef = torch.linalg.lstsq(others, tgt).solution
        r = tgt - others @ coef
        D5r.append((r ** 2).sum() / (tgt ** 2).sum())
    out["D5r"] = torch.stack(D5r)
    return {k: v.real.float().cpu().numpy() if torch.is_complex(v) else v.float().cpu().numpy() for k, v in out.items()}

def ind_vote(split_list, step=None):
    step = step or IND_STEP
    d = load_chunks("D_ind"); st = list(d["steps"]).index(step)
    z = zscore_log(d["embed"][:, st])
    return np.stack([z[d["split"] == s].mean(0) for s in split_list])     # [S,56]

# -------------------------------------------------------------------------------------
#  Stage E — causal data test (datamodels + interventions)
# -------------------------------------------------------------------------------------
def stage_E():
    if stage_done("E"): return
    log("Stage E: causal data test")
    I, S, E, A0, _ = factorial_arrays("B_dev", WIN_STEP)
    Z = zscore_log(E)
    beta = Z.mean(0)                                                    # [S,56]
    order = np.argsort(-beta.max(1))[:2]
    chosen = [(S[o], int(beta[o].argmax())) for o in order]             # (split, k* 0-indexed)
    log("  chosen splits/k*:", [(s, k + 1) for s, k in chosen])
    rng = np.random.default_rng(123)
    results = []
    for si, (split, kstar) in enumerate(chosen):
        tr, te = split_indices(split); N = len(tr); nrm = N // 10
        fdm = OUT / f"E_dm_{split}.npz"
        if not fdm.exists():
            masks = np.zeros((N_DM, N), dtype=bool); outs = np.zeros(N_DM); bs = batch_size_for()
            custom = []
            for r_ in range(N_DM):
                rm = rng.choice(N, nrm, replace=False); masks[r_, rm] = True
                add = te[torch.from_numpy(rng.choice(len(te), nrm, replace=False))]
                keep = tr[torch.from_numpy(np.setdiff1d(np.arange(N), rm))]
                custom.append(torch.cat([keep, add]))
            for c0 in range(0, N_DM, bs):
                sp = [dict(init=8000 + si * 10000 + r_, split=split) for r_ in range(c0, min(N_DM, c0 + bs))]
                rr = train_batch(sp, DM_STEP, record=(DM_STEP,), custom_train_idx=custom[c0:c0 + bs])
                outs[c0:c0 + len(sp)] = zscore_log(rr["rec"][DM_STEP]["embed"])[:, kstar]
                if (c0 // bs) % 8 == 0: log(f"  datamodel split {split}: {c0+len(sp)}/{N_DM}")
            np.savez(fdm, masks=masks, outs=outs)
        dm = np.load(fdm); masks, outs = dm["masks"].astype(float), dm["outs"]
        coef = ridge_cv(masks, outs)                                     # effect of REMOVING pair j on z(k*)
        top = np.argsort(coef)[:nrm]                                     # removal lowers k* the most
        bot = np.argsort(coef)[-nrm:]
        conds = {"A_top": [top], "C_bottom": [bot],
                 "B_random": [rng.choice(N, nrm, replace=False) for _ in range(5)]}
        cond_out = {}
        for cname, rms in conds.items():
            vals = []
            for rep, rm in enumerate(rms):
                add = te[torch.from_numpy(np.random.default_rng(1000 + rep).choice(len(te), nrm, replace=False))]
                keep = tr[torch.from_numpy(np.setdiff1d(np.arange(N), rm))]
                idx = torch.cat([keep, add])
                n_init = 8 if SMOKE else 64
                sp = [dict(init=9000 + j, split=split) for j in range(n_init)]
                zs = []
                for c0 in range(0, n_init, batch_size_for()):
                    rr = train_batch(sp[c0:c0 + batch_size_for()], DM_STEP, record=(DM_STEP,),
                                     custom_train_idx=[idx] * len(sp[c0:c0 + batch_size_for()]))
                    zs.append(zscore_log(rr["rec"][DM_STEP]["embed"])[:, kstar])
                vals.append(np.concatenate(zs))
            cond_out[cname] = np.stack(vals)                              # [reps, n_init]
        A = cond_out["A_top"][0]; Bm = cond_out["B_random"].mean(0); Cc = cond_out["C_bottom"][0]
        diff = Bm - A; obs = diff.mean()
        prng = np.random.default_rng(7); cnt = 0
        for _ in range(10000):
            if (diff * prng.choice([-1, 1], len(diff))).mean() >= obs: cnt += 1
        p = (cnt + 1) / 10001
        results.append(dict(split=split, kstar=kstar + 1, mean_A=float(A.mean()), mean_B=float(Bm.mean()),
                            mean_C=float(Cc.mean()), win_A=float(np.mean(A > 0)), p_one_sided=p,
                            coef_top=coef[top[:20]].tolist(),
                            top_pairs=[[int(x) for x in divmod(int(tr[j]), P)] for j in top[:383]]))
        log(f"  split {split} k*={kstar+1}: A {A.mean():.3f}  B {Bm.mean():.3f}  C {Cc.mean():.3f}  p {p:.4f}")
    adj = holm([r["p_one_sided"] for r in results])
    for r, a_ in zip(results, adj):
        r["p_holm"] = a_; r["PASS"] = bool(a_ < 0.01 and r["mean_C"] > r["mean_A"])
    mark_done("E", dict(results=results))

def ridge_cv(X, y, alphas=(1, 10, 100, 1000, 10000), folds=5):
    """Ridge in dual form (n samples << p features): w = X^T (X X^T + a I)^-1 y. Alpha by k-fold CV."""
    X = X - X.mean(0); y = y - y.mean(); n = len(y)
    idx = np.random.default_rng(0).permutation(n); fs = np.array_split(idx, folds)
    def fit(Xt, yt, al): return Xt.T @ np.linalg.solve(Xt @ Xt.T + al * np.eye(len(yt)), yt)
    best, besta = 1e18, alphas[0]
    for al in alphas:
        err = 0.0
        for f in fs:
            trn = np.setdiff1d(idx, f)
            err += ((X[f] @ fit(X[trn], y[trn], al) - y[f]) ** 2).sum()
        if err < best: best, besta = err, al
    return fit(X, y, besta)

# -------------------------------------------------------------------------------------
#  Stage F — pre-registered analysis of Q1 (C1, C2, C4, R1)
# -------------------------------------------------------------------------------------
def votes(E):
    Z = zscore_log(E); I, S = Z.shape[:2]
    iv = np.stack([np.stack([np.delete(Z[a], b, 0).mean(0) for b in range(S)]) for a in range(I)])
    dv = np.stack([np.stack([np.delete(Z[:, b], a, 0).mean(0) for b in range(S)]) for a in range(I)])
    return Z, iv, dv
def zz(x): return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)

def stage_F():
    log("Stage F: pre-registered Q1 analysis")
    res = dict(prereg_sha256=PREREG_HASH)
    out_test = {}
    for name in ("B_dev", "B_test"):
        I, S, E, A0, _ = factorial_arrays(name, WIN_STEP)
        Z, iv, dv = votes(E); W3 = np.argsort(-E, -1)[..., :3]
        out_test[name] = dict(I=I, S=S, E=E, A0=A0, iv=iv, dv=dv, W3=W3)
    # C1
    t = out_test["B_test"]; n = len(t["I"]) * len(t["S"])
    H = {k: topk_hits_cells(v, t["W3"]) for k, v in
         dict(init=t["iv"], data=t["dv"], both=zz(t["iv"]) + zz(t["dv"])).items()}
    ci1 = paired_boot_IS(H["both"], H["init"]); ci2 = paired_boot_IS(H["both"], H["data"])
    res["C1"] = dict(hits={k: int(v.sum()) for k, v in H.items()}, of=3 * n, chance=n * 9 / NF,
                     ci_both_minus_init=ci1, ci_both_minus_data=ci2, PASS=bool(ci1[0] > 0 and ci2[0] > 0))
    # same on dev (exploratory)
    dd = out_test["B_dev"]
    res["C1_dev_exploratory"] = {k: int(topk_hits_cells(v, dd["W3"]).sum()) for k, v in
                                 dict(init=dd["iv"], data=dd["dv"], both=zz(dd["iv"]) + zz(dd["dv"])).items()}
    # C2: closed-form statistic chosen on dev against independent votes
    st = np.load(OUT / "D_stats.npz"); splits = list(st["splits"])
    dev_rows = [splits.index(s) for s in DEV_SPLITS]; test_rows = [splits.index(s) for s in TEST_SPLITS]
    tgt_dev = ind_vote(DEV_SPLITS); tgt_test = ind_vote(TEST_SPLITS)
    cands = {}
    for name in ["D1", "D2", "D3", "D4", "D5min", "D5cond", "D5r"]:
        x = np.log(np.abs(st[name]) + 1e-30)
        for sign in (1, -1):
            cands[(name, sign)] = within_split_spearman(sign * x[dev_rows], tgt_dev)
    best = max(cands, key=cands.get)
    x = best[1] * np.log(np.abs(st[best[0]]) + 1e-30)
    rho, p = perm_p_within(x[test_rows], tgt_test)
    res["C2"] = dict(dev_scores={f"{k[0]}{'+' if k[1] > 0 else '-'}": v for k, v in cands.items()},
                     chosen=f"{best[0]}{'+' if best[1] > 0 else '-'}", test_rho=rho, test_p=p,
                     PASS=bool(rho > 0 and p < 0.01))
    # reliability of the independent vote vs factorial data vote (descriptive)
    res["ind_vote_vs_factorial_dv_test"] = within_split_spearman(tgt_test, t["dv"].mean(0))
    # C4 / R1
    def combo(tset, lam, dstat):
        return zz(np.log(tset["A0"])) + lam * zz(dstat)[None]
    lam_grid = [0, 0.25, 0.5, 1, 2, 4]
    dev_hits = {lam: int(topk_hits_cells(combo(dd, lam, tgt_dev), dd["W3"]).sum()) for lam in lam_grid}
    lam = max(dev_hits, key=dev_hits.get)
    Hc = topk_hits_cells(combo(t, lam, tgt_test), t["W3"]); Ha = topk_hits_cells(combo(t, 0, tgt_test), t["W3"])
    ci = paired_boot_IS(Hc, Ha)
    a0_test = np.log(t["A0"][:, 0])                                      # attention path at init depends on the init only
    _, p_comb = perm_p_hits(a0_test, tgt_test, lam, t["W3"]) if lam > 0 else perm_p_hits(a0_test, None, 0, t["W3"])
    res["C4"] = dict(dev_hits_by_lambda=dev_hits, lambda_=lam, hits_combined=int(Hc.sum()), hits_attn=int(Ha.sum()),
                     of=3 * n, chance=n * 9 / NF, ci_diff=ci, p_combined_vs_chance=p_comb,
                     PASS=bool(ci[0] > 0 and p_comb < 1e-3))
    xs_dev = x[dev_rows]; xs_test = x[test_rows]
    dev_hits_b = {l: int(topk_hits_cells(combo(dd, l, xs_dev), dd["W3"]).sum()) for l in lam_grid}
    lb = max(dev_hits_b, key=dev_hits_b.get)
    Hb = topk_hits_cells(combo(t, lb, xs_test), t["W3"])
    res["C4b"] = dict(lambda_=lb, hits=int(Hb.sum()), ci_diff_vs_attn=paired_boot_IS(Hb, Ha))
    _, p_r1 = perm_p_hits(a0_test, None, 0, t["W3"])
    res["R1"] = dict(hits=int(Ha.sum()), of=3 * n, chance=n * 9 / NF, p=p_r1,
                     p_hypergeometric_reference=p_above_chance(int(Ha.sum()), n), PASS=bool(p_r1 < 1e-3))
    # oracle ceiling (descriptive): factorial votes from the cell's own split/init
    res["oracle_both_test_hits"] = res["C1"]["hits"]["both"]
    # when does the data vote appear? correlation of independent vote at early steps with step-200 vote (descriptive)
    res["data_vote_timecourse_test"] = {int(s_): within_split_spearman(ind_vote(TEST_SPLITS, int(s_)), tgt_test)
                                        for s_ in REC_IND if s_ > 0}
    (OUT / "F_q1_results.json").write_text(json.dumps(res, indent=1, default=float))
    log("  C1", res["C1"]["hits"], res["C1"]["PASS"], "| C2", res["C2"]["chosen"], round(rho, 3), p, res["C2"]["PASS"],
        "| C4", res["C4"]["hits_combined"], "vs", res["C4"]["hits_attn"], res["C4"]["PASS"], "| R1", res["R1"]["PASS"])
    return res

# -------------------------------------------------------------------------------------
#  Stage H — generality (Q2)
# -------------------------------------------------------------------------------------
def stage_H():
    if stage_done("H"): return
    log("Stage H: generality")
    n_init, n_split, n_full = Q2_N
    win_step = max(Q2_REC)
    res = {}
    for task, arch in [("sub", "attn"), ("add", "mlp")]:
        tag = f"{task}_{arch}"
        f = OUT / f"H_{tag}.json"
        if f.exists(): res[tag] = json.loads(f.read_text()); continue
        specs = [dict(init=i, split=s) for s in range(n_split) for i in range(n_init)]
        d = run_chunked(f"H_short_{tag}", specs, win_step, Q2_REC, arch=arch, task=task)
        E = d["embed"]; steps = list(d["steps"])
        top = np.argsort(-E, -1)[..., :3]
        commit = []
        for r_ in range(len(E)):
            fin = set(top[r_, -1]); cs = win_step
            for j in range(len(steps) - 1, -1, -1):
                if set(top[r_, j]) == fin: cs = steps[j]
                else: break
            commit.append(cs)
        path0 = d["path"][:, 0]
        hits = int(sum(len(set(np.argsort(-path0[r_])[:3]) & set(top[r_, -1])) for r_ in range(len(E))))
        inits_ = sorted(set(d["init"].tolist())); splits_ = sorted(set(d["split"].tolist()))
        W3h = np.zeros((len(inits_), len(splits_), 3), int); A0h = np.zeros((len(inits_), NF))
        for r_ in range(len(E)):
            ia, sb = inits_.index(d["init"][r_]), splits_.index(d["split"][r_])
            W3h[ia, sb] = top[r_, -1]; A0h[ia] = np.log(path0[r_])
        _, path_p = perm_p_hits(A0h, None, 0, W3h)
        fr = []
        bs = batch_size_for(arch)
        fspecs = [dict(init=i, split=i % n_split) for i in range(n_full)]
        for c0 in range(0, n_full, bs):
            rr = train_batch(fspecs[c0:c0 + bs], STEPS_Q2, eval_every=100 if not SMOKE else 5, arch=arch, task=task,
                             stop_after_grok=1000 if not SMOKE else 5)
            fr += [dict(grok=rr["grok"][m], keys=rr["final"][m]["keys"], test_acc=rr["final"][m]["test_acc"])
                   for m in range(len(fspecs[c0:c0 + bs]))]
        gs = [x["grok"] for x in fr if x["grok"] is not None]
        out = dict(n_grok=len(gs), n_full=n_full, median_grok=float(np.median(gs)) if gs else None,
                   median_commit=float(np.median(commit)), path_predictor_hits=hits, of=3 * len(E),
                   path_p=path_p, full=fr)
        out["applicable"] = len(gs) >= 0.75 * n_full
        out["PASS"] = bool(out["applicable"] and out["median_commit"] <= 0.05 * out["median_grok"]) if out["applicable"] else None
        f.write_text(json.dumps(out, default=float)); res[tag] = out
        log(f"  {tag}: grokked {len(gs)}/{n_full}, median grok {out['median_grok']}, median commit {out['median_commit']}, "
            f"path hits {hits}/{3*len(E)} (p={out['path_p']:.2g})")
    # base config, from B_dev (commitment) and C (grok steps)
    d = load_chunks("B_dev"); E = d["embed"]; steps = list(d["steps"]); top = np.argsort(-E, -1)[..., :3]
    commit = []
    for r_ in range(len(E)):
        fin = set(top[r_, -1]); cs = steps[-1]
        for j in range(len(steps) - 1, -1, -1):
            if set(top[r_, j]) == fin: cs = steps[j]
            else: break
        commit.append(cs)
    cinfo = json.loads((OUT / "C.done").read_text())
    groks = []
    for fC in sorted(OUT.glob("C_chunk_*.json")): groks += [o["grok"] for o in json.loads(fC.read_text()) if o["grok"] is not None]
    res["add_attn_base"] = dict(median_commit=float(np.median(commit)), median_grok=float(np.median(groks)) if groks else None,
                                n_grok=len(groks), n_full=cinfo["runs"])
    b = res["add_attn_base"]; b["applicable"] = b["n_grok"] >= 0.75 * b["n_full"]
    b["PASS"] = bool(b["applicable"] and b["median_commit"] <= 0.05 * b["median_grok"]) if b["applicable"] else None
    mark_done("H", res)

# -------------------------------------------------------------------------------------
#  Stage G — forecasting grok time (Q3)
# -------------------------------------------------------------------------------------
def stage_G():
    if stage_done("G"): return
    log("Stage G: forecasting")
    wds = [0.1, 0.3, 1.0, 3.0]; fracs = [0.25, 0.3, 0.4]
    rows = []
    for frac in fracs:
        for part, seeds in [("dev", range(N_Q3SEED)), ("test", range(1000, 1000 + N_Q3SEED))]:
            f = OUT / f"G_{frac}_{part}.json"
            if f.exists(): rows += json.loads(f.read_text()); continue
            if time_left_h() < 0.5: log("  G: out of time"); break
            specs = [dict(init=s, split=3000 + s, wd=w) for w in wds for s in seeds]
            bs = batch_size_for()
            part_rows = []
            for c0 in range(0, len(specs), bs):
                sp = specs[c0:c0 + bs]
                rr = train_batch(sp, STEPS_Q3, frac=frac, eval_every=100 if not SMOKE else 5, analyze_at=T0S,
                                 stop_after_grok=500 if not SMOKE else 2)
                for m, s in enumerate(sp):
                    feats = {}
                    curves = rr["curves"]
                    for t0 in T0S:
                        an = rr["analyses"][t0][m]
                        rec = [c for c in curves if c["step"] <= t0]
                        prev = [c for c in curves if c["step"] <= t0 - (500 if not SMOKE else 10)]
                        def slope(key):
                            if not prev: return 0.0
                            return (rec[-1][key][m] - prev[-1][key][m])
                        feats[t0] = dict(
                            neuron_key_share=an["neuron_key_share"],
                            restricted_ratio=math.log((an.get("restricted_loss", an["test_loss"]) + 1e-9) / (an["test_loss"] + 1e-9)),
                            excluded_minus_clean=an.get("excluded_loss", an["test_loss"]) - an["test_loss"],
                            wnorm_slope=slope("wnorm"),
                            b_test_loss=an["test_loss"], b_test_acc=an["test_acc"], b_train_loss=an["train_loss"],
                            b_test_loss_slope=slope("test_loss"))
                    part_rows.append(dict(part=part, frac=frac, wd=s["wd"], init=s["init"], grok=rr["grok"][m],
                                          last_step=rr["last_step"], feats=feats))
            f.write_text(json.dumps(part_rows, default=float)); rows += part_rows
            log(f"  G frac {frac} {part}: grokked {sum(r['grok'] is not None for r in part_rows)}/{len(part_rows)}")
    res = analyze_G(rows)
    log("  C6:", res)
    mark_done("G", res)

def analyze_G(rows, cap=None):
    cap = cap or STEPS_Q3
    def arrays(part, metric, t0):
        rs = [r for r in rows if r["part"] == part]
        sc = np.array([(r["feats"].get(str(t0)) or r["feats"].get(t0))[metric] for r in rs], float)
        tm = np.array([r["grok"] if r["grok"] is not None else cap + 1 for r in rs], float)
        ev = np.array([r["grok"] is not None for r in rs])
        return sc, tm, ev
    metrics_int = ["neuron_key_share", "restricted_ratio", "excluded_minus_clean", "wnorm_slope"]
    metrics_beh = ["b_test_loss", "b_test_acc", "b_train_loss", "b_test_loss_slope"]
    def best(metrics):
        bestc, bestk = -1.0, None
        for mtr in metrics:
            for t0 in T0S:
                for sign in (1, -1):
                    sc, tm, ev = arrays("dev", mtr, t0)
                    c = harrell_c(sign * sc, tm, ev)
                    if not np.isnan(c) and c > bestc: bestc, bestk = c, (mtr, t0, sign)
        return bestk, bestc
    if not any(r["part"] == "test" for r in rows): return dict(note="incomplete: no test runs", PASS=None)
    ki, cdi = best(metrics_int); kb, cdb = best(metrics_beh)
    if ki is None or kb is None:
        return dict(note="no comparable pairs on dev (too few grokked runs)", PASS=None,
                    n_dev_grokked=int(sum(r["grok"] is not None for r in rows if r["part"] == "dev")))
    sci, tm, ev = arrays("test", ki[0], ki[1]); scb, _, _ = arrays("test", kb[0], kb[1])
    ci_ = harrell_c(ki[2] * sci, tm, ev); cb_ = harrell_c(kb[2] * scb, tm, ev)
    rng = np.random.default_rng(0); ds = []
    for _ in range(2000 if not SMOKE else 50):
        j = rng.integers(0, len(tm), len(tm))
        a_ = harrell_c(ki[2] * sci[j], tm[j], ev[j]); b_ = harrell_c(kb[2] * scb[j], tm[j], ev[j])
        if not (np.isnan(a_) or np.isnan(b_)): ds.append(a_ - b_)
    ci = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if ds else [float("nan")] * 2
    ok = not (np.isnan(ci_) or np.isnan(cb_))
    return dict(internal_choice=list(ki), internal_dev_C=cdi, behavioural_choice=list(kb), behavioural_dev_C=cdb,
                internal_test_C=ci_, behavioural_test_C=cb_, diff_ci=ci,
                n_test=len(tm), n_test_grokked=int(ev.sum()),
                PASS=bool(ok and ci_ >= 0.75 and ci_ - cb_ >= 0.05 and ci[0] > 0) if ok else None)

# -------------------------------------------------------------------------------------
#  Driver
# -------------------------------------------------------------------------------------
def summary():
    s = dict(prereg_sha256=PREREG_HASH, device=str(DEV), smoke=SMOKE,
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             torch=torch.__version__, minutes=(time.time() - T_START) / 60)
    for n in ["A", "C", "E", "H", "G"]:
        f = OUT / f"{n}.done"
        if f.exists(): s[n] = json.loads(f.read_text())
    f = OUT / "F_q1_results.json"
    if f.exists(): s["F"] = json.loads(f.read_text())
    (OUT / "SUMMARY.json").write_text(json.dumps(s, indent=1, default=float))
    return s

def run_all():
    (OUT / "PREREG.txt").write_text(PREREG + f"\nsha256: {PREREG_HASH}\n")
    log("device", DEV, "| out", OUT, "| smoke", SMOKE, "| prereg", PREREG_HASH[:16])
    for fn in [stage_A, stage_B, stage_C, stage_D, stage_E]:
        if time_left_h() < 0.3: log("stopping: time budget"); break
        fn(); summary()
    try:
        stage_F()
    except Exception as e:
        log("stage F failed:", repr(e))
    summary()
    for fn in [stage_H, stage_G]:
        if time_left_h() < 0.5: log("skipping", fn.__name__, "(time)"); continue
        try: fn()
        except Exception as e: log(fn.__name__, "failed:", repr(e))
        summary()
    s = summary(); log("ALL DONE"); return s

if __name__ == "__main__":
    run_all()
