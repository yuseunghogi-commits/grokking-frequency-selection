# =====================================================================================
#  v2 — Anatomy of the data's vote.  Which part of the data casts it: inputs, labels, task,
#  or architecture?  Is a pair's influence a property of the data (same across inits)?
#  When and where does the vote first appear?   Uses gpu_grok.py (v1 core) for the model.
# =====================================================================================
import os
from pathlib import Path
if "GROK_OUT" not in os.environ:
    for base in ("/content/drive/MyDrive", "/kaggle/working"):
        if Path(base).exists(): os.environ["GROK_OUT"] = f"{base}/grok2_out"; break
    else: os.environ["GROK_OUT"] = "grok2_out"
import json, time, math, hashlib
import numpy as np, torch, torch.nn.functional as F
import gpu_grok as C                                    # v1 core (model, data, optimiser, stats)
from gpu_grok import (P, NF, DEV, OUT, log, init_state, split_indices, grid_pairs, labels, full_grid,
                      forward, StackedAdamW, lr_at, per_freq_share, path_spectrum, zscore_log, batch_size_for,
                      within_split_spearman, perm_p_within, ridge_cv, SMOKE, time_left_h, stage_done, mark_done)

PREREG2 = r"""
PRE-REGISTRATION v2 — written 2026-10-18 (KST) before any run of this notebook.
All IDs are fresh (never used before): splits 2000-2031 (frac 0.3, p=113, wd 1.0, AdamW as v1). Each condition uses its
  own 32 inits (TRUE 2000-2031, SHUF 2032-2063, SHIFT 2064-2095, SUB 2096-2127, MLP 2128-2159), so no init is shared.
Data vote of condition c on split s: V_c[s,k] = mean over that condition's 32 inits of z(log embedding Fourier share at
  step 200), minus its mean over the 32 splits (removes any preference common to all splits, e.g. an architecture prior).
Conditions (same training pairs (a,b) for every condition of a split):
  TRUE  : labels (a+b) mod p, 1-layer attention model (the v1 setting)
  SHUF  : the same pairs, labels permuted among the training pairs (fixed permutation per split)
  SHIFT : labels (a+b+7) mod p  (relabelling the outputs; expected to leave the vote unchanged)
  SUB   : labels (a-b) mod p
  MLP   : labels (a+b) mod p, attention-free MLP model (v1 Stage H architecture)
Agreement A(c) = mean over splits of within-split Spearman(V_TRUE[s], V_c[s]) across the 56 frequencies;
  p from a within-split permutation null (10000 perms). Ceiling(c) = sqrt(rel_TRUE * rel_c), rel = Spearman-Brown
  corrected split-half (16 vs 16 inits) reliability of the vote. Disattenuated agreement = A(c)/Ceiling(c).
H1 (inputs vs labels): PASS iff A(SHUF) > 0 with p < 0.01.   [vote is cast, at least partly, by WHICH pairs are present]
H2 (sanity / validity): PASS iff A(SHIFT) >= 0.8 * Ceiling(SHIFT).
H3 (task): PASS iff A(SUB) > 0 with p < 0.01.
H4 (architecture): PASS iff A(MLP) > 0 with p < 0.01.
H5 (pair influence is a property of the data): splits 2000-2003, k* = argmax_k V_TRUE[s,k]. For each split, two
  fixed-init datamodels (init 3100 and 3101; 1024 runs each; 10% of training pairs swapped for test pairs; outcome
  z(k*) at step 200; dual ridge, alpha by 5-fold CV). Influence vectors from the two inits are compared by Spearman
  across the 3830 pairs; p by permuting pairs (2000 perms). PASS iff rho > 0 with p < 0.01 in >= 3 of the 4 splits.
  The held-out R^2 of each datamodel is reported (descriptive).
Timecourse (descriptive, no test): splits 2000-2007 x inits 3200-3215, spectra every step 0-100 of the embedding,
  attention path, embedding->MLP path and unembedding; per quantity and step, Spearman of its data vote with V_TRUE.
Failures are reported as failures. Holm adjustment across H1, H3, H4 is reported.
"""
PREREG2_HASH = hashlib.sha256(PREREG2.encode()).hexdigest()

if SMOKE:
    N_I, N_S, N_DM, DM_SPLITS, TC_S, TC_I, STEPS, TC_STEPS = 3, 3, 8, 2, 2, 2, 12, 12
    REC = (0, 5, 12); VOTE_STEP = 12; TC_REC = tuple(range(0, 13))
else:
    N_I, N_S, N_DM, DM_SPLITS, TC_S, TC_I, STEPS, TC_STEPS = 32, 32, 1024, 4, 8, 16, 200, 200
    REC = (0, 20, 50, 100, 200); VOTE_STEP = 200; TC_REC = tuple(range(0, 101)) + (150, 200)
SPLITS = list(range(2000, 2000 + N_S))
COND_INIT0 = {"TRUE": 2000, "SHUF": 2032, "SHIFT": 2064, "SUB": 2096, "MLP": 2128}
def inits_of(c): return list(range(COND_INIT0[c], COND_INIT0[c] + N_I))
CONDS = {"TRUE": ("add", "true", "attn"), "SHUF": ("add", "shuffle", "attn"), "SHIFT": ("add", "shift7", "attn"),
         "SUB": ("sub", "true", "attn"), "MLP": ("add", "true", "mlp")}

# -------------------------------------------------------------------------------------
def train_labels(task, mode, split, tr):
    a, b = grid_pairs(); y = labels(task, a, b)[tr]
    if mode == "shift7": y = (y + 7) % P
    elif mode == "shuffle":
        g = torch.Generator().manual_seed(77000 + split)
        y = y[torch.randperm(len(y), generator=g)]
    return y

def spectra(W, arch):
    out = dict(embed=per_freq_share(W["W_E"][:, :P]), path=path_spectrum(W, arch),
               unembed=per_freq_share(W["W_U"].transpose(1, 2)))
    if arch == "attn": out["win"] = per_freq_share(W["W_E"][:, :P] @ W["W_in"])
    return {k: v.cpu().numpy() for k, v in out.items()}

def train2(specs, steps, record, arch="attn", custom_train_idx=None):
    """specs: dict(init, split, task, label). One arch per batch. Returns {step: spectra dict}."""
    M = len(specs)
    sts = [init_state(s["init"], arch) for s in specs]
    W = {k: torch.stack([st[k] for st in sts]).to(DEV).requires_grad_(True) for k in sts[0]}
    G = full_grid(); TRs, Ys = [], []
    for j, s in enumerate(specs):
        tr, _ = split_indices(s["split"])
        if custom_train_idx is not None: tr = custom_train_idx[j]
        TRs.append(tr); Ys.append(train_labels(s.get("task", "add"), s.get("label", "true"), s["split"], tr))
    TR = torch.stack(TRs).to(DEV); X = G[TR]; Y = torch.stack(Ys).to(DEV)
    opt = StackedAdamW(W, 1e-3, torch.full((M,), 1.0, device=DEV))
    rec = {}; record = set(record)
    for step in range(steps + 1):
        logits = forward(W, X, arch)
        if step in record:
            with torch.no_grad():
                rec[step] = spectra(W, arch)
                rec[step]["train_acc"] = (logits.argmax(-1) == Y).float().mean(1).cpu().numpy()
        if step == steps: break
        F.cross_entropy(logits.reshape(-1, P), Y.reshape(-1), reduction="none").reshape(M, -1).mean(1).sum().backward()
        opt.step(lr_at(step))
    return rec

def run_specs(stage, specs, steps, record, arch, keys=("embed",), custom=None):
    d = OUT / stage; d.mkdir(exist_ok=True)
    bs = batch_size_for(arch); rec_steps = sorted(record)
    chunks = [list(range(i, min(len(specs), i + bs))) for i in range(0, len(specs), bs)]
    t0 = time.time(); n_new = 0
    for ci, ix in enumerate(chunks):
        f = d / f"chunk_{ci:04d}.npz"
        if f.exists(): continue
        if time_left_h() < 0.3: log(f"{stage}: out of time at chunk {ci}"); break
        r = train2([specs[i] for i in ix], steps, rec_steps, arch,
                   custom_train_idx=[custom[i] for i in ix] if custom is not None else None)
        arrs = {k: np.stack([r[t][k] for t in rec_steps], 1).astype(np.float32) for k in keys if k in r[rec_steps[0]]}
        np.savez(str(f) + ".tmp.npz", idx=np.array(ix), steps=np.array(rec_steps),
                 train_acc=np.stack([r[t]["train_acc"] for t in rec_steps], 1), **arrs)
        os.replace(str(f) + ".tmp.npz", f); n_new += 1
        el = time.time() - t0
        log(f"{stage}: {ci+1}/{len(chunks)}  ETA {el/n_new*(len(chunks)-ci-1)/60:.0f} min")
    fs = sorted(f for f in d.glob("chunk_*.npz") if not f.name.endswith(".tmp.npz"))
    ds = [np.load(f) for f in fs]
    out = {k: np.concatenate([x[k] for x in ds]) for k in ds[0].files if k != "steps"}
    out["steps"] = ds[0]["steps"]
    return out

# -------------------------------------------------------------------------------------
def stage_votes():
    for name, (task, lab, arch) in CONDS.items():
        if stage_done(f"V_{name}"): continue
        log(f"votes: condition {name} ({task}, labels={lab}, arch={arch})")
        specs = [dict(init=i, split=s, task=task, label=lab) for s in SPLITS for i in inits_of(name)]
        d = run_specs(f"V_{name}", specs, STEPS, REC, arch)
        if len(d["idx"]) == len(specs): mark_done(f"V_{name}")

def vote_matrix(name, step=None, inits_subset=None):
    fs = sorted(f for f in (OUT / f"V_{name}").glob("chunk_*.npz") if not f.name.endswith(".tmp.npz"))
    ds = [np.load(f) for f in fs]
    idx = np.concatenate([x["idx"] for x in ds]); E = np.concatenate([x["embed"] for x in ds])
    st = list(ds[0]["steps"]).index(step or VOTE_STEP)
    Z = zscore_log(E[:, st])                                        # [runs,56]
    spl = np.array([SPLITS[i // N_I] for i in idx]); ini = np.array([i % N_I for i in idx])   # init position 0..N_I-1
    keep = np.ones(len(idx), bool) if inits_subset is None else np.isin(ini, inits_subset)
    V = np.stack([Z[(spl == s) & keep].mean(0) for s in SPLITS])     # [S,56]
    return V - V.mean(0, keepdims=True)                               # split-specific part only

def reliability(name):
    h1 = vote_matrix(name, inits_subset=list(range(0, N_I, 2))); h2 = vote_matrix(name, inits_subset=list(range(1, N_I, 2)))
    r = within_split_spearman(h1, h2)
    return 2 * r / (1 + r) if r > -1 else float("nan")

def stage_datamodels():
    if stage_done("DM"): return
    log("fixed-init datamodels")
    rng = np.random.default_rng(321)
    for s in SPLITS[:DM_SPLITS]:
        for init in (3100, 3101):
            tag = f"DM_{s}_{init}"
            if (OUT / f"{tag}.npz").exists(): continue
            tr, te = split_indices(s); N = len(tr); nrm = N // 10
            masks = np.zeros((N_DM, N), bool); custom = []
            for r_ in range(N_DM):
                rm = rng.choice(N, nrm, replace=False); masks[r_, rm] = True
                add = te[torch.from_numpy(rng.choice(len(te), nrm, replace=False))]
                custom.append(torch.cat([tr[torch.from_numpy(np.setdiff1d(np.arange(N), rm))], add]))
            specs = [dict(init=init, split=s) for _ in range(N_DM)]
            d = run_specs(tag, specs, STEPS, (VOTE_STEP,), "attn", custom=custom)
            np.savez(OUT / f"{tag}.npz", masks=masks, z=zscore_log(d["embed"][:, 0]))
            log(f"  {tag} done")
    mark_done("DM")

def stage_timecourse():
    if stage_done("TC"): return
    log("timecourse")
    specs = [dict(init=i, split=s) for s in SPLITS[:TC_S] for i in range(3200, 3200 + TC_I)]
    run_specs("TC", specs, TC_STEPS, TC_REC, "attn", keys=("embed", "path", "win", "unembed"))
    mark_done("TC")

# -------------------------------------------------------------------------------------
def heldout_r2(X, y, seed=0):
    n = len(y); idx = np.random.default_rng(seed).permutation(n); te = idx[: n // 5]; tr = idx[n // 5:]
    Xc = X - X[tr].mean(0); yy = y - y[tr].mean(); best = -9
    for al in (1, 10, 100, 1000, 10000):
        A = Xc[tr]; w = A.T @ np.linalg.solve(A @ A.T + al * np.eye(len(tr)), yy[tr])
        best = max(best, 1 - ((yy[te] - Xc[te] @ w) ** 2).sum() / ((yy[te] - yy[te].mean()) ** 2).sum())
    return float(best)

def analysis():
    log("analysis (pre-registered v2)")
    from scipy.stats import spearmanr
    res = dict(prereg2_sha256=PREREG2_HASH, v1_prereg_sha256=C.PREREG_HASH)
    Vt = vote_matrix("TRUE"); rel = {c: reliability(c) for c in CONDS if stage_done(f"V_{c}")}
    res["reliability"] = rel
    for c, H in [("SHUF", "H1"), ("SHIFT", "H2"), ("SUB", "H3"), ("MLP", "H4")]:
        if not stage_done(f"V_{c}"): res[H] = dict(note="not run"); continue
        Vc = vote_matrix(c); A, p = perm_p_within(Vc, Vt)
        ceil = math.sqrt(max(rel["TRUE"], 1e-9) * max(rel[c], 1e-9))
        r = dict(condition=c, agreement=A, p=p, ceiling=ceil, disattenuated=A / ceil if ceil > 0 else None)
        r["PASS"] = bool(A >= 0.8 * ceil) if H == "H2" else bool(A > 0 and p < 0.01)
        res[H] = r
    ps = [res[h]["p"] for h in ("H1", "H3", "H4") if "p" in res[h]]
    if len(ps) == 3: res["holm_H1_H3_H4"] = C.holm(ps)
    # H5
    h5 = []
    for s_i, s in enumerate(SPLITS[:DM_SPLITS]):
        f1, f2 = OUT / f"DM_{s}_3100.npz", OUT / f"DM_{s}_3101.npz"
        if not (f1.exists() and f2.exists()): continue
        kstar = int(np.argmax(Vt[s_i])); infl, r2 = [], []
        for f in (f1, f2):
            d = np.load(f); X = d["masks"].astype(float); y = d["z"][:, kstar]
            infl.append(ridge_cv(X, y)); r2.append(heldout_r2(X, y))
        rho = spearmanr(infl[0], infl[1])[0]
        rng = np.random.default_rng(5); null = [spearmanr(infl[0], rng.permutation(infl[1]))[0] for _ in range(2000 if not SMOKE else 50)]
        p = (np.sum(np.array(null) >= rho) + 1) / (len(null) + 1)
        h5.append(dict(split=s, kstar=kstar + 1, rho=float(rho), p=float(p), heldout_r2=r2, PASS=bool(rho > 0 and p < 0.01)))
    res["H5"] = dict(splits=h5, PASS=bool(sum(x["PASS"] for x in h5) >= 3) if len(h5) == DM_SPLITS else None)
    # timecourse (descriptive)
    if stage_done("TC"):
        fs = sorted(f for f in (OUT / "TC").glob("chunk_*.npz") if not f.name.endswith(".tmp.npz"))
        ds = [np.load(f) for f in fs]; idx = np.concatenate([x["idx"] for x in ds]); steps = list(ds[0]["steps"])
        spl = np.array([SPLITS[i // TC_I] for i in idx])
        tc = {}
        for q in ("embed", "path", "win", "unembed"):
            Q = np.concatenate([x[q] for x in ds])                     # [runs, T, 56]
            curve = []
            for t in range(len(steps)):
                Z = zscore_log(Q[:, t]); V = np.stack([Z[spl == s].mean(0) for s in SPLITS[:TC_S]])
                V = V - V.mean(0, keepdims=True); Vref = Vt[:TC_S] - Vt[:TC_S].mean(0, keepdims=True)
                curve.append(within_split_spearman(V, Vref))
            tc[q] = dict(steps=[int(x) for x in steps], agreement_with_true_vote=curve)
        res["timecourse"] = tc
    (OUT / "V2_results.json").write_text(json.dumps(res, indent=1, default=float))
    for H in ("H1", "H2", "H3", "H4"):
        log(" ", H, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[H].items()})
    log("  H5", res["H5"])
    return res

def run_all2():
    (OUT / "PREREG2.txt").write_text(PREREG2 + f"\nsha256: {PREREG2_HASH}\n")
    log("v2 | device", DEV, "| out", OUT, "| prereg2", PREREG2_HASH[:16])
    bs = batch_size_for(); t0 = time.time(); train2([dict(init=9990 + i, split=2000) for i in range(bs)], 10 if not SMOKE else 2, ())
    log(f"  speed: batch {bs}, {(time.time()-t0)/(11 if not SMOKE else 3):.3f} s/step")
    for fn in (stage_votes, stage_datamodels, stage_timecourse):
        if time_left_h() < 0.3: log("time budget reached"); break
        try: fn()
        except Exception as e: log(fn.__name__, "failed:", repr(e))
    try: res = analysis()
    except Exception as e:
        import traceback; log("analysis failed:", traceback.format_exc()); res = {}
    log("ALL DONE"); return res

if __name__ == "__main__":
    run_all2()
