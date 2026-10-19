"""Exploratory (not pre-registered): theory of the init side.
Claims tested on stored v1 (dev/test factorials, step-500 winners; 64 full runs) and v3 (step-200 winners) data:
  T1  the MLP sees tokens only through the attention OV path, so the OV-path spectrum should beat the raw W_E spectrum;
  T2  a frequency's growth needs input AND output amplitude: log P_in(k) + log P_out(k), with P_out the Fourier power of
      W_U over answer classes, should beat P_in alone (parameter-free; also scanned);
  T3  the simplest unified model, effective input power P_in + gamma*D, should match the z-sum rule (it does not);
  T4  one exact gradient at init (no update) as a predictor;
  T5  the product init side combined with D (equal-weight z-sum) and its precision on final circuits.
Writes results/init_side.json."""
import os, sys, json, glob, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
V1 = os.path.join(ROOT, "experiments", "02_kaggle_v1"); os.environ["GROK_OUT"] = os.path.join(V1, "results")
sys.path.insert(0, V1); sys.path.insert(0, os.path.join(ROOT, "experiments", "07_steering"))
import gpu_grok as G
from steer import D_of, split_perm
torch.set_num_threads(1)
B = G.fourier_basis().double()
def spec(M):
    q = ((B @ M.double()) ** 2).sum(-1); return (q[1:113:2] + q[2:113:2]).numpy()
def sides(seed):
    W = G.init_state(seed); E = W["W_E"][:113]; OV = torch.einsum("hde,hef->df", W["W_V"], W["W_O"])
    return dict(E=spec(E), EOV=spec(E @ OV), U=spec(W["W_U"].T))
Gd = G.full_grid(); ga, gb = G.grid_pairs(); yall = G.labels("add", ga, gb)
def grad_spec(init, split):
    W0 = G.init_state(init); W = {k: v[None].clone().requires_grad_(True) for k, v in W0.items()}
    tr, _ = G.split_indices(split); F.cross_entropy(G.forward(W, Gd[tr][None])[0], yall[tr]).backward()
    return spec(W["W_E"].grad[0, :113])
def load(name):
    if name == "v3":
        I, S = list(range(6000, 6016)), list(range(5000, 5016)); E = np.full((16, 16, 56), np.nan)
        for f in glob.glob(os.path.join(ROOT, "experiments", "04_cpu_v3_formula", "results", "runs", "*.npz")):
            d = np.load(f); s = S.index(int(d["split"]))
            for j, i in enumerate(d["inits"]): E[I.index(int(i)), s] = d["embed200"][j]
    else:
        I, S, E, _, _ = G.factorial_arrays(name, G.WIN_STEP)
    sd = [sides(i) for i in I]; L = {k: np.log(np.stack([s[k] for s in sd])) for k in sd[0]}
    L["D"] = np.log(np.stack([D_of(split_perm(s)[0]) for s in S])); return I, S, E, L
hits = lambda sc, E: int(G.topk_hits_cells(np.broadcast_to(sc, E.shape) if sc.ndim == 2 else sc, np.argsort(-E, -1)[..., :3]).sum())
nrm = lambda x: x / x.mean(-1, keepdims=True)
res = {}
for name in ("B_dev", "B_test", "v3"):
    I, S, E, L = load(name); n = len(I) * len(S); r = dict(of=3 * n, chance=n * 9 / 56)
    target = G.zscore_log(E).mean(1)
    r["T1"] = {k: dict(hits=hits(L[k][:, None], E), rho_init_vote=G.within_split_spearman(L[k], target)) for k in ("E", "EOV", "U")}
    r["T2"] = {str(w): dict(hits=hits((L["EOV"] + w * L["U"])[:, None], E),
                            rho_init_vote=G.within_split_spearman(L["EOV"] + w * L["U"], target)) for w in (0, 0.5, 1.0)}
    Pin, D = np.exp(L["EOV"]), np.exp(L["D"])
    r["T3"] = {str(g): hits(np.log(nrm(Pin)[:, None] + g * nrm(D)[None]) + L["U"][:, None], E) for g in (0, 0.5, 1, 2, 4)}
    r["zsum_OV_D"] = hits(G.zz(L["EOV"])[:, None] + G.zz(L["D"])[None], E)
    r["T5_zsum_OVxU_D"] = hits(G.zz(L["EOV"] + L["U"])[:, None] + G.zz(L["D"])[None], E)
    if name != "B_dev":
        Gs = np.log(np.stack([np.stack([grad_spec(i, s) for s in S]) for i in I]))
        r["T4"] = dict(grad_alone=hits(Gs, E), zOV_plus_zgrad=hits(G.zz(L["EOV"])[:, None] + G.zz(Gs), E))
    res[name] = r; print(name, json.dumps(r, default=float))
runs = json.load(open(os.path.join(V1, "results", "C_chunk_0000.json"))); prec = {"zOV+zD": [], "z(OV*U)+zD": [], "chance": []}
for q in runs:
    s = sides(q["init"]); d = np.log(D_of(split_perm(q["split"])[0])); keys = {k - 1 for k in q["final_keys"]}
    for lab, sc in (("zOV+zD", G.zz(np.log(s["EOV"])) + G.zz(d)), ("z(OV*U)+zD", G.zz(np.log(s["EOV"]) + np.log(s["U"])) + G.zz(d))):
        prec[lab].append(len(keys & set(np.argsort(-sc)[:3].tolist())) / 3)
    prec["chance"].append(len(keys) / 56)
res["final_circuit_top3_precision"] = {k: float(np.mean(v)) for k, v in prec.items()}
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "results", "init_side.json"), "w"), indent=1, default=float)
print(res["final_circuit_top3_precision"])
