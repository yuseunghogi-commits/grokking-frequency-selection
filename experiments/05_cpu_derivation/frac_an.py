import os, sys, glob
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("GROK_OUT", os.path.join(_HERE, "results")); sys.path.insert(0, os.path.join(_ROOT, "experiments", "02_kaggle_v1"))
import numpy as np, gpu_grok as G
from deltaE import D_shared
SPLITS = list(range(7200, 7208)); import json; OUTJ = {}
for frac in (0.15, 0.3, 0.5, 0.7):
    fs=[G.OUT/"frac"/f"f{frac}_s{s}.npz" for s in SPLITS]
    if not all(f.exists() for f in fs): print(frac,'incomplete'); continue
    Z=np.stack([G.zscore_log(np.load(f)["embed60"]) for f in fs])      # [S,12,56]
    V=Z.mean(1); V-=V.mean(0)
    h1=Z[:,0::2].mean(1); h1-=h1.mean(0); h2=Z[:,1::2].mean(1); h2-=h2.mean(0)
    r=G.within_split_spearman(h1,h2); rel=2*r/(1+r)
    D=np.stack([D_shared(s,frac) for s in SPLITS]); Dsd=np.mean(np.exp(D).std(1)/np.exp(D).mean(1)); Dc=D-D.mean(0)
    rho,p=G.perm_p_within(Dc,V,reps=2000)
    # strength of the data vote: spread of split-specific vote relative to per-run spread
    strength=np.mean(np.abs(V))/np.mean(np.abs(Z-Z.mean(1,keepdims=True)))
    OUTJ[str(frac)] = dict(spearman=float(rho), p=float(p), vote_reliability=float(rel), D_relative_spread=float(Dsd), data_vote_strength=float(strength))
    print(f"frac {frac:4.2f}: Spearman(D, vote) {rho:+.3f} (p {p:.4f}) | vote reliability {rel:.2f} | ceiling-relative {rho/np.sqrt(max(rel,1e-9)):.2f} | D relative spread {Dsd:.3f} | data-vote strength {strength:.2f}")

json.dump(OUTJ, open(os.path.join(_HERE, "results", "frac_sweep.json"), "w"), indent=1)
