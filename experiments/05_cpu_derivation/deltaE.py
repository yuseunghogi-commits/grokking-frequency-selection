"""Per-run test of the derivation: spectrum of the embedding CHANGE dE_t = E_t - E_0 vs D(k), single models."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault("GROK_OUT", os.path.join(_HERE, "results")); sys.path.insert(0, os.path.join(_ROOT, "experiments", "02_kaggle_v1"))
import numpy as np, torch, torch.nn.functional as F
import gpu_grok as G
from scipy.stats import spearmanr
torch.set_num_threads(2); P = 113; w = 2*np.pi*np.arange(1,57)/P
def D_shared(split, frac=0.3):
    tr,_ = G.split_indices(split, frac); a=(tr//P).numpy(); b=(tr%P).numpy(); c=(a+b)%P
    S=np.zeros((56,P),complex); np.add.at(S.T,c,np.exp(1j*np.outer(w,a)).T); return np.log((np.abs(S)**2).sum(1))
def spec(M):  # M [p,d] -> log power per freq
    q=(G.B.cpu()@M)**2; q=q.sum(-1); return np.log((q[1::2]+q[2::2]).numpy())
def run(inits, split, frac=0.3, steps=(1,2,5,10,20,30,60), sgd=False):
    sts=[G.init_state(i) for i in inits]; M=len(inits)
    W={k: torch.stack([s[k] for s in sts]).requires_grad_(True) for k in sts[0]}
    tr,_=G.split_indices(split, frac); X=G.full_grid()[tr][None].expand(M,-1,-1); a,b=G.grid_pairs(); Y=G.labels("add",a,b)[tr][None].expand(M,-1)
    opt=G.StackedAdamW(W,1e-3,torch.ones(M)); E0=W["W_E"][:,:P].detach().clone(); out={}
    for t in range(max(steps)+1):
        if t in steps: out[t]=[(W["W_E"][m,:P].detach()-E0[m]) for m in range(M)]
        loss=F.cross_entropy(G.forward(W,X).reshape(-1,P),Y.reshape(-1),reduction="none").reshape(M,-1).mean(1).sum(); loss.backward()
        if sgd:
            with torch.no_grad():
                for v in W.values(): v-=0.1*v.grad; v.grad=None
        else: opt.step(G.lr_at(t))
    return out
if __name__=="__main__":
    res={}
    for split in (7000,7001,7002,7003):
        D=D_shared(split); out=run([7100,7101,7102,7103],split)
        for t,dEs in out.items():
            res.setdefault(t,[]).extend(spearmanr(spec(dE),D)[0] for dE in dEs)
    for t in sorted(res): print(f"step {t:3d}: per-run Spearman(spectrum of dE, D) mean {np.mean(res[t]):.3f}  min {np.min(res[t]):.3f}  (16 single runs)")
