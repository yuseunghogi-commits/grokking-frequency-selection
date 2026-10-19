import torch, torch.nn.functional as F, json, time, sys
from model import OneLayer, dataset

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
WD = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "."
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
CK = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
torch.manual_seed(SEED)
import os; os.makedirs(f"{OUT}/ckpt", exist_ok=True)
x, y, tr, te = dataset()
m = OneLayer()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=WD, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1.0))
log = []
t0 = time.time()
for step in range(STEPS + 1):
    logits = m(x[tr])
    loss = F.cross_entropy(logits, y[tr])
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 100 == 0 and step >= 0:
        with torch.no_grad():
            lt = m(x[te])
            tl = F.cross_entropy(lt, y[te]).item()
            ta = (lt.argmax(-1) == y[te]).float().mean().item()
            tra = (logits.argmax(-1) == y[tr]).float().mean().item()
            wn = sum((p_**2).sum() for p_ in m.parameters()).item()
        log.append(dict(step=step, train_loss=loss.item(), test_loss=tl,
                        train_acc=tra, test_acc=ta, wnorm=wn))
        if step % CK == 0:
            print(f"{step:6d} train {loss.item():.4f}/{tra:.3f}  test {tl:.4f}/{ta:.3f}  "
                  f"|W|^2 {wn:.0f}  {time.time()-t0:.0f}s", flush=True)
            torch.save(m.state_dict(), f"{OUT}/ckpt/step{step}.pt")
            json.dump(log, open(f"{OUT}/log.json", "w"))
torch.save(m.state_dict(), f"{OUT}/final.pt")
json.dump(log, open(f"{OUT}/log.json", "w"))
