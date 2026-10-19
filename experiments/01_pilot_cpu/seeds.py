"""Across init seeds: are the frequencies chosen early, and the same with and without weight decay?"""
import json, glob, re, torch
from analyze import load, embed_freq_norms, key_freqs
torch.set_grad_enabled(False)

def top(path, n=3):
    fn = embed_freq_norms(load(path)) ** 2
    v, i = fn.sort(descending=True)
    return [int(k) + 1 for k in i[:n]], (fn / fn.sum()).tolist()

def last_ckpt(d):
    return sorted(glob.glob(f"{d}/ckpt/step*.pt"), key=lambda s: int(re.findall(r"step(\d+)", s)[0]))[-1]

runs = {0: (".", "wd0")} | {s: (f"seeds/s{s}-wd1", f"seeds/s{s}-wd0") for s in (1, 2, 3)}
out = {}
for s, (d1, d0) in runs.items():
    log1 = json.load(open(f"{d1}/log.json"))
    grok = next((l["step"] for l in log1 if l["test_acc"] > 0.9), None)
    final = last_ckpt(d1)
    keys = key_freqs(load(final))
    r = dict(seed=s, grok_step_90=grok, final_test_acc=log1[-1]["test_acc"], final_keys=keys,
             init_top=top(f"{d1}/ckpt/step0.pt")[0])
    for step in (500, 1000):
        for tag, d in (("wd1", d1), ("wd0", d0)):
            p = f"{d}/ckpt/step{step}.pt"
            try:
                t, share = top(p, 4)
            except FileNotFoundError:
                continue
            r[f"{tag}_top4_at_{step}"] = t
            # where do the final keys rank in this early spectrum?
            order = sorted(range(56), key=lambda k: -share[k])
            r[f"{tag}_rank_of_keys_at_{step}"] = [order.index(k - 1) + 1 for k in keys]
    out[s] = r
    print(json.dumps(r))
json.dump(out, open("seeds/summary.json", "w"), indent=1)
