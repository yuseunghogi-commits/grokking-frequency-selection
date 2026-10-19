"""Is the cross-seed frequency bias caused by the fixed training split?
Run seeds 10..25 on a different split (split seed given) and record proxy winners (top-3 at step 500)."""
import json, sys, torch
from collections import Counter
import early
from model import dataset
torch.set_grad_enabled(True)
split = int(sys.argv[1])
early.x, early.y, early.tr, early.te = dataset(seed=split)
out = []
for s in range(10, 26):
    spec = early.train(early.init_state(s), 500, 500)[-1]["spec"]
    keys = [k + 1 for k in sorted(range(56), key=lambda k: -spec[k])[:3]]
    out.append(dict(seed=s, keys=keys)); print(s, keys, flush=True)
cnt = Counter(k for r in out for k in r["keys"])
print("COUNTS", cnt.most_common(10), flush=True)
json.dump(out, open(f"seeds/split{split}.json", "w"))
