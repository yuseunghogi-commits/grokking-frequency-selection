import nbformat as nbf, os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
rd = lambda p: open(os.path.join(R, p), encoding="utf-8").read()
REGEN = """import json, hashlib, numpy as np
from steer import steer, split_perm
I = json.load(open('splits_info.json')); out = {}; h = hashlib.sha256()
for k in I['targets']:
    base = I['base'][str(k)]
    out[f'k{k}_g1.0'] = steer(base, k - 1, 1.0, rng_seed=k)[0]
    out[f'k{k}_g0.0'] = np.sort(split_perm(base)[0])
    for g in (1.0, 0.0): h.update(np.ascontiguousarray(out[f'k{k}_g{g}'], dtype=np.int64).tobytes())
assert h.hexdigest() == '10a5469341b48f2b7783ecadb2956f464b991e6fcd24a534dedc3535f3596d3a', 'splits differ: ' + h.hexdigest()
np.savez('splits.npz', **out); print('splits regenerated and verified')"""
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nb.cells = [
 nbf.v4.new_markdown_cell("""# Grokking v7: steering in a 2-layer transformer

**실행 방법 (Kaggle)**: Settings → Accelerator → **GPU T4 x2** → 오른쪽 위 **Save Version → Save & Run All (Commit)** → Save.
끝나면 Output 탭에서 `v7_out.zip`을 받아 Claude에게 주세요. 예상 시간 약 2~4시간.

사전 등록: PREREG7 (sha256 318f50ae…, 실행 전 GitHub에 커밋)."""),
 nbf.v4.new_code_cell("import os; os.environ['GROK_OUT'] = '/kaggle/working/v7_out' if os.path.exists('/kaggle') else 'v7_out'\n!nvidia-smi"),
 nbf.v4.new_code_cell("%%writefile gpu_grok.py\n" + rd("experiments/02_kaggle_v1/gpu_grok.py")),
 nbf.v4.new_code_cell("%%writefile twolayer.py\n" + rd("experiments/10_two_layer/twolayer.py")),
 nbf.v4.new_code_cell("%%writefile steer.py\n" + rd("experiments/07_steering/steer.py")),
 nbf.v4.new_code_cell("%%writefile splits_info.json\n" + rd("experiments/07_steering/splits_info.json")),
 nbf.v4.new_code_cell(REGEN),
 nbf.v4.new_code_cell("%%writefile v7_gpu.py\n" + rd("experiments/10_two_layer/v7_gpu.py")),
 nbf.v4.new_code_cell("exec(open('v7_gpu.py').read())"),
 nbf.v4.new_code_cell("import shutil; shutil.make_archive(str(G.OUT), 'zip', str(G.OUT)); print('zip:', str(G.OUT) + '.zip')"),
]
nbf.write(nb, os.path.join(H, "grokking_v7_two_layer.ipynb")); print("written")
