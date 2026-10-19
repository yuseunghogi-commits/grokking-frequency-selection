import base64, nbformat as nbf, os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H))
core = open(os.path.join(R, "experiments/02_kaggle_v1/gpu_grok.py"), encoding="utf-8").read()
stage = open(os.path.join(H, "stage_b_gpu.py"), encoding="utf-8").read()
npz = base64.b64encode(open(os.path.join(H, "splits.npz"), "rb").read()).decode()
info = open(os.path.join(H, "splits_info.json"), encoding="utf-8").read()
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nb.cells = [
 nbf.v4.new_markdown_cell("""# Grokking v5 stage B: does steering reach the final circuit?

**실행 방법 (Kaggle)**: Settings → Accelerator → **GPU T4 x2** → 오른쪽 위 **Save Version → Save & Run All (Commit)** → Save.
끝나면 Output 탭에서 `v5_out.zip`을 받아 Claude에게 주세요. 예상 시간 약 1시간.

사전 등록: PREREG5 (sha256 8f973240…, GitHub에 실행 전 커밋). 이 노트북은 run5.py stage B와 같은 설정으로 남은 28개 실행만 돌립니다."""),
 nbf.v4.new_code_cell("import os; os.environ['GROK_OUT'] = '/kaggle/working/v5_out' if os.path.exists('/kaggle') else 'v5_out'\n!nvidia-smi"),
 nbf.v4.new_code_cell("%%writefile gpu_grok.py\n" + core),
 nbf.v4.new_code_cell("%%writefile stage_b_gpu.py\n" + stage),
 nbf.v4.new_code_cell("%%writefile splits_info.json\n" + info),
 nbf.v4.new_code_cell(f"import base64\nopen('splits.npz','wb').write(base64.b64decode('{npz}'))\nprint('splits.npz written')"),
 nbf.v4.new_code_cell("exec(open('stage_b_gpu.py').read())"),
 nbf.v4.new_code_cell("import shutil; shutil.make_archive(str(G.OUT), 'zip', str(G.OUT)); print('zip:', str(G.OUT) + '.zip')"),
]
nbf.write(nb, os.path.join(H, "grokking_v5_stageB.ipynb")); print("written")
