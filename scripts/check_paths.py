"""
Kaggle 경로 전체 점검 스크립트
노트북 첫 번째 셀에 붙여넣고 실행하세요.
"""

import os
import sys
import glob

# ── 색상 출력 헬퍼 ──────────────────────────────────────────
OK   = "\033[92m[OK]  \033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

def chk(label, path, required=True):
    exists = os.path.exists(path)
    if exists:
        size = ""
        if os.path.isfile(path):
            mb = os.path.getsize(path) / 1e6
            size = f"  ({mb:.1f} MB)"
        elif os.path.isdir(path):
            n = len(os.listdir(path))
            size = f"  ({n} items)"
        print(f"{OK} {label:<45} {path}{size}")
    else:
        tag = FAIL if required else WARN
        print(f"{tag} {label:<45} NOT FOUND → {path}")
    return exists

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ── 경로 상수 (inference_notebook.py 와 동일) ───────────────
INPUT = "/kaggle/input"
COMP  = f"{INPUT}/stanford-rna-3d-folding-2"
DS    = f"{INPUT}"   # 데이터셋은 /kaggle/input/<dataset-name> 으로 마운트됨

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("1. /kaggle/input 루트 구조")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if os.path.exists(INPUT):
    top_dirs = sorted(os.listdir(INPUT))
    print(f"{INFO} /kaggle/input 에 있는 디렉토리 목록:")
    for d in top_dirs:
        full = os.path.join(INPUT, d)
        n = len(os.listdir(full)) if os.path.isdir(full) else "-"
        print(f"       {d}  ({n} items)")
else:
    print(f"{FAIL} /kaggle/input 자체가 없음 — Kaggle 환경인지 확인!")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("2. 대회 데이터 (stanford-rna-3d-folding-2)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chk("대회 루트",         COMP)
chk("test_sequences.csv", f"{COMP}/test_sequences.csv")
chk("PDB_RNA 폴더",       f"{COMP}/PDB_RNA")
chk("MMseqs2 DB (seqres)", f"{COMP}/PDB_RNA/pdb_seqres_NA")
chk("mmCIF 폴더",         f"{COMP}/PDB_RNA/mmcif")
chk("MSA 폴더",           f"{COMP}/MSA")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("3. 우리 코드 데이터셋 (rna-3d-folding-code)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Kaggle 데이터셋은 /kaggle/input/<dataset-slug> 으로 마운트됨
CODE_CANDIDATES = [
    "/kaggle/input/rna-3d-folding-code",
    "/kaggle/working/RNA_3D_Folding",
]
CODE_SRC = None
for c in CODE_CANDIDATES:
    if os.path.exists(c):
        CODE_SRC = c
        break

if CODE_SRC:
    chk("코드 루트",           CODE_SRC)
    chk("inference.py",        f"{CODE_SRC}/inference.py")
    chk("rna_folding 패키지",  f"{CODE_SRC}/rna_folding/__init__.py")
    chk("inference_notebook",  f"{CODE_SRC}/scripts/inference_notebook.py")
else:
    print(f"{FAIL} 코드 데이터셋을 찾지 못했습니다. 확인한 경로:")
    for c in CODE_CANDIDATES:
        print(f"       {c}")
    # 실제 경로 탐색
    found = glob.glob("/kaggle/input/*/inference.py")
    if found:
        print(f"{WARN} inference.py 발견된 실제 경로: {found}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("4. Protenix 관련 데이터셋")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 가중치 파일 — 여러 가능한 경로 탐색
WEIGHT_CANDIDATES = glob.glob(
    "/kaggle/input/**/1599_ema_0.999.pt", recursive=True
)
if WEIGHT_CANDIDATES:
    for w in WEIGHT_CANDIDATES:
        chk("Protenix 가중치 (.pt)", w)
else:
    print(f"{FAIL} Protenix 가중치 (1599_ema_0.999.pt) — 찾을 수 없음")
    print(f"       데이터셋 'protenix-finetuned-rna3db-all-1599' 가 추가됐는지 확인")

# Protenix 소스 코드 repo
RMSA_CANDIDATES = [
    "/kaggle/input/protenix-rmsa-repo",
    "/kaggle/input/protenix-packages",
]
for r in RMSA_CANDIDATES:
    chk(f"Protenix repo ({os.path.basename(r)})", r, required=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("5. 오프라인 패키지 휠 (.whl)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for pkg in ["biopython", "ml-collections"]:
    whl_found = glob.glob(f"/kaggle/input/**/{pkg}*.whl", recursive=True)
    if whl_found:
        for w in whl_found:
            chk(f"{pkg} 휠", w, required=False)
    else:
        print(f"{WARN} {pkg} 휠 없음 (온라인 설치 필요할 수 있음)")

protenix_whls = glob.glob("/kaggle/input/**/*.whl", recursive=True)
if protenix_whls:
    print(f"{INFO} 발견된 모든 .whl 파일:")
    for w in protenix_whls:
        print(f"       {w}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("6. sys.path 및 import 확인")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"{INFO} 현재 sys.path:")
for p in sys.path:
    print(f"       {p}")

print()
for mod in ["numpy", "pandas", "torch", "Bio"]:
    try:
        __import__(mod)
        import importlib
        v = getattr(importlib.import_module(mod), "__version__", "?")
        print(f"{OK} import {mod:<15} (version: {v})")
    except ImportError as e:
        print(f"{FAIL} import {mod:<15} — {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("7. 요약 및 다음 단계 안내")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 실제 경로 자동 감지해서 출력
print(f"{INFO} 자동 감지된 실제 경로 (inference_notebook.py 수정용):\n")

# COMP 실제 경로
comp_guess = COMP if os.path.exists(COMP) else "NOT FOUND"
print(f"  COMP  = \"{comp_guess}\"")

# 코드 경로
code_guess = CODE_SRC or "NOT FOUND — 업로드 후 경로 확인 필요"
print(f"  CODE  = \"{code_guess}\"")

# 가중치 경로
wt_guess = WEIGHT_CANDIDATES[0] if WEIGHT_CANDIDATES else "NOT FOUND"
print(f"  PT_WT = \"{wt_guess}\"")

print(f"""
{INFO} 모든 경로가 OK 이면 아래 셀을 실행하세요:

  import sys, os, shutil
  CODE_SRC = "{code_guess}"
  CODE_DST = "/kaggle/working/RNA_3D_Folding"
  if not os.path.exists(CODE_DST):
      shutil.copytree(CODE_SRC, CODE_DST)
  sys.path.insert(0, CODE_DST)
  exec(open(CODE_DST + "/scripts/inference_notebook.py").read())
""")
