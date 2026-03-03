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
        if os.path.isfile(path):
            size = f"  ({os.path.getsize(path)/1e6:.1f} MB)"
        else:
            size = f"  ({len(os.listdir(path))} items)"
        print(f"{OK} {label:<45} {path}{size}")
    else:
        tag = FAIL if required else WARN
        print(f"{tag} {label:<45} NOT FOUND → {path}")
    return exists

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ── 실제 Kaggle 경로 상수 ────────────────────────────────────
# Kaggle에서 데이터셋은 /kaggle/input/{slug}/ 에 마운트됩니다.
# competition data도 /kaggle/input/{competition-slug}/ 구조입니다.
INPUT = "/kaggle/input"

COMP           = f"{INPUT}/stanford-rna-3d-folding-2"
CODE_DATASET   = f"{INPUT}/rna-3d-folding-code"
CODE_SRC       = f"{CODE_DATASET}/RNA_3D_Folding-claude-sweet-maxwell-rteQC"
PROTENIX_WT    = f"{INPUT}/protenix-finetuned-rna3db-all-1599/1599_ema_0.999.pt"
PROTENIX_REPO  = f"{INPUT}/protenix-rmsa-repo/protenix_kaggle"
PKG_DIR        = f"{INPUT}/protenix-packages"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("1. /kaggle/input 루트 구조")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if os.path.exists(INPUT):
    entries = sorted(os.listdir(INPUT))
    print(f"{INFO} {INPUT}/")
    for e in entries:
        full = os.path.join(INPUT, e)
        n = len(os.listdir(full)) if os.path.isdir(full) else "-"
        print(f"       {e}/  ({n} items)")
else:
    print(f"{FAIL} {INPUT} 없음")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("2. 대회 데이터  (stanford-rna-3d-folding-2)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chk("대회 루트",               COMP)
chk("test_sequences.csv",      f"{COMP}/test_sequences.csv")
chk("PDB_RNA 폴더",            f"{COMP}/PDB_RNA")
chk("MMseqs2 DB (pdb_seqres)", f"{COMP}/PDB_RNA/pdb_seqres_NA")
chk("mmCIF 폴더",              f"{COMP}/PDB_RNA/mmcif")
chk("MSA 폴더",                f"{COMP}/MSA")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("3. 우리 코드  (rna-3d-folding-code)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chk("코드 데이터셋 루트",       CODE_DATASET)
chk("코드 서브폴더",           CODE_SRC)
chk("inference.py",            f"{CODE_SRC}/inference.py")
chk("rna_folding 패키지",      f"{CODE_SRC}/rna_folding/__init__.py")
chk("scripts/check_paths.py",  f"{CODE_SRC}/scripts/check_paths.py")
chk("scripts/inference_nb.py", f"{CODE_SRC}/scripts/inference_notebook.py")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("4. Protenix 가중치  (protenix-finetuned-rna3db-all-1599)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chk("Protenix 가중치 (.pt)",   PROTENIX_WT)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("5. Protenix 소스  (protenix-rmsa-repo)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chk("Protenix 소스 repo",      PROTENIX_REPO, required=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("6. 오프라인 패키지  (protenix-packages)")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chk("패키지 폴더",             PKG_DIR,  required=False)

all_whls = glob.glob(f"{PKG_DIR}/**/*.whl", recursive=True)
if all_whls:
    print(f"{INFO} 발견된 .whl 파일:")
    for w in sorted(all_whls):
        print(f"       {w}")
else:
    print(f"{WARN} .whl 파일 없음 — 온라인 pip install 이 필요할 수 있음")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("7. working 디렉토리 및 import 확인")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chk("/kaggle/working 폴더",    "/kaggle/working")
CODE_DST = "/kaggle/working/RNA_3D_Folding"
copied = os.path.exists(CODE_DST)
print(f"{'  '+OK if copied else '  '+WARN} 코드 working 복사 여부: {CODE_DST} {'(있음)' if copied else '(아직 없음 — Cell2 실행 전)'}")

print()
for mod in ["numpy", "pandas", "torch", "Bio"]:
    try:
        import importlib
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"{OK} import {mod:<15} version={v}")
    except ImportError as e:
        print(f"{FAIL} import {mod:<15} — {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section("8. 최종 요약 — 실행 셀 복사용")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ok_code    = os.path.exists(CODE_SRC)
ok_weights = os.path.exists(PROTENIX_WT)
ok_comp    = os.path.exists(f"{COMP}/test_sequences.csv")

print(f"  대회 데이터    : {'OK' if ok_comp    else 'MISSING'}")
print(f"  코드 데이터셋  : {'OK' if ok_code    else 'MISSING'}")
print(f"  Protenix 가중치: {'OK' if ok_weights else 'MISSING'}")

if ok_code:
    print(f"""
{INFO} 모든 경로 확인 후 아래 셀들을 순서대로 실행하세요.

─── Cell 1 (패키지 설치) ──────────────────────────────────
import subprocess, sys, glob
PKG = "{PKG_DIR}"
for whl in glob.glob(f"{{PKG}}/**/*.whl", recursive=True):
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--quiet", whl], check=False)

─── Cell 2 (경로 설정 + 코드 복사) ────────────────────────
import sys, os, shutil
_code_candidates = [
    "{CODE_SRC}",
    "/kaggle/input/datasets/doyhud/rna-3d-folding-code/RNA_3D_Folding-claude-sweet-maxwell-rteQC",
]
CODE_SRC = next((p for p in _code_candidates if os.path.isdir(p)), _code_candidates[0])
CODE_DST = "/kaggle/working/RNA_3D_Folding"
if not os.path.exists(CODE_DST):
    shutil.copytree(CODE_SRC, CODE_DST)
sys.path.insert(0, CODE_DST)
_protenix_candidates = [
    "{PROTENIX_REPO}",
    "/kaggle/input/datasets/zoushuxian/protenix-rmsa-repo/protenix_kaggle",
]
for _p in _protenix_candidates:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break

─── Cell 3 (파이프라인 실행) ──────────────────────────────
exec(open("/kaggle/working/RNA_3D_Folding/scripts/inference_notebook.py").read())
""")
else:
    print(f"\n{FAIL} 코드 데이터셋이 없습니다. Kaggle에 rna-3d-folding-code 데이터셋을 추가했는지 확인하세요.")
    print(f"      기대 경로: {CODE_SRC}")
