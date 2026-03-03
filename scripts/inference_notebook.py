"""
Kaggle notebook entry point for the Stanford RNA 3D Folding Part 2 pipeline.

=============================================================================
SETUP INSTRUCTIONS (paste into Kaggle notebook cells before running):
=============================================================================

Cell 1 – Install offline wheels from attached datasets:
    import subprocess, sys, glob
    DS = "/kaggle/input/datasets"
    for wheel_dir in [
        f"{DS}/ogurtsov/biopython",
        f"{DS}/ogurtsov/ml-collections",
    ]:
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "--no-index", "--find-links=" + wheel_dir,
                        "--quiet", "."],
                       check=False)
    # Install Protenix wheel from zoushuxian/protenix-packages/packages/
    for whl in glob.glob(f"{DS}/zoushuxian/protenix-packages/packages/*.whl"):
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "--quiet", whl], check=False)

Cell 2 – Add repo and Protenix source to path:
    import sys, os, shutil
    DS = "/kaggle/input/datasets"
    # RNA_3D_Folding 코드 데이터셋 → working 복사 (import 경로 확보)
    CODE_SRC = f"{DS}/YOUR_USERNAME/rna-3d-folding-code"   # ← 업로드 후 실제 경로로 변경
    CODE_DST = "/kaggle/working/RNA_3D_Folding"
    if not os.path.exists(CODE_DST):
        shutil.copytree(CODE_SRC, CODE_DST)
    sys.path.insert(0, CODE_DST)
    sys.path.insert(0, f"{DS}/zoushuxian/protenix-rmsa-repo")

Cell 3 – Run this file:
    exec(open("/kaggle/working/RNA_3D_Folding/scripts/inference_notebook.py").read())

=============================================================================
DATASETS TO ATTACH IN KAGGLE (6 total + competition data):
=============================================================================
- stanford-rna-3d-folding-2          (competition data: test_sequences.csv,
                                       PDB_RNA/, MSA/)
- protenix-finetuned-rna3db-all-1599 (Protenix weights: 1599_ema_0.999.pt)
- protenix-packages                  (USalign binary + Protenix wheel)
- protenix-rmsa-repo                 (Protenix source code)
- protenix-mg-packages               (AIDO ModelGenerator)
- biopython                          (offline .whl)
- ml-collections                     (offline .whl)
=============================================================================
"""

import logging
import os
import sys
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_PATH = "/kaggle/working/RNA_3D_Folding"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Protenix source repo (zoushuxian/protenix-rmsa-repo dataset)
PROTENIX_SRC = "/kaggle/input/datasets/zoushuxian/protenix-rmsa-repo"
if os.path.isdir(PROTENIX_SRC) and PROTENIX_SRC not in sys.path:
    sys.path.insert(0, PROTENIX_SRC)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kaggle_runner")

# ---------------------------------------------------------------------------
# Kaggle path constants
# ---------------------------------------------------------------------------
INPUT = "/kaggle/input"
COMP  = f"{INPUT}/competitions/stanford-rna-3d-folding-2"
DS    = f"{INPUT}/datasets"

PATHS = {
    # Competition data
    "test_csv":         f"{COMP}/test_sequences.csv",
    "pdb_db":           f"{COMP}/PDB_RNA/pdb_seqres_NA",
    "cif_dir":          f"{COMP}/PDB_RNA/mmcif",
    "msa_root":         f"{COMP}/MSA",
    # Model weights (zoushuxian/protenix-finetuned-rna3db-all-1599)
    "protenix_weights": f"{DS}/zoushuxian/protenix-finetuned-rna3db-all-1599/1599_ema_0.999.pt",
    "drfold2_weights":  "",   # DRFold2 not available — skipped automatically
    # Output
    "submission":       "/kaggle/working/submission.csv",
}

# Pipeline configuration — mirrors Part 1 winning strategy + competition analysis
CONFIG = {
    # TBM (MMseqs2 with --search-type 3, 1st-place approach)
    "pdb_db_path":       PATHS["pdb_db"],
    "cif_dir":           PATHS["cif_dir"],
    "mmseqs_bin":        "mmseqs",          # pre-installed on Kaggle or via dataset
    "mmseqs_tmp_dir":    "/tmp/mmseqs_rna",
    "tbm_top_k":         5,                 # top-1 through top-5 → one prediction each

    # Protenix (d4t4 RNA-MSA integration)
    "protenix_weights":  PATHS["protenix_weights"],
    "use_rna_msa":       True,              # key improvement over base AlphaFold 3
    "protenix_recycling_steps": 3,
    "protenix_samples":  3,                 # 3 diverse samples per target

    # DRFold2 lightweight fallback
    "drfold2_weights":   PATHS["drfold2_weights"],

    # MSA
    "msa_root":          PATHS["msa_root"],

    # Selection (post-competition agentic tree search weights)
    "w_div":  0.40,
    "w_dist": 0.25,

    # Hardware
    "vram_gb":          16,   # Kaggle T4 (FP16 only, no BF16)
    "time_limit_hours":  8.0,
}


def load_test_data() -> pd.DataFrame:
    """Load competition test CSV, falling back to a mock frame for local dev."""
    if os.path.exists(PATHS["test_csv"]):
        df = pd.read_csv(PATHS["test_csv"])
        # Normalise column name
        if "sequence_id" in df.columns and "target_id" not in df.columns:
            df = df.rename(columns={"sequence_id": "target_id"})
        logger.info("Loaded %d targets from %s", len(df), PATHS["test_csv"])
        return df

    logger.warning("test_sequences.csv not found — using mock data.")
    return pd.DataFrame([
        {"target_id": "mock_R001", "sequence": "ACGUACGUACGUACGU"},
        {"target_id": "mock_R002", "sequence": "GCGCUAGCUAGCUAGC"},
    ])


def run_kaggle_pipeline() -> None:
    """Entry point: initialise pipeline, process all targets, write submission."""
    from inference import RNAFoldingPipeline

    logger.info("=== Stanford RNA 3D Folding Part 2 — Pipeline Start ===")
    logger.info("Config: %s", {k: v for k, v in CONFIG.items() if "weights" not in k})

    pipeline = RNAFoldingPipeline(CONFIG)
    test_df = load_test_data()

    start = time.time()
    submission = pipeline.run(test_df, output_path=PATHS["submission"])

    elapsed = time.time() - start
    logger.info(
        "=== Done in %.1f min — %d submission rows written to %s ===",
        elapsed / 60,
        len(submission),
        PATHS["submission"],
    )
    return submission


# ---------------------------------------------------------------------------
# Run when executed as a script or sourced in a Kaggle cell
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_kaggle_pipeline()
