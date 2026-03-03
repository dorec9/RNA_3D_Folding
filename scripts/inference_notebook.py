"""
Kaggle notebook entry point for the Stanford RNA 3D Folding Part 2 pipeline.

=============================================================================
SETUP INSTRUCTIONS (paste into Kaggle notebook cells before running):
=============================================================================

Cell 1 – Install offline wheels (protenix-packages dataset):
    import subprocess, sys, glob
    PKG = "/kaggle/input/protenix-packages"
    for whl in glob.glob(f"{PKG}/**/*.whl", recursive=True):
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "--quiet", whl], check=False)

Cell 2 – Copy repo to /kaggle/working and add paths:
    import sys, os, shutil

    # ── Our code dataset (doyhud/rna-3d-folding-code) ──
    _code_candidates = [
        "/kaggle/input/rna-3d-folding-code/RNA_3D_Folding-claude-sweet-maxwell-rteQC",
        "/kaggle/input/datasets/doyhud/rna-3d-folding-code/RNA_3D_Folding-claude-sweet-maxwell-rteQC",
    ]
    CODE_SRC = next((p for p in _code_candidates if os.path.isdir(p)), _code_candidates[0])
    CODE_DST = "/kaggle/working/RNA_3D_Folding"
    if not os.path.exists(CODE_DST):
        shutil.copytree(CODE_SRC, CODE_DST)
    sys.path.insert(0, CODE_DST)

    # ── Protenix source repo (zoushuxian/protenix-rmsa-repo) ──
    _protenix_candidates = [
        "/kaggle/input/protenix-rmsa-repo/protenix_kaggle",
        "/kaggle/input/datasets/zoushuxian/protenix-rmsa-repo/protenix_kaggle",
    ]
    for _p in _protenix_candidates:
        if os.path.isdir(_p):
            sys.path.insert(0, _p)
            break

Cell 3 – Run the pipeline:
    exec(open("/kaggle/working/RNA_3D_Folding/scripts/inference_notebook.py").read())

=============================================================================
DATASETS TO ATTACH IN KAGGLE (5 datasets + competition data):
=============================================================================
- stanford-rna-3d-folding-2          (competition: test_sequences.csv,
                                       PDB_RNA/, MSA/)
- protenix-finetuned-rna3db-all-1599 (Protenix weights: 1599_ema_0.999.pt)
- protenix-packages                  (biopython, ml-collections, other .whl)
- protenix-rmsa-repo                 (Protenix source: protenix_kaggle/)
- rna-3d-folding-code                (this repository)
=============================================================================
"""

import logging
import os
import sys
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------

# Our repo (copied to working dir by Cell 2)
REPO_PATH = "/kaggle/working/RNA_3D_Folding"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Protenix source repo: try the simple /kaggle/input/{slug} mount first,
# then fall back to the datasets/username/slug layout.
_PROTENIX_SRC_CANDIDATES = [
    "/kaggle/input/protenix-rmsa-repo/protenix_kaggle",
    "/kaggle/input/datasets/zoushuxian/protenix-rmsa-repo/protenix_kaggle",
]
for _p in _PROTENIX_SRC_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

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

# Competition data — mounted directly under /kaggle/input/{slug}/
COMP = f"{INPUT}/stanford-rna-3d-folding-2"

# Protenix weights dataset — slug: protenix-finetuned-rna3db-all-1599
_WEIGHTS_CANDIDATES = [
    f"{INPUT}/protenix-finetuned-rna3db-all-1599/1599_ema_0.999.pt",
    f"{INPUT}/datasets/zoushuxian/protenix-finetuned-rna3db-all-1599/1599_ema_0.999.pt",
]
_PROTENIX_WEIGHTS = next(
    (p for p in _WEIGHTS_CANDIDATES if os.path.isfile(p)),
    _WEIGHTS_CANDIDATES[0],  # default; will log a warning if missing
)

PATHS = {
    # Competition data
    "test_csv":         f"{COMP}/test_sequences.csv",
    "pdb_db":           f"{COMP}/PDB_RNA/pdb_seqres_NA",
    "cif_dir":          f"{COMP}/PDB_RNA/mmcif",
    "msa_root":         f"{COMP}/MSA",
    # Model weights
    "protenix_weights": _PROTENIX_WEIGHTS,
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
    # Verify repo path before importing; emit a helpful error if Cell 2 was skipped.
    if not os.path.isfile(os.path.join(REPO_PATH, "inference.py")):
        raise RuntimeError(
            f"inference.py not found in {REPO_PATH}.\n"
            "Did you run Cell 2 (shutil.copytree + sys.path setup) before Cell 3?"
        )

    from inference import RNAFoldingPipeline

    logger.info("=== Stanford RNA 3D Folding Part 2 — Pipeline Start ===")
    logger.info("REPO_PATH  : %s", REPO_PATH)
    logger.info("COMP       : %s", COMP)
    logger.info("weights    : %s", PATHS["protenix_weights"])
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
