"""
Kaggle notebook entry point for the Stanford RNA 3D Folding Part 2 pipeline.

=============================================================================
SETUP INSTRUCTIONS (paste into Kaggle notebook cells before running):
=============================================================================

Cell 1 – Install offline wheels (attach dataset containing pre-downloaded wheels):
    import subprocess, sys
    WHEELS = "/kaggle/input/rna-prediction-engines/wheels"
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--no-index", "--find-links=" + WHEELS,
                    "torch", "einops", "ml_collections"],
                   check=False)

Cell 2 – Add repo to path:
    import sys
    sys.path.insert(0, "/kaggle/working/RNA_3D_Folding")

Cell 3 – Run this file:
    exec(open("/kaggle/working/RNA_3D_Folding/scripts/inference_notebook.py").read())

=============================================================================
DATASETS TO ATTACH IN KAGGLE:
=============================================================================
- stanford-rna-3d-folding-part-2   (competition data: test.csv)
- rna-pdb-20250529                 (PDB RNA MMseqs2 database + mmCIF files)
- rna-msa-db                       (precomputed rMSA outputs)
- rna-prediction-engines           (model weights + Python wheel files)
- rna-3d-folding-templates         (organiser-released precomputed templates)
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

PATHS = {
    "test_csv":         f"{INPUT}/stanford-rna-3d-folding-part-2/test_sequences.csv",
    "pdb_db":           f"{INPUT}/rna-pdb-20250529/pdb_seqres_NA",
    "cif_dir":          f"{INPUT}/rna-pdb-20250529/mmcif",
    "msa_root":         f"{INPUT}/rna-msa-db",
    "protenix_weights": f"{INPUT}/rna-prediction-engines/protenix_weights",
    "drfold2_weights":  f"{INPUT}/rna-prediction-engines/drfold2_weights",
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
