# ==============================================================================
# Kaggle Stanford RNA 3D Folding Part 2 - Interference Notebook Wrapper
# ==============================================================================
# 
# Instructions for Kaggle:
# 1. Upload this structure to a Kaggle dataset or clone your repo in the Kaggle /kaggle/working dir.
# 2. Add datasets: `rna-msa-db`, `hf-rna-foundation-models`, `rna-prediction-engines` etc.
# 3. Paste this cell to run the offline requirements and start inference.

import os
import sys
import subprocess

# 1. Setup offline dependencies (Wheels should be attached in a dataset)
WHEELS_DIR = "/kaggle/input/rna-prediction-engines/wheels"
if os.path.exists(WHEELS_DIR):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--find-links=" + WHEELS_DIR, "torch", "openfold", "einops"], check=False)

# Add your repository to the path if not using a Python module layout directly
REPO_PATH = "/kaggle/working/RNA_3D_Folding"
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

import time
import pandas as pd
from inference import RNAFoldingPipeline

def run_kaggle_pipeline():
    print("Initializing RNA 3D Folding Enchanced Pipeline...")
    
    config = {
        "pdb_db_path": "/kaggle/input/rna-pdb-20250529/pdb_index",
        "protenix_weights": "/kaggle/input/rna-prediction-engines/protenix_w",
        "drfold2_weights": "/kaggle/input/rna-prediction-engines/drfold2_w",
        "aido_weights": "/kaggle/input/hf-rna-foundation-models/aido-rna-650m",
        "mmseqs_bin": "/opt/conda/bin/mmseqs", # Assuming installed in Kaggle environment or uploaded
        "alpha_diversity": 0.5
    }
    
    pipeline = RNAFoldingPipeline(config)
    
    csv_path = '/kaggle/input/stanford-rna-3d-folding-part-2/test.csv'
    if not os.path.exists(csv_path):
        print(f"Warning: test.csv not found at {csv_path}. Running mock fallback.")
        test_df = pd.DataFrame([{"target_id": "mock_id", "sequence": "ACGUACGUACGU"}])
    else:
        test_df = pd.read_csv(csv_path)
    
    total_targets = len(test_df)
    submission_rows = []
    
    start_time = time.time()
    
    for idx, row in test_df.iterrows():
        print(f"[{idx+1}/{total_targets}] Processing target {row.target_id} (len={len(row.sequence)})")
        
        budget = pipeline.check_time_budget(start_time, idx, total_targets, row.sequence)
        best_5_coords = pipeline.process_target(row.target_id, row.sequence, budget)
        
        target_rows = pipeline.serialize_for_submission(row.target_id, best_5_coords)
        submission_rows.extend(target_rows)
        
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv('submission.csv', index=False)
    print("Done! submission.csv saved.")

if __name__ == "__main__":
    run_kaggle_pipeline()
