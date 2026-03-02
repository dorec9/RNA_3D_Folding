"""
Data loading utilities for the Stanford RNA 3D Folding pipeline.

Handles competition CSV files, A3M multiple sequence alignments, and
pre-computed foundation-model embeddings across both Kaggle and local
development environments.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional


def get_dataset_dir(dataset_name: str) -> str:
    """
    Return the filesystem path for a named dataset.

    On Kaggle (detected via the KAGGLE_KERNEL_RUN_TYPE env var) datasets live
    under /kaggle/input/.  Locally they are expected under a sibling `data/`
    directory relative to this file's parent package.

    Args:
        dataset_name: Dataset folder name, e.g. 'stanford-rna-3d-folding-part-2'.

    Returns:
        Absolute path to the dataset directory.
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""):
        return os.path.join("/kaggle/input", dataset_name)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "data", dataset_name)


def load_competition_data(csv_path: str, is_train: bool = False) -> pd.DataFrame:
    """
    Load competition data from a CSV file (train.csv or test.csv).

    Expected columns: `target_id`, `sequence`.  Additional columns are
    preserved unchanged.

    Args:
        csv_path: Absolute path to the CSV file.
        is_train: When True, additional filtering hooks can be applied (e.g.
                  temporal cutoff for data-leakage prevention).

    Returns:
        DataFrame with at least `target_id` and `sequence` columns.

    Raises:
        FileNotFoundError: If the CSV does not exist at `csv_path`.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Competition CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalise column names: the competition uses both 'sequence_id' and 'target_id'
    if "sequence_id" in df.columns and "target_id" not in df.columns:
        df = df.rename(columns={"sequence_id": "target_id"})

    if is_train:
        # Placeholder for training-set-specific pre-processing (e.g. date filtering)
        pass

    return df


def load_msa(a3m_path: str) -> str:
    """
    Read an A3M-formatted multiple sequence alignment from disk.

    Args:
        a3m_path: Path to the .a3m file.  The file is allowed to be absent
                  (returns an empty string so callers can fall back to
                  single-sequence mode).

    Returns:
        Raw A3M file contents, or '' if the file does not exist.
    """
    if not os.path.exists(a3m_path):
        return ""

    with open(a3m_path, "r") as fh:
        return fh.read()


def find_msa_path(target_id: str, msa_root: str) -> Optional[str]:
    """
    Search common MSA directory layouts for a target's A3M file.

    Tries several naming conventions used by the competition datasets:
    ``<target_id>.a3m``, ``<target_id>/hhblits.a3m``,
    ``<target_id>/rnacentral.a3m``, etc.

    Args:
        target_id: Competition target identifier.
        msa_root: Root directory under which MSAs are stored.

    Returns:
        Absolute path to the A3M file, or None if not found.
    """
    candidates = [
        os.path.join(msa_root, f"{target_id}.a3m"),
        os.path.join(msa_root, target_id, "hhblits.a3m"),
        os.path.join(msa_root, target_id, "rnacentral.a3m"),
        os.path.join(msa_root, target_id, "rfam.a3m"),
        os.path.join(msa_root, target_id, "combined.a3m"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_foundation_embeddings(target_id: str, emb_dir: str) -> np.ndarray:
    """
    Load pre-computed foundation-model embeddings for a target.

    Searches for `<target_id>.npy` then `<target_id>.npz` under `emb_dir`.

    Args:
        target_id: Competition target identifier.
        emb_dir: Directory containing embedding files.

    Returns:
        2-D numpy array of shape (seq_len, embedding_dim).

    Raises:
        FileNotFoundError: If neither .npy nor .npz is found.
    """
    npy_path = os.path.join(emb_dir, f"{target_id}.npy")
    npz_path = os.path.join(emb_dir, f"{target_id}.npz")

    if os.path.exists(npy_path):
        return np.load(npy_path)

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        keys = list(data.keys())
        key = "embedding" if "embedding" in keys else keys[0]
        return data[key]

    raise FileNotFoundError(
        f"No embeddings found for target '{target_id}' in '{emb_dir}'"
    )
