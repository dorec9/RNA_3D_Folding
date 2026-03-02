"""
Coordinate utilities for RNA 3D structure analysis.

Provides C1' extraction from mmCIF files, Kabsch alignment, RMSD, and
an approximate TM-score used for diversity scoring in the selection step.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def extract_c1_prime_from_cif(cif_path: str, chain_id: str = None) -> Dict[int, np.ndarray]:
    """
    Extract C1' atom coordinates from an mmCIF file, keyed by residue sequence number.

    Uses a simple line-based parser that handles both PDBx/mmCIF loop_ tables.
    Falls back gracefully when optional columns are absent.

    Args:
        cif_path: Path to the .cif or .cif.gz file.
        chain_id: If provided, restrict extraction to this chain. Otherwise use the
                  first chain encountered.

    Returns:
        Dict mapping residue sequence number (int) → [x, y, z] (np.ndarray, shape (3,)).
    """
    import gzip
    import os

    opener = gzip.open if cif_path.endswith(".gz") else open
    try:
        with opener(cif_path, "rt") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return {}

    # Locate the _atom_site loop and parse column indices
    in_loop = False
    col_map: Dict[str, int] = {}
    col_index = 0
    coords: Dict[int, np.ndarray] = {}
    detected_chain: Optional[str] = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("loop_"):
            in_loop = True
            col_map = {}
            col_index = 0
            i += 1
            continue

        if in_loop and line.startswith("_atom_site."):
            col_name = line.split(".")[1]
            col_map[col_name] = col_index
            col_index += 1
            i += 1
            continue

        if in_loop and col_map and not line.startswith("_"):
            if line == "" or line.startswith("loop_") or line.startswith("#"):
                in_loop = False
                i += 1
                continue

            parts = line.split()
            if len(parts) < len(col_map):
                i += 1
                continue

            atom_id = parts[col_map["label_atom_id"]] if "label_atom_id" in col_map else ""
            if atom_id.strip("\"'") != "C1'":
                i += 1
                continue

            # Determine chain
            asym_id_key = "label_asym_id" if "label_asym_id" in col_map else "auth_asym_id"
            current_chain = parts[col_map[asym_id_key]].strip("\"'") if asym_id_key in col_map else "A"

            if detected_chain is None:
                detected_chain = chain_id if chain_id else current_chain
            if current_chain != detected_chain:
                i += 1
                continue

            try:
                seq_id_key = "label_seq_id" if "label_seq_id" in col_map else "auth_seq_id"
                seq_id = int(parts[col_map[seq_id_key]])
                x = float(parts[col_map["Cartn_x"]])
                y = float(parts[col_map["Cartn_y"]])
                z = float(parts[col_map["Cartn_z"]])
                coords[seq_id] = np.array([x, y, z], dtype=np.float32)
            except (ValueError, KeyError):
                pass

        i += 1

    return coords


def extract_c1_prime_from_pdb(pdb_string: str) -> np.ndarray:
    """
    Extract C1' coordinates from a PDB-format string.

    Returns:
        np.ndarray of shape (N, 3) with coordinates in order of appearance.
    """
    coords = []
    for line in pdb_string.splitlines():
        if (line.startswith("ATOM  ") or line.startswith("HETATM")) and "C1'" in line[12:16]:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def align_kabsch(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Superimpose *predicted* onto *target* using the Kabsch algorithm.

    Handles reflections by checking the determinant of the rotation matrix.

    Args:
        predicted: (N, 3) array of predicted coordinates.
        target: (N, 3) array of target coordinates.

    Returns:
        (N, 3) rotated+translated predicted coordinates aligned to target.
    """
    if len(predicted) != len(target):
        raise ValueError("Arrays must have the same length for Kabsch alignment.")

    pred_c = predicted - predicted.mean(axis=0)
    targ_c = target - target.mean(axis=0)

    cov = pred_c.T @ targ_c
    U, _, Vt = np.linalg.svd(cov)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    aligned = pred_c @ R.T + target.mean(axis=0)
    return aligned


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Kabsch-aligned RMSD between two coordinate sets."""
    if len(coords1) != len(coords2) or len(coords1) == 0:
        return float("inf")
    aligned = align_kabsch(coords1, coords2)
    diff = aligned - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def approximate_tm_score(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Approximate TM-score using the standard d0 formula for RNA.

    The formula uses the length-dependent d0 threshold:
        d0 = 1.24 * (L - 15)^(1/3) - 1.8   for L > 15
        d0 = 0.5                              for L <= 15

    Args:
        predicted: (N, 3) predicted C1' coordinates (already aligned or raw).
        target: (N, 3) target C1' coordinates.

    Returns:
        float in [0, 1], higher is better.
    """
    L = len(target)
    if L == 0:
        return 0.0
    d0 = 1.24 * max(L - 15, 1) ** (1.0 / 3.0) - 1.8 if L > 15 else 0.5
    d0 = max(d0, 0.5)

    aligned = align_kabsch(predicted, target)
    di_sq = np.sum((aligned - target) ** 2, axis=1)
    score = float(np.sum(1.0 / (1.0 + di_sq / d0 ** 2)) / L)
    return score


def build_coordinate_array(
    seq_len: int,
    coord_dict: Dict[int, np.ndarray],
    q_start: int,
    q_end: int,
    t_start: int,
    t_end: int,
    qaln: str,
    taln: str,
) -> np.ndarray:
    """
    Map template coordinates onto the query sequence using the MMseqs2 alignment strings.

    Gap characters ('-') in the alignment are skipped. Unaligned query positions
    are left as NaN and should be filled by the caller.

    Args:
        seq_len: Total length of the query sequence.
        coord_dict: Dict from residue number → [x, y, z] for the template chain.
        q_start: 0-indexed start of aligned region in query (MMseqs2 qstart - 1).
        q_end: 0-indexed exclusive end of aligned region in query.
        t_start: 1-indexed start of aligned region in template (MMseqs2 tstart).
        t_end: 1-indexed end of aligned region in template (inclusive).
        qaln: Alignment string for the query.
        taln: Alignment string for the template.

    Returns:
        np.ndarray of shape (seq_len, 3) with NaN for unaligned positions.
    """
    coords = np.full((seq_len, 3), np.nan, dtype=np.float32)

    q_pos = q_start
    t_pos = t_start  # 1-indexed residue number in template

    for q_char, t_char in zip(qaln, taln):
        if q_char != "-" and t_char != "-":
            # Match state: map template residue t_pos to query position q_pos
            if t_pos in coord_dict and q_pos < seq_len:
                coords[q_pos] = coord_dict[t_pos]
            q_pos += 1
            t_pos += 1
        elif q_char == "-":
            # Insertion in template: advance template only
            t_pos += 1
        else:
            # Deletion in template (gap in template): advance query only
            q_pos += 1

    return coords


def fill_gaps_linear(coords: np.ndarray) -> np.ndarray:
    """
    Fill NaN positions in a coordinate array using linear interpolation between
    neighbouring valid positions. Endpoints are filled by nearest-neighbour.

    Args:
        coords: (N, 3) array potentially containing NaNs.

    Returns:
        (N, 3) array with NaNs replaced.
    """
    filled = coords.copy()
    N = len(filled)
    valid_mask = ~np.isnan(filled[:, 0])

    if not np.any(valid_mask):
        # No valid coordinates at all — return zeros
        filled[:] = 0.0
        return filled

    valid_indices = np.where(valid_mask)[0]

    for i in range(N):
        if valid_mask[i]:
            continue

        before = valid_indices[valid_indices < i]
        after = valid_indices[valid_indices > i]

        if len(before) == 0:
            filled[i] = filled[after[0]]
        elif len(after) == 0:
            filled[i] = filled[before[-1]]
        else:
            i0, i1 = before[-1], after[0]
            t = (i - i0) / (i1 - i0)
            filled[i] = (1 - t) * filled[i0] + t * filled[i1]

    return filled
