import numpy as np

def extract_c1_prime(pdb_string: str) -> np.ndarray:
    coords = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM  ") and "C1'" in line:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue
    return np.array(coords)

def align_kabsch(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(predicted) != len(target):
        raise ValueError("Arrays must have the same length for Kabsch alignment.")
    pred_centroid = np.mean(predicted, axis=0)
    targ_centroid = np.mean(target, axis=0)
    pred_centered = predicted - pred_centroid
    targ_centered = target - targ_centroid
    cov = np.dot(pred_centered.T, targ_centered)
    U, S, Vt = np.linalg.svd(cov)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    aligned_predicted = np.dot(pred_centered, R.T) + targ_centroid
    return aligned_predicted

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

def approximate_tm_score(predicted: np.ndarray, target: np.ndarray) -> float:
    L = len(target)
    if L <= 15:
        d0 = 0.5 
    else:
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    diff = np.sum((predicted - target)**2, axis=1)
    di = np.sqrt(diff)
    score_components = 1 / (1 + (di / d0)**2)
    return float(np.sum(score_components) / L)
