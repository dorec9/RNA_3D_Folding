import unittest
import numpy as np
from rna_folding.coordinate_utils import align_kabsch, calculate_rmsd, approximate_tm_score

class TestCoordinateUtils(unittest.TestCase):
    def setUp(self):
        # Create a synthetic right triangle point cloud
        self.base_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
    def test_kabsch_and_rmsd_translation_rotation(self):
        # 90 degree rotation around Z, and translation by [10, -5, 3]
        theta = np.pi / 2
        rot_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        target_coords = self.base_coords.copy()
        rotated_translated_coords = np.dot(target_coords, rot_z.T) + np.array([10.0, -5.0, 3.0])
        
        # Predicted is the transformed one
        aligned = align_kabsch(predicted=target_coords, target=rotated_translated_coords)
        
        # RMSD should be near 0
        rmsd = calculate_rmsd(aligned, rotated_translated_coords)
        self.assertLess(rmsd, 1e-5, "RMSD after Kabsch should be near 0 for identical rotated shapes")
        
    def test_approximate_tm_score_perfect(self):
        # Perfect match after alignment
        aligned = self.base_coords.copy()
        score = approximate_tm_score(aligned, self.base_coords)
        self.assertAlmostEqual(score, 1.0, places=5)
        
    def test_approximate_tm_score_mismatch(self):
        # Huge mismatch
        mismatched = self.base_coords + np.array([100.0, 100.0, 100.0])
        score = approximate_tm_score(mismatched, self.base_coords)
        self.assertLess(score, 0.1, "TM-score should be low for mismatched coordinates")

if __name__ == '__main__':
    unittest.main()
