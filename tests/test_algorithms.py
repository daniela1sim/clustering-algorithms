"""Simple numerical and CLI checks (run after building the extension)."""
import subprocess
import sys
import unittest
import numpy as np
import symnmfmodule
from src.python.kmeans import kmeans
from src.python.symnmf_cli import factorize
from src.python.tools import load_points


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0.], [0., 1.], [4., 4.], [4., 5.]])

    def test_gaussian_affinity_symmetric_and_zero_diagonal(self):
        a = np.array(symnmfmodule.sym(self.points.tolist()))
        np.testing.assert_allclose(a, a.T)
        np.testing.assert_array_equal(np.diag(a), 0)
        self.assertAlmostEqual(a[0, 1], np.exp(-0.5), places=12)

    def test_degree_and_normalization(self):
        a = np.array(symnmfmodule.sym(self.points.tolist()))
        d = np.array(symnmfmodule.ddg(self.points.tolist()))
        w = np.array(symnmfmodule.norm(self.points.tolist()))
        np.testing.assert_allclose(np.diag(d), a.sum(axis=1))
        np.testing.assert_allclose(w, np.diag(1 / np.sqrt(np.diag(d))) @ a @ np.diag(1 / np.sqrt(np.diag(d))))

    def test_factorization_output_shape_and_sign(self):
        h = factorize(self.points, 2)
        self.assertEqual(h.shape, (4, 2))
        self.assertTrue(np.isfinite(h).all())
        self.assertTrue((h >= 0).all())

    def test_factorization_does_not_mutate_initial_input(self):
        w = np.array(symnmfmodule.norm(self.points.tolist()))
        h0 = [[0.1, 0.2] for _ in self.points]
        copy = [r[:] for r in h0]
        _ = symnmfmodule.symnmf(h0, w.tolist())
        self.assertEqual(h0, copy)

    def test_invalid_ragged_input(self):
        with self.assertRaises(ValueError):
            symnmfmodule.sym([[1., 2.], [3.]])

    def test_kmeans(self):
        centers, labels = kmeans(self.points, 2)
        self.assertEqual(centers.shape, (2, 2))
        self.assertEqual(labels.shape, (4,))


if __name__ == "__main__":
    unittest.main()
