"""K-means++ sampling, C/Python equivalence, and ID merge smoke tests."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from src.python.kmeans_plus_plus import initialize_plus_plus, fit_plus_plus, load_merged_csv
from src.python.compare import compare


class TestPlusPlus(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0,0],[0.1,0.2],[-0.1,0.1],[4,4],[4.2,4.1],[3.9,3.8]], dtype=float)

    def test_seed_and_unique_indices(self):
        a = initialize_plus_plus(self.x, 2, seed=1234)
        self.assertEqual(a, initialize_plus_plus(self.x, 2, seed=1234))
        self.assertEqual(len(set(a)), 2)

    def test_degenerate_all_identical(self):
        x = np.ones((6,2))
        a = initialize_plus_plus(x, 3)
        self.assertEqual(len(set(a)), 3)

    def test_two_obvious_clusters(self):
        for engine in ["python", "c"]:
            centers, labels, indices = fit_plus_plus(self.x, 2, engine=engine)
            self.assertEqual(centers.shape, (2,2))
            self.assertEqual(len(indices), 2)
            self.assertEqual(len(np.unique(labels)), 2)

    def test_python_and_c_agree(self):
        a, _, ia = fit_plus_plus(self.x, 2, engine="python")
        b, _, ib = fit_plus_plus(self.x, 2, engine="c")
        self.assertEqual(ia, ib)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_invalid_c_input(self):
        import kmeans_module
        with self.assertRaises(ValueError):
            kmeans_module.fit(3, .001, [[1,2],[3]], [0])
        with self.assertRaises(ValueError):
            kmeans_module.fit(3, .001, [[1,2],[3,4],[5,6]], [0,0])

    def test_merged_csv_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            a=Path(temp)/"a.csv"; b=Path(temp)/"b.csv"
            a.write_text("3,5\n1,1\n2,2\n")
            b.write_text("2,20\n1,10\n3,30\n")
            ids, points = load_merged_csv(str(a), str(b))
            self.assertEqual(ids.tolist(), [1,2,3])
            np.testing.assert_allclose(points, [[1,10],[2,20],[5,30]])

    def test_all_three_methods_comparison(self):
        res = compare(self.x, 2)
        self.assertEqual(set(res), {"kmeans", "kmeans++", "symnmf"})
        self.assertTrue(all(v is None or np.isfinite(v) for v in res.values()))


if __name__ == "__main__":
    unittest.main()
