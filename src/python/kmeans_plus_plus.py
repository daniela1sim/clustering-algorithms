"""K-means++: D² initialization, then Lloyd iterations (Python or C)."""
from __future__ import annotations
import argparse
import numpy as np
from .tools import check_k, load_points


def initialize_plus_plus(points: np.ndarray, k: int, *, seed: int = 1234) -> list[int]:
    """Return distinct row indices using probability proportional to squared distance."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("points must be a nonempty finite 2D array")
    check_k(k, len(x))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(x)))]
    available = np.ones(len(x), dtype=bool)
    available[chosen[0]] = False
    distances = np.sum((x - x[chosen[0]]) ** 2, axis=1)
    for _ in range(1, k):
        distances = np.maximum(distances, 0)
        probabilities = np.where(available, distances, 0.0)
        total = probabilities.sum()
        if total > 0 and np.isfinite(total):
            probabilities = probabilities / total
            idx = int(rng.choice(len(x), p=probabilities))
        else:
            # Identical data / degenerate case: choose another unused row.
            idx = int(rng.choice(np.flatnonzero(available)))
        chosen.append(idx)
        available[idx] = False
        distances = np.minimum(distances, np.sum((x - x[idx]) ** 2, axis=1))
    return chosen


def fit_plus_plus(points: np.ndarray, k: int, *, seed: int = 1234,
                  max_iterations: int = 300, epsilon: float = 0.001,
                  engine: str = "python") -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (centers, nearest-center labels, initial indices)."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("points must be a finite 2D matrix")
    check_k(k, len(x))
    if max_iterations < 1 or epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("max_iterations and epsilon must be positive and finite")
    indices = initialize_plus_plus(x, k, seed=seed)
    if engine == "c":
        try:
            import kmeans_module
        except ImportError as exc:
            raise RuntimeError("C extension unavailable; build with python setup.py build_ext --inplace") from exc
        centers = np.asarray(kmeans_module.fit(max_iterations, epsilon, x.tolist(), indices), dtype=float)
    elif engine == "python":
        centers = x[indices].copy()
        for _ in range(max_iterations):
            d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            updated = centers.copy()
            for i in range(k):
                members = x[labels == i]
                if len(members):
                    updated[i] = members.mean(axis=0)
            converged = np.linalg.norm(updated - centers, axis=1).max() < epsilon
            centers = updated
            if converged:
                break
    else:
        raise ValueError("engine must be 'python' or 'c'")
    labels = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return centers, labels, indices


def load_merged_csv(first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the coursework's ID-based inner merge of two headerless CSVs."""
    import pandas as pd
    a = pd.read_csv(first, header=None)
    b = pd.read_csv(second, header=None)
    if a[0].duplicated().any() or b[0].duplicated().any():
        raise ValueError("IDs must be unique within each CSV")
    merged = pd.merge(a, b, on=0, how="inner").sort_values(by=0)
    if len(merged) < 2:
        raise ValueError("at least two overlapping IDs are required")
    ids = merged[0].to_numpy()
    points = merged.drop(columns=[0]).to_numpy(dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("coordinates must be finite")
    return ids, points


def main() -> None:
    parser = argparse.ArgumentParser(description="K-means++ initialization followed by Lloyd iterations")
    parser.add_argument("k", type=int)
    parser.add_argument("file", nargs="?", help="comma-separated .txt file (without IDs)")
    parser.add_argument("--merge", nargs=2, metavar=("FIRST_CSV", "SECOND_CSV"), help="merge two ID-based CSVs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--engine", choices=("python", "c"), default="python")
    args = parser.parse_args()
    if bool(args.file) == bool(args.merge):
        parser.error("provide exactly one of FILE or --merge FIRST_CSV SECOND_CSV")
    if args.merge:
        ids, points = load_merged_csv(*args.merge)
    else:
        points = load_points(args.file)
        ids = np.arange(len(points))
    centers, labels, initial = fit_plus_plus(points, args.k, seed=args.seed,
        max_iterations=args.iterations, epsilon=args.epsilon, engine=args.engine)
    print("initial IDs: " + ",".join(str(ids[i]) for i in initial))
    for row in centers:
        print(",".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    main()
