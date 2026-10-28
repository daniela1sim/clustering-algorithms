"""Pure Python/NumPy baseline for Lloyd's K-means algorithm.

The first k data points initialize the centers, matching the original coursework
baseline. This is not k-means++, nor does it use scikit-learn's implementation.
"""
from __future__ import annotations
import numpy as np
from .tools import check_k, load_points


def kmeans(points: np.ndarray, k: int, *, max_iterations: int = 300, epsilon: float = 0.001):
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Points must be a finite 2D array")
    check_k(k, len(x))
    if max_iterations < 1 or epsilon <= 0:
        raise ValueError("Iterations and epsilon must be positive")
    centers = x[:k].copy()
    for _ in range(max_iterations):
        squared = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        converged = np.linalg.norm(updated - centers, axis=1).max() < epsilon
        centers = updated
        if converged:
            break
    labels = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return centers, labels


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="K-means with first-k initialization")
    parser.add_argument("k", type=int)
    parser.add_argument("file", help="Comma-separated .txt dataset")
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    centers, _ = kmeans(load_points(args.file), args.k, max_iterations=args.iterations)
    for row in centers:
        print(",".join(f"{value:.4f}" for value in row))


if __name__ == "__main__":
    main()
