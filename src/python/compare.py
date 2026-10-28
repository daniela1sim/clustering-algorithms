"""Compare baseline K-means, K-means++, and SymNMF on the same data."""
import argparse
import numpy as np
from sklearn.metrics import silhouette_score
from .tools import load_points, check_k
from .kmeans import kmeans
from .kmeans_plus_plus import fit_plus_plus
from .symnmf_cli import factorize


def compare(points: np.ndarray, k: int, seed: int = 1234):
    check_k(k, len(points))
    if k < 2:
        raise ValueError("silhouette requires >= 2 clusters")
    baseline_centers, baseline_labels = kmeans(points, k)
    plus_centers, plus_labels, _ = fit_plus_plus(points, k, seed=seed)
    h = factorize(points, k, seed=seed)
    labels = [("kmeans", baseline_labels), ("kmeans++", plus_labels), ("symnmf", h.argmax(axis=1))]
    results = {}
    for name, lab in labels:
        unique = len(np.unique(lab))
        if 2 <= unique < len(points):
            results[name] = float(silhouette_score(points, lab))
        else:
            results[name] = None
    return results


def main():
    parser = argparse.ArgumentParser(description="Silhouette comparison of 3 clustering methods")
    parser.add_argument("k", type=int)
    parser.add_argument("file")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    for name, score in compare(load_points(args.file), args.k, args.seed).items():
        print(f"{name:9s}: {score:.4f}" if score is not None else f"{name:9s}: unavailable (<2 nonempty clusters)")


if __name__ == "__main__":
    main()
