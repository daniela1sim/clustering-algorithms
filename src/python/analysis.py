"""Compare SymNMF and K-means using the silhouette coefficient."""
import argparse
import numpy as np
from sklearn.metrics import silhouette_score
from .kmeans import kmeans
from .symnmf_cli import factorize
from .tools import check_k, load_points


def evaluate(points: np.ndarray, k: int, seed: int = 1234) -> dict[str, float]:
    check_k(k, len(points))
    if k < 2:
        raise ValueError("Silhouette score requires at least two clusters")
    _, kmeans_labels = kmeans(points, k)
    h = factorize(points, k, seed=seed)
    nmf_labels = h.argmax(axis=1)
    result = {}
    for name, labels in (("nmf", nmf_labels), ("kmeans", kmeans_labels)):
        num_labels = len(np.unique(labels))
        if num_labels < 2 or num_labels >= len(points):
            raise ValueError(f"Cannot compute silhouette for {name}: {num_labels} distinct clusters")
        result[name] = float(silhouette_score(points, labels))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SymNMF against K-means")
    parser.add_argument("k", type=int)
    parser.add_argument("file", help="Comma-separated numeric .txt file")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    for name, score in evaluate(load_points(args.file), args.k, args.seed).items():
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()
