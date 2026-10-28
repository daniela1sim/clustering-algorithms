"""CLI for the native SymNMF implementation and matrix-construction goals."""
import argparse
import numpy as np
import symnmfmodule
from .tools import check_k, load_points


def factorize(points: np.ndarray, k: int, seed: int = 1234) -> np.ndarray:
    check_k(k, len(points))
    w = np.array(symnmfmodule.norm(points.tolist()), dtype=float)
    mean = float(np.mean(w))
    upper = 2 * np.sqrt(mean / k)
    h0 = np.random.default_rng(seed).uniform(0, upper, size=(len(points), k))
    return np.asarray(symnmfmodule.symnmf(h0.tolist(), w.tolist()), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="SymNMF clustering and matrices")
    parser.add_argument("k", type=int, help="Number of clusters, for goal symnmf")
    parser.add_argument("goal", choices=("symnmf", "sym", "ddg", "norm"))
    parser.add_argument("file", help="Comma-separated numeric .txt file")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    points = load_points(args.file)
    check_k(args.k, len(points))
    matrix = factorize(points, args.k, args.seed) if args.goal == "symnmf" else np.asarray(
        getattr(symnmfmodule, args.goal)(points.tolist()), dtype=float)
    for row in matrix:
        print(",".join(f"{value:.4f}" for value in row))


if __name__ == "__main__":
    main()
