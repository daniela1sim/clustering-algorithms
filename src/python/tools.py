"""Input validation shared by the command-line entry points."""
from pathlib import Path
import numpy as np


def load_points(path: str) -> np.ndarray:
    """Return rectangular N x D finite data from a comma-delimited .txt file."""
    file = Path(path)
    if file.suffix.lower() != ".txt":
        raise ValueError("Expected a .txt file containing comma-separated numeric rows")
    data = np.loadtxt(file, delimiter=",", ndmin=2)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("Input must be a nonempty matrix of finite numeric values")
    return data


def check_k(k: int, count: int) -> None:
    if not 1 <= k < count:
        raise ValueError(f"k must satisfy 1 <= k < N (received k={k}, N={count})")
