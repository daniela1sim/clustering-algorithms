"""Optional elbow-plot demonstration, inspired by the K-means++ coursework bonus."""
from pathlib import Path
import argparse
import numpy as np
from sklearn.cluster import KMeans
from .tools import load_points


def main():
    parser = argparse.ArgumentParser(description="Create inertia-vs-k elbow plot")
    parser.add_argument("file")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--output", default="elbow.png")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = load_points(args.file)
    kmax = min(args.max_k, len(x)-1)
    if kmax < 2:
        parser.error("need at least 3 observations for an elbow plot")
    ks = list(range(1, kmax+1))
    inertia = [KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=1234).fit(x).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(ks, inertia, "o-")
    ax.set(xlabel="Number of clusters (k)", ylabel="Inertia", title="K-means++ elbow curve")
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print("Saved", Path(args.output))


if __name__ == "__main__":
    main()
