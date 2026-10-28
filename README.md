# Clustering Algorithms — K-means, K-means++, and SymNMF

**University coursework portfolio project** combining three closely related
unsupervised clustering implementations using **Python, C, and the CPython C API**.
The algorithms are demonstrated independently and can be evaluated together on
the same numerical dataset.

> **Project origin:** Reorganized and updated from three university projects;
> not a byte-for-byte export. The source repositories, changes, and authorship
> considerations are documented in [source notes](docs/SOURCES_AND_CHANGES.md).

## Implemented methods

| Algorithm | Core concept | Implementation |
|---|---|---|
| **K-means** | Lloyd's assignment/update iterations with first-K-centers initialization | Python/NumPy |
| **K-means++** | Probabilistic **D²** seeding, followed by Lloyd iterations | Python/NumPy and optional native C extension |
| **SymNMF** | Build a normalized Gaussian similarity matrix and approximate it by a nonnegative factorization | C numerical core, CPython extension, Python CLI |

K-means++ is **an initialization strategy for K-means**, not a fundamentally
different clustering objective. SymNMF takes a graph-factorization approach to
clustering and can be compared through its induced cluster assignments.

### Mathematical overview

**K-means / K-means++.** Given input points \(x_i\), minimize the total
within-cluster sum of squared distances to centers \(\mu_j\):

\[\sum_i \|x_i-\mu_{c_i}\|_2^2.\]

Standard K-means here initializes centers from the first \(K\) rows.
K-means++ chooses the first center randomly, then samples each subsequent
center with probability proportional to its **squared distance** from the
nearest existing center. Both then use Lloyd updates until convergence.

**SymNMF.** Form an affinity matrix with \(A_{ij}=\exp(-\|x_i-x_j\|^2/2)\)
for \(i\ne j\) and \(A_{ii}=0\), then the diagonal degree matrix
\(D_{ii}=\sum_j A_{ij}\) and the normalized matrix
\(W=D^{-1/2}AD^{-1/2}\). Factorize \(W\approx HH^T\) with \(H\ge0\):

\[\min_{H\ge0}\|W-HH^T\|_F^2.\]

The C implementation uses a multiplicative update with relaxation
\(\beta=0.5\), a stopping threshold of \(10^{-4}\) for squared Frobenius
change, and a maximum of 300 iterations. Cluster assignments are \(\arg\max\)
across the rows of \(H\).

## Layout

```text
clustering-algorithms/
├── README.md
├── Makefile                       # standalone C CLI and build/test helpers
├── setup.py                       # builds both native Python extensions
├── requirements.txt
├── src/
│   ├── c/
│   │   ├── symnmf.c
│   │   ├── symnmf.h
│   │   ├── symnmfmodule.c         # SymNMF CPython bridge
│   │   └── kmeans_module.c       # C-backed Lloyd iterations for K-means++
│   └── python/
│       ├── kmeans.py             # K-means baseline
│       ├── kmeans_plus_plus.py   # K-means++ and two-CSV input support
│       ├── symnmf_cli.py         # SymNMF command-line interface
│       ├── analysis.py           # original SymNMF-vs-K-means comparison
│       ├── compare.py            # comparison of all three methods
│       ├── elbow.py              # optional inertia/elbow graph
│       └── tools.py
├── examples/
│   ├── two_clusters.txt
│   ├── features_a.csv
│   └── features_b.csv
├── tests/
│   ├── test_algorithms.py
│   └── test_plus_plus.py
└── docs/
    ├── SOURCES_AND_CHANGES.md
    └── SYMNMF_ORIGIN.md
```

## Setup

Requires Python 3.9+, NumPy, pandas, scikit-learn, a suitable C compiler,
and Python development headers. Matplotlib is used only for the optional elbow
plot. On Windows, building the native extensions requires a compatible C build
chain such as Microsoft Visual C++ Build Tools. The standalone C executable
is intended for Unix-like systems.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

Run these commands from the repository root. The plain Python K-means variants
do not require native extensions, but SymNMF and the optional K-means++ C engine
do. On Linux/macOS, `make` also builds the standalone C executable.

## Quickstart: single CSV-style text dataset

The included `examples/two_clusters.txt` contains one point per row and
comma-separated coordinates (no header, no ID column).

**K-means** (first K rows initialize centroids):

```bash
python -m src.python.kmeans 2 examples/two_clusters.txt
```

**K-means++** (D² initialization, Python Lloyd iterations):

```bash
python -m src.python.kmeans_plus_plus 2 examples/two_clusters.txt
```

**K-means++**, using native C iterations:

```bash
python -m src.python.kmeans_plus_plus 2 examples/two_clusters.txt --engine c
```

**SymNMF**:

```bash
python -m src.python.symnmf_cli 2 symnmf examples/two_clusters.txt
```

Intermediate SymNMF matrices and standalone C CLI:

```bash
python -m src.python.symnmf_cli 2 sym examples/two_clusters.txt
python -m src.python.symnmf_cli 2 ddg examples/two_clusters.txt
python -m src.python.symnmf_cli 2 norm examples/two_clusters.txt
make
./bin/symnmf norm examples/two_clusters.txt
```

**Compare all three methods** (silhouette scores):

```bash
python -m src.python.compare 2 examples/two_clusters.txt
```

The silhouette coefficient is defined only if the output includes at least
two distinct clusters and fewer clusters than samples. The comparison prints
`unavailable` for degenerate assignments; results are dataset-dependent.

## K-means++ coursework mode: merge two datasets by ID

The original project joined two headerless input CSVs on a shared ID column,
sorted them by ID, and clustered the concatenated feature vectors. The updated
CLI supports that workflow:

```bash
python -m src.python.kmeans_plus_plus 2 --merge examples/features_a.csv examples/features_b.csv --engine c
```

The first output line lists the **chosen original IDs**; subsequent lines
contain the final centroids. This differs from Python 0-based row positions.

## Optional elbow analysis

Create an inertia-versus-k chart using scikit-learn's K-means++ initialization
as an illustrative reference (the chart does not call the custom C extension):

```bash
python -m src.python.elbow examples/two_clusters.txt --max-k 5 --output elbow.png
```

## Tests

```bash
python -m unittest discover -s tests -v
# On Linux/macOS, after dependencies are installed:
make test
```

Includes tests for SymNMF affinity and factorization, K-means output, K-means++
seeding and degenerate cases, agreement between Python and C Lloyd iterations,
and the ID-based CSV merge.

## Engineering choices / limitations

- Intended as an educational project for small/medium datasets, **not a
  production numerical library**. SymNMF uses a dense \(N\times N\) matrix,
  hence \(O(N^2)\) space usage.
- Results depend on initialization. The default seed for SymNMF and K-means++
  is 1234; the baseline uses the first K input points.
- Empty-cluster centers are preserved during K-means iterations. The simple
  K-means loops may be slower than optimized numerical libraries.
- K-means++ here uses the standard **squared** distance probabilities rather
  than the original coursework's unsquared distance sampling.
- Numerical edge cases, large data, Windows builds, and performance require
  more extensive testing before claiming broad reliability.
- No rights or permissions are implied for code contributed by collaborators.
  Check original coursework rules before publishing and preserve attribution.

## Source repositories

- [First K-means project](https://github.com/daniella1sim/softwareproject)
- [K-means++ project](https://github.com/daniella1sim/hw2_projecto)
- [SymNMF project](https://github.com/daniella1sim/softproj)

See [docs/SOURCES_AND_CHANGES.md](docs/SOURCES_AND_CHANGES.md) for details.
