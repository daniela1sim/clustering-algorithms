# Source references and modernization notes

This portfolio project consolidates three related coursework projects. The original
Git repositories remain unchanged. The code here is **reorganized/refactored**, not
a direct export of all original files, and does not contain Git history.

| Topic | University GitHub source | Included here |
|---|---|---|
| K-means | https://github.com/daniella1sim/softwareproject | Modernized NumPy/Lloyd implementation (`src/python/kmeans.py`), derived from the initial-course concept |
| K-means++ | https://github.com/daniella1sim/hw2_projecto | Reimplemented D² initialization and ID-based CSV merge in Python (`src/python/kmeans_plus_plus.py`), with refactored CPython C extension (`src/c/kmeans_module.c`); optional elbow plotting tool |
| SymNMF | https://github.com/daniella1sim/softproj | Previously cleaned-up C implementation, Python extension and analysis scripts, preserving the method while addressing noted defects |

## Technical behavior changes

- Original `hw2_projecto/kmeans_pp.py` used distances directly when sampling centers;
  the version here uses **squared distances**, as in standard K-means++.
- Original K-means++ C extension used module-level global state and a linked-list
  representation. The replacement uses per-call allocations, input validation, and
  a dense representation. Center ordering is maintained (not reversed).
- The K-means baseline preserves **first K input rows** as the initialization,
  rather than using K-means++.
- The two-file K-means++ workflow preserves the original idea of an ID-based,
  sorted inner join, and now checks duplicate IDs.
- The SymNMF implementation and changes are documented separately in
  [`SYMNMF_ORIGIN.md`](SYMNMF_ORIGIN.md).
- Tests and examples have been added. The numerical outputs may differ from the
  original coursework due to improved initialization and validation.

## Attribution and public release

The project was done by Daniella Simonovsky and Laura Shopov in colaboration
