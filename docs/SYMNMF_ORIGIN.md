# Origin, authorship, and packaging notes

**Original university repository:** https://github.com/daniella1sim/softproj

This ZIP is a **cleaned-up, reorganized implementation based on the original
university project**, not a byte-for-byte export of its Git history or source
files. The original repository has not been modified.

## Scope

The original project included these components:
`symnmf.c`, `symnmf.h`, `symnmfmodule.c`, `symnmf.py`, `kmeans.py`,
`analysis.py`, `tools.py`, `setup.py`, and `makefile`.

They correspond to the C core and CLI, CPython extension, Python CLIs and
baseline/evaluation code, and build configuration in this ZIP.

## Notable changes compared with the university submission

- Reorganized source files into `src/c/` and `src/python/`.
- Reworked C matrix allocation/ownership with contiguous storage and safer cleanup.
- Corrected the Python module initializer to match `symnmfmodule`.
- Corrected command-line interfaces and more explicit input validation.
- Used an algebraically equivalent `H(H^T H)` computation in the iterative update.
- Added example data, build instructions, tests, and improved error messages.
- Updated the baseline K-means implementation to a NumPy-based version and
  used NumPy's `default_rng` for SymNMF initialization. This may change exact
  numerical outputs compared with the original coursework submission.

No original Git history is included in this ZIP. If migrating to a personal
repository, consider preserving history from the original repository separately.

**Authorship:** Review the original repository's commits and any course/partner
contribution agreements before publishing. The absence of a license in the
original repository does *not* grant permission to redistribute another
collaborator's contributions. Do not add an open-source license without
confirming that you have the necessary rights.

## Review status

This is a refreshed educational implementation rather than a formally verified
numerical library. It is not a replacement for a maintained production-grade
factorization library. Test behavior on your datasets before publication.
