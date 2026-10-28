"""Build both CPython extensions (SymNMF and K-means++ iterations)."""
import os
from setuptools import Extension, find_packages, setup

common = dict(extra_compile_args=["-O2"] if os.name != "nt" else [],
              libraries=["m"] if os.name != "nt" else [])
setup(name="clustering-algorithms-coursework", version="0.2.0",
      description="Coursework clustering algorithms: K-means, K-means++, SymNMF",
      packages=find_packages(),
      ext_modules=[
          Extension("symnmfmodule", sources=["src/c/symnmf.c", "src/c/symnmfmodule.c"],
                    define_macros=[("SYMNMF_NO_MAIN", "1")], **common),
          Extension("kmeans_module", sources=["src/c/kmeans_module.c"], **common),
      ])
