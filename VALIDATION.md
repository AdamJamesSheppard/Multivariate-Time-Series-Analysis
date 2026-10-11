# Repair validation — 2026-10-11

Historical base: `bf14dcc48e5d46516bd6a1e3e0b993a74e05e6bb`.
Original notebook unchanged. Additive exploratory baseline, eight tests:
**8/8 pass** via `python -m unittest discover -s tests -v`.
Local environment: Python3.12.13, NumPy2.5.2, pandas3.0.5, SciPy1.18.0,
scikit-learn1.7.2. Elapsed suite approximately1.01s including CLI execution.
scikit-learn/joblib/threadpoolctl were installed in an isolated temporary target;
the active FEM environment was not changed. CI tests a separate pinned stack;
read its actual result before assuming compatibility with that stack.

First local attempt could not import scikit-learn; no tests executed in that
attempt. After supplying the isolated test dependency, the seven initial tests
passed. The final eight-test suite also exercises CLI report/provenance and
overwrite refusal. This failure history is retained rather than called a clean
first run.

Self-review by implementing assistant; no independent assessor. Synthetic fixtures
test software invariants only. No source Madrid data were available or downloaded;
no real-data accuracy, reproduction of the notebook, or untouched evaluation is
claimed. Historical full-series preprocessing/feature inspection prevents
retrospective claims of an untouched old holdout interval.
