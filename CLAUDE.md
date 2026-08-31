# Project

Fraud detection pipeline on the Sparkov synthetic card transaction dataset
(Kaggle: `kartik2112/fraud-detection`, ~1.85M rows, ~0.5% fraud rate).

Goal: a portfolio piece demonstrating production ML engineering practice.
Correctness, reproducibility and defensible design decisions matter more than
maximum AUC. Every design decision must be explainable in an interview.

# Hard rules

- All splits are temporal. Random splits are forbidden. Never use
  `train_test_split`. Never shuffle. Leakage is the primary failure mode of
  this project.
- Feature computation lives in `src/features/` and is imported by BOTH
  training and serving code. Feature logic is never duplicated or
  reimplemented.
- Rolling, velocity and aggregate features must use only data strictly
  earlier than the transaction being scored.
- Every feature function gets a unit test, including a window boundary case.
- Type hints everywhere. Docstrings explain WHY, not what.
- No new dependencies without asking.

# Stack

Python 3.12, pandas, numpy, LightGBM, scikit-learn, FastAPI, pytest.
Tooling: ruff, mypy, pyarrow.

Python 3.12, not 3.11: numpy 2.5 requires >=3.12, so a 3.11 target would
silently resolve a different numpy in the container than in development. One
version everywhere beats matching the originally-planned number.

Deployment target is `linux/amd64` while development happens on an Apple
Silicon Mac, so container builds must go through `docker buildx` with an
explicit `--platform linux/amd64`.

# Conventions

- Tests run on synthetic fixtures, never on the real CSVs, so the suite works
  on a fresh clone with no download. Tests that genuinely need the real files
  are marked `@pytest.mark.data` and deselected by default.
- Schema assumptions are asserted at load time, not trusted.
- Split boundaries are printable and serializable so they can be recorded in
  the model card.
