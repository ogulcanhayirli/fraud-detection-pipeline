# Fraud Detection Pipeline

![CI](https://github.com/ogulcanhayirli/fraud-detection-pipeline/actions/workflows/ci.yml/badge.svg)

Card-transaction fraud detection on the Sparkov synthetic dataset
([Kaggle: `kartik2112/fraud-detection`](https://www.kaggle.com/datasets/kartik2112/fraud-detection),
~1.85M transactions, ~0.5% fraud).

The objective of this repository is **production ML engineering practice**, not a
leaderboard score. The dataset is synthetic and a high AUC on it means very
little; what transfers to real work is temporal validation that does not lie,
feature code that cannot drift between training and serving, and design
decisions that survive being questioned.

## Status

| Stage | State |
|---|---|
| Data loading + schema validation | Done |
| Temporal splitting | Done |
| Feature engineering (`src/features/`) | Done |
| Model training + threshold selection | Not started |
| FastAPI scoring service | Not started |
| Container build (`linux/amd64`) | Not started |

## Design decisions

The reasoning behind the non-obvious choices, and the alternatives rejected, is
in [`docs/design-decisions.md`](docs/design-decisions.md). The short version:

- **No random splits, anywhere.** The train/test boundary is the publisher's
  own; the validation set is carved off the *end* of the training period by
  time. `TemporalSplit` raises on construction if the boundaries overlap, so an
  invalid split cannot reach a model.
- **The schema is asserted, not trusted.** ZIP codes and card numbers are read
  as strings, date formats are explicit, and unexpected columns are an error.
  The publisher's `Unnamed: 0` index column is dropped because it is monotonic
  in time and would leak the ordering the split exists to protect.
- **Feature logic will have exactly one home.** `src/features/` is imported by
  both the training and serving paths. This is enforced socially by
  `CLAUDE.md` and structurally by there being no second implementation.
- **Tests do not need the dataset.** The suite runs on synthetic frames carrying
  the real schema, so `make test` works on a fresh clone.
- **Training and serving call the same function.** `build_features` is the only
  feature implementation; the serving path passes it a card's history plus the
  incoming transaction and reads the last row, so train/serve skew is
  structurally impossible rather than something to stay vigilant about.
- **Causality is tested, not asserted.** One test checks that features for the
  first N rows are unchanged by appending everything that happened afterwards;
  another checks that same-second transactions never see each other.

## Setup

```bash
make install        # creates .venv and installs the package with dev extras
```

Download the dataset (Kaggle account required) and place both CSVs in
`data/raw/`:

```
data/raw/fraudTrain.csv
data/raw/fraudTest.csv
```

## Usage

```bash
make test           # pytest
make lint           # ruff check + ruff format --check + mypy
```

```python
from pathlib import Path

from src.data.load import load_train_test
from src.data.split import make_temporal_split

train_df, test_df = load_train_test(Path("data/raw"))
split = make_temporal_split(train_df, test_df, val_fraction=0.15)
print(split)
```

Illustrative output (row counts and date ranges match the published dataset;
the fraud counts below come from a synthetic smoke test, not a real run):

```
temporal split (val_fraction=0.15, snap='day', val_cutoff=2020-04-02 00:00:00)
split  start               end                       rows   fraud     rate
--------------------------------------------------------------------------
train  2019-01-01 00:00:18 2020-04-01 23:59:52  1,101,911   6,340  0.575%
val    2020-04-02 00:00:23 2020-06-21 12:13:31    194,764   1,096  0.563%
test   2020-06-21 12:14:40 2020-12-31 23:59:08    555,719   2,124  0.382%
```

`split.to_dict()` returns the same boundaries as JSON, so the exact split a
model was trained under can be recorded in its model card.

## Features

19 features, all of which see only data strictly earlier than the transaction
being scored. Computing them for the full 1.3M-row training period takes ~3s.

| Group | Features |
|---|---|
| Row-local | `amt`, `hour`, `day_of_week`, `customer_age_years`, `home_to_merchant_km`, `city_pop`, `merchant_category` |
| Card velocity (rolling, strictly earlier) | `card_txn_count_1h/24h/7d`, `card_amt_sum_24h`, `card_amt_mean_7d/30d`, `card_amt_std_30d`, `card_amt_zscore_30d`, `card_amt_ratio_to_mean_7d` |
| Previous transaction on the card | `card_seconds_since_prev_txn`, `card_km_from_prev_txn`, `card_kmh_from_prev_txn` |

Training and serving call the same function:

```python
from src.features import build_features, compute_features_for_transaction

# Training: every row at once, each seeing only its own past.
matrix = build_features(split.train)

# Serving: one transaction against that card's history, same code path.
row = compute_features_for_transaction(card_history, incoming_transaction)
```

Scoring one transaction costs ~16 ms and is dominated by fixed pandas overhead
rather than by history length (18 ms against a 2000-row history).

## Layout

```
src/data/       loading, schema validation, temporal splitting
src/features/   feature builders, imported by BOTH training and serving
src/models/     training, evaluation, threshold selection
src/serving/    FastAPI scoring service
tests/          synthetic-fixture test suite
docs/           model card and design decisions
notebooks/      exploration only; no pipeline logic lives here
```

## Conventions

Type hints everywhere, enforced by `mypy` and ruff's `ANN` rules. Docstrings
explain *why*, not *what*, and are required by ruff's `D` rules. Project rules
live in `CLAUDE.md`.

Development happens on Apple Silicon; the deployment target is `linux/amd64`, so
container builds go through `docker buildx --platform linux/amd64`.
