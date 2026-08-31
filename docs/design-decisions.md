# Design decisions

Decisions in this project that were genuine judgement calls, with the
alternatives that were rejected and why. Anything obvious is omitted.

## Splitting

### The validation set is carved by row count, not by calendar span

"The last 15%" is ambiguous: 15% of the *time range* and 15% of the *rows* are
different sets whenever transaction volume varies over the year.

Row count was chosen because the validation set's usefulness is governed by how
many **fraud cases** it contains, not how many days it covers. At a ~0.5% base
rate a 195k-row validation set holds roughly 1,100 positives; letting that
number drift with seasonality would make model selection noisier for no benefit.

*Rejected:* a fixed calendar window (e.g. the last 60 days), which is easier to
describe but ties the positive count to the season the window lands in.

### The cutoff snaps to midnight

`snap="day"` rounds the cutoff down to the start of the day. Measured on a synthetic
frame with the real dataset's row counts and date ranges, this moves the
validation share from 15.00% to 15.02% — a negligible cost — and buys a boundary that is stated exactly in the model card
and that never splits a calendar day between two sets.

*Rejected:* the exact row-quantile timestamp (`snap="none"`, still available),
which is marginally more precise but produces boundaries like
`2020-04-02 07:43:11` that are awkward to reason about in day-level reporting.

### Rows sharing the cutoff timestamp all go to the later split

Transactions can share a timestamp to the second. If ties were allowed to fall
on both sides, the same instant would appear in two splits — a small leak, but
exactly the kind that is invisible in aggregate metrics. The rule is that a row
belongs to the later split when `timestamp >= cutoff`, which makes the sets
disjoint by construction. This is the window-boundary case covered by
`test_duplicate_timestamps_at_cutoff_stay_in_one_split`.

### No embargo gap between train and validation

Some time-series pipelines insert a purge/embargo gap at the boundary. This one
does not, deliberately.

An embargo protects against *forward label leakage*: labels that are only
determined some time after the event, so a training row near the boundary can
encode information from the evaluation period. Here the fraud label is known at
transaction time, so that mechanism does not apply.

The reverse direction — a validation transaction whose 30-day rolling window
reaches back into the training period — is not leakage. It is precisely what
production looks like, where the model scores today's transaction using the
customer's real history. Inserting a gap would make validation *pessimistic*
relative to serving.

This is the decision most likely to be challenged, so: the argument for adding a
gap would be that features fitted on training data (target encodings, aggregate
statistics) bleed across the boundary. That is a real risk, and it is handled by
fitting every such statistic on the training split only — not by discarding data.

### The train/test boundary falls mid-day, and is left alone

The published files split at roughly `2020-06-21 12:13`. That boundary is taken
as given rather than tidied to midnight, because moving rows across it would
mean no longer evaluating on the split the dataset publishes.

The consequence matters for feature work: **any calendar-day aggregate for
2020-06-21 straddles train and test.** Day-level features must therefore be
computed causally per scored row (expanding over that row's own history), never
as a whole-day `groupby`. This is a direct application of the hard rule that
aggregate features use only strictly earlier data.

### Cards appear in more than one split

The same `cc_num` shows up in train, validation and test. This is correct for
the problem being solved: the model scores transactions from customers it has
already seen, and per-card history is the main source of signal.

*Rejected:* splitting by card (a "cold start" split). That answers a different
question — how the model performs on customers with no history — which is not
the deployment scenario here. Worth stating explicitly, because entity leakage
is a real concern in other problems and its absence here is a choice, not an
oversight.

### The split object validates itself on construction

`TemporalSplit.__post_init__` raises if any split is empty, if a bounds record
disagrees with its frame, or if two splits touch or overlap in time.

Putting the check in the constructor rather than in a separate `validate()`
means no caller can forget it, and nothing downstream needs to re-derive whether
the split is sound. The cost is a dataclass that does work in `__post_init__`,
which is unusual; the benefit is that an invalid split cannot exist.

## Loading

### Identifiers are strings

`cc_num` as an integer invites arithmetic that means nothing and, as a float,
silently loses its low digits. `zip` as an integer loses its leading zero, which
corrupts every New England ZIP in the file. Both are read as `string`.

### The publisher's index column is dropped

`Unnamed: 0` is a per-file row counter. It is monotonic in transaction time, so
if it ever reached a model it would act as a proxy for the timestamp — leaking
the exact ordering the temporal split exists to protect. It is excluded at read
time rather than dropped later, so it never enters a frame at all.

### Date formats are explicit

`pd.to_datetime` inference is slow over 1.85M rows and can settle on the wrong
day/month order without complaining. Both date columns declare a format and fail
loudly if the file does not match.

### The two time columns are cross-checked by ordering, not by value

The file carries both `trans_date_trans_time` and `unix_time`. Asserting a fixed
offset between them would bake in a timezone assumption that may not hold.
Instead the loader asserts that the two agree on **ordering**, which is
timezone-agnostic and still catches a misparsed timestamp immediately, since a
transposed day/month scrambles the order.

### Unexpected columns are an error, not a warning

A new column means the publisher changed the file. Ignoring it quietly would let
the schema drift away from what the feature code was written against.

## Testing

### The suite never reads the real dataset

Tests run on synthetic frames carrying the real schema. This keeps `make test`
working on a fresh clone with no 150MB download, and — more usefully — lets every
edge case be constructed directly rather than hunted for in real data. Tests that
genuinely need the published files are marked `@pytest.mark.data` and deselected
by default.

### Lint rules encode the project's own rules

Ruff's `ANN` (type annotations) and `D` (docstrings) rule sets are enabled, and
`mypy` runs with `disallow_untyped_defs`. Two of this project's stated hard rules
are therefore enforced by tooling rather than by a reviewer noticing.

## Features

### Training and serving run the same function, not the same specification

`build_features` is the only implementation of any feature in this project. The
serving path (`compute_features_for_transaction`) does not recompute windows
incrementally; it concatenates the card's history with the incoming transaction,
calls `build_features`, and returns the last row.

The usual objection is cost: scoring one transaction rebuilds features over its
whole history. Measured, that is **16 ms for a 50-row history and 18 ms for a
2000-row history** - the time is fixed pandas overhead, not the rebuild, so the
approach does not degrade as a card accumulates history. Against that, train/serve
skew becomes structurally impossible instead of a thing to stay vigilant about:
there is no second implementation that could drift.

*Rejected:* a separate streaming feature implementation for serving, which is the
right answer at much higher throughput but buys nothing here except a second
place for the definition of "transactions in the last 24 hours" to live.
`test_serving_path_matches_training_path` would catch drift between them, but not
having two implementations is better than testing that two implementations agree.

### Causality is carried by two pandas arguments

Every history-dependent feature is causal because of exactly two settings, both
covered by dedicated tests:

- `rolling(..., closed="left")` spans `[t - window, t)`. The scored row is
  excluded, and so is any other transaction sharing its exact timestamp.
- `merge_asof(..., allow_exact_matches=False)` looks strictly backwards, so a
  transaction never treats a same-second sibling as its predecessor.

Same-timestamp transactions are the boundary case that matters in practice: a
card can be charged twice in one second, and a window that included those rows
would let a transaction see its own batch. Default settings on both APIs would
have included them.

A useful side effect: because the previous-transaction lookup is strictly
backwards, elapsed time is always positive, so the implied-speed feature needs no
divide-by-zero guard. The strictness of the window is what makes the arithmetic
safe.

The property is also tested end-to-end rather than only per-function:
`test_future_rows_do_not_change_past_features` asserts that features for the
first N rows are identical whether or not the frame also contains everything
that happened afterwards. Any window that reached forwards would fail there.

### Monotone transforms are omitted; ratios are not

There is no `log(amount)` feature, and no `is_night` or `is_weekend` flag. A
gradient-boosted tree is invariant to monotone transforms of a single column and
can split a range directly, so those features add width to the matrix and no
signal. They appear in most tutorials on this dataset as a habit carried over
from linear models.

The mirror image is why `card_amt_zscore_30d` and `card_amt_ratio_to_mean_7d`
*are* included: a tree sees one feature per split and cannot construct
`(amt - mean) / std` from three separate columns. Combinations of features earn
their place; transforms of one do not.

### `hour` is a plain integer, not a sine/cosine pair

Cyclical encoding exists to tell a linear model that 23:00 and 00:00 are
adjacent. A tree isolates a late-night band with two splits on the raw integer,
so the encoding would only obscure the feature.

### Age is measured at the transaction, not today

`age_years` is computed against the transaction timestamp. Computing it against
the current date would silently change every historical row on every rerun,
making a trained model impossible to reproduce.

### Empty windows: counts are zero, averages are missing

A card with no prior activity has genuinely made zero transactions and spent
zero, so `count` and `sum` are filled with 0. Means and standard deviations are
left as NaN because they are undefined, and LightGBM routes missing values
natively. Imputing them - with zero, or with a global average - would invent
history that does not exist and would make "new card" indistinguishable from
"card that happens to average zero".

### `gender` is deliberately excluded

The raw data carries a gender column and it is not used as a feature. Using a
protected attribute in a decisioning model creates disparate-treatment exposure
that no plausible lift on a synthetic dataset would justify. Excluding it is the
decision; leaving it out silently would not be.

### High-cardinality categoricals are deferred, not forgotten

`merchant` (~690 values), `job`, `city` and `state` are not yet features. They
carry real signal, but extracting it means target encoding, and a target encoding
computed over the whole training set is one of the most common leakage bugs in
fraud pipelines. Doing it correctly means encoding each row from strictly earlier
labels only - the same discipline as the rolling features - and that deserves its
own module and its own boundary tests rather than being bolted on.

`category` (14 values) is passed through as a native categorical, which LightGBM
handles without encoding.

### Haversine, not Euclidean, distance

A degree of longitude is about 111 km at the equator but roughly 82 km at the
mid-latitudes this dataset covers. Treating degrees as a flat plane would
understate north-south separation relative to east-west by about a quarter, which
matters for a feature whose entire job is to say whether two points are
implausibly far apart.

## Toolchain

### The project targets Python 3.12, not the originally planned 3.11

numpy 2.5 declares `requires-python >= 3.12`. Targeting 3.11 while developing on
3.12 would therefore have resolved a *different* numpy inside the container than
the one tests ran against - the exact class of environment drift this project is
supposed to demonstrate avoiding. One Python version everywhere was worth more
than matching the number originally written down.

*Alternative:* stay on 3.11 and pin `numpy<2.5`. Workable, but it freezes a
core library backwards to preserve a version choice nothing else depends on.

### The test suite fails on warnings

`filterwarnings = ["error"]`. A deprecation from pandas is a change in behaviour
arriving on a schedule, and it should surface as a red test rather than as
scrollback nobody reads. This immediately caught `"7d"` being deprecated in
favour of `"7D"` for rolling windows.

## Open items

- **Dependency pinning.** `pyproject.toml` currently declares lower bounds only,
  so a clone months from now may resolve different versions. A lockfile
  (`pip-compile`, or `uv lock`) would make builds reproducible; it needs a new
  tool, so it is deferred pending a decision.
- **High-cardinality categorical encoding.** `merchant`, `job`, `city` and
  `state` need a causal target encoder before they can be used; see above.
- **Distinct-merchant counts in a window.** A useful velocity signal, but
  `nunique` over a time window has no vectorised form in pandas, so it needs a
  different approach than the other rolling features.
