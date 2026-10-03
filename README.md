# Battery Ageing Analysis — MATLAB to Python

Re-implementation in Python of a battery cycling analysis I originally wrote
in MATLAB: IQR outlier removal, Savitzky-Golay smoothing, SOH and end-of-life.

The original measurement data belongs to a former employer and is not included.
Instead the repo generates **synthetic data with known ground truth**, which
also makes the analysis testable: the validation script checks whether the
pipeline recovers the injected outliers and the true end-of-life cycle.

## Files

| file | what it does |
|---|---|
| `make_data.py` | generates `data.csv` (noisy, with injected outliers) and `truth.json` |
| `analyze.py`   | the pipeline: rolling-IQR outlier removal, Savitzky-Golay, SOH, EOL |
| `validate.py`  | compares the analysis output against `truth.json` and the acceptance criteria |

## Run

```
pip install pandas numpy scipy matplotlib
python make_data.py
python analyze.py
python validate.py
```

## Acceptance criteria

Defined before running, checked by `validate.py`:

- outlier recall >= 0.80
- false-positive rate <= 0.05
- end-of-life estimate within +/- 15 cycles of truth

Current result: PASS (recall 1.00, FP rate 0.006, EOL error 1 cycle).

## Notes on two design decisions

**Rolling IQR, not global.** Capacity falls ~20% across the test, so a global
IQR is dominated by the ageing trend rather than by measurement noise and
flags almost nothing. A 31-cycle moving window asks the right question: is
this point odd compared with its neighbours.

**Interpolation before smoothing.** Savitzky-Golay assumes evenly spaced
samples. Removing outliers leaves gaps, so the gaps are filled by linear
interpolation before filtering; otherwise the filter distorts the curve
exactly where the outliers were.

## Known limitation

Near the ends of the series the moving window is partial, which biases it
towards flagging. In the current run that accounts for the 3 false positives.
