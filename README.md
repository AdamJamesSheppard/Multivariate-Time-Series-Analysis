# Madrid air-quality time-series experiments

Historical notebook exploration plus an auditable one-step baseline. Original
data are not included here; no fresh Madrid forecasting result is claimed.

## Research-quality audit (2026-10-11)

The pre-repair `TimeSeries.ipynb` is preserved in Git history. Its historical
results need re-evaluation; the current notebook uses the new baseline below:

- Cells22/25 estimate medians and imputation from the complete series before
  the training/test split. This leaks held-out distribution information.
- Feature decisions use complete-series correlations. This is development
  inspection, not untouched evaluation.
- Cell37 reports `100*(1-sum(abs(error))/sum(target))` as accuracy. It is neither
  MSE nor a generic percentage accuracy; it can be negative or undefined.
- Cell10 divides mg/m³ by1000 while its narrative claims conversion to µg/m³.
  The dimensional conversion in that direction is multiplication by1000.
  Source measurement units still need verification before changing the data.
- ADF rejection addresses a specified unit-root null; it does not prove that
  a series has no trend. A correlation coefficient does not establish causal
  influence. Missingness mechanisms and sampling frequency need documentation.
- Local absolute file paths and deprecated DataFrame.append impede reproduction.

## New baseline

`evaluate_forecast.py` uses **only past CO observations** to predict one next
observed CO value. It introduces a fixed Ridge baseline rather than silently
claiming to reproduce the notebook's multivariate random-forest experiment.
Median imputation and scaling are fitted only on the chronological training
portion. Missing targets are never imputed. Model and persistence are scored
on identical eligible rows. MAE, MSE, RMSE and R² have explicit standard semantics.

```sh
python3 -m unittest discover -s tests -v
python3 evaluate_forecast.py measurements.csv --station 28079004 \
  --test-start 2017-03-01 --lags 12 --output fresh-report.json
```

Requires NumPy, pandas and scikit-learn. CSV must contain `date`, `CO`, and,
when selecting a station, `station`. Select a single station; timestamps must
be unique and equally spaced. `--lags 12` means12 sampling intervals, not
necessarily12 hours. No unit conversion, resampling, interpolation or target
repair is inferred. Input and source hashes, runtime versions, sample accounting,
parameters and preprocessing medians are saved. Existing report files are refused.

Test inputs use earlier *observed* test values, so this is rolling one-step
prediction. Autonomous multi-step rollout needs a different experiment. Once
the test results inform method decisions, that set is development data. Before
scientific reporting, verify data provenance/licensing/units, missingness, temporal
resolution, alternative models and an untouched evaluation protocol. The historical
notebook's data access means its old test interval cannot be newly called untouched.

### Method attribution and validation scope

Engineering design: train-only preprocessing, chronological split and a fixed
Ridge/persistence comparison. Rationale and APIs:
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[time-sensitive evaluation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[regression metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html).
The implementation uses a single fixed split, not TimeSeriesSplit cross-validation.
Tests falsify preprocessing leakage by changing every holdout value and asserting
identical fitted preprocessing/model coefficients. Other tests check lag alignment,
sample counts, analytic metric values, missing-target exclusion and timestamp errors.
Synthetic fixtures are software tests, not Madrid research evidence. No claim of
optimal model, real-data improvement, causal interpretation or full replication.

## Entry-point repairs — 15 October 2026
The primary notebook now invokes the chronological, train-only preprocessing utility. Unsupported unit conversions and historical scores have been removed. Real-data re-evaluation remains outstanding. Historical versions remain available in Git history.
