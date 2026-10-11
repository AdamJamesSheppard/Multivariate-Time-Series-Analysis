"""Auditable one-step, observed-history baseline; no held-out-target repair."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare(series, test_start, lags=12):
    """Split one regularly sampled station, with only strictly past target lags.

    Test inputs may contain already observed earlier test targets: this is a
    rolling one-step experiment, not an autonomous multi-step forecast.
    """
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Supply a pandas Series with a DatetimeIndex")
    if series.index.hasnans or not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be finite, unique and sorted")
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 1:
        raise ValueError("Lags must be a positive integer")
    if len(series) < lags + 4:
        raise ValueError("Insufficient samples")
    spacing = np.diff(series.index.asi8)
    if not np.all(spacing == spacing[0]):
        raise ValueError("Timestamps must be equally spaced; declare resampling separately")
    values = pd.to_numeric(series, errors='raise').astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite values are invalid; NaN denotes missing observations")
    features = pd.concat({f'lag_{i}': values.shift(i) for i in range(1, lags+1)}, axis=1)
    # Same evaluation rows for model and persistence; target is never imputed.
    eligible = values.notna() & features['lag_1'].notna()
    eligible.iloc[:lags] = False
    boundary = pd.Timestamp(test_start)
    train = eligible & (values.index < boundary)
    test = eligible & (values.index >= boundary)
    if train.sum() < 3 or test.sum() < 2:
        raise ValueError("Require at least three training and two test rows")
    if features.loc[train].isna().all().any():
        raise ValueError("A training feature is entirely missing")
    return features.loc[train], features.loc[test], values.loc[train], values.loc[test], {
        'total_rows': len(values), 'eligible_rows': int(eligible.sum()),
        'excluded_rows': int((~eligible).sum()), 'train_rows': int(train.sum()),
        'test_rows': int(test.sum()), 'sampling_interval_ns': int(spacing[0]),
        'test_start': boundary.isoformat(), 'train_last_timestamp': values.index[train][-1].isoformat(),
        'test_first_timestamp': values.index[test][0].isoformat(), 'lags': lags,
    }


def metrics(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.ndim != 1 or target.shape != prediction.shape or len(target) < 2:
        raise ValueError("Require aligned vectors with at least two values")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite values")
    mse = float(mean_squared_error(target, prediction))
    return {'mae': float(mean_absolute_error(target, prediction)), 'mse': mse,
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2_score(target, prediction)) if np.ptp(target) > 0 else None}


def evaluate(series, test_start, lags=12):
    x_train, x_test, y_train, y_test, ledger = prepare(series, test_start, lags)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    report = {'status': 'EXPLORATORY_ONE_STEP_BASELINE', 'sample_ledger': ledger,
              'model': metrics(y_test, prediction), 'persistence': metrics(y_test, x_test['lag_1']),
              'task': 'Predict current observation from strictly earlier observations',
              'test_access': 'Development evaluation; never presumed untouched after inspection',
              'imputer_training_medians': model[0].statistics_.tolist(),
              'model_configuration': {'type': 'median -> StandardScaler -> Ridge', 'alpha': 1.0},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}}
    return report, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--test-start', required=True)
    parser.add_argument('--date-column', default='date')
    parser.add_argument('--target', default='CO')
    parser.add_argument('--station')
    parser.add_argument('--station-column', default='station')
    parser.add_argument('--lags', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists; choose a fresh report path')
    raw = args.csv.read_bytes()
    frame = pd.read_csv(args.csv)
    if args.station is not None:
        frame = frame.loc[frame[args.station_column].astype(str) == args.station]
    elif args.station_column in frame and frame[args.station_column].nunique(dropna=False) != 1:
        parser.error('Multiple stations present; select --station')
    dates = pd.to_datetime(frame[args.date_column], errors='raise')
    series = pd.Series(frame[args.target].to_numpy(), index=pd.DatetimeIndex(dates)).sort_index()
    report, _ = evaluate(series, args.test_start, args.lags)
    report['input_sha256'] = hashlib.sha256(raw).hexdigest()
    report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['input_configuration'] = vars(args) | {'csv': str(args.csv), 'output': str(args.output)}
    report['units'] = 'Unchanged input units; no inferred unit conversion'
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__': main()
