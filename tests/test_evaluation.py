import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from evaluate_forecast import evaluate, metrics, prepare


class EvaluationTests(unittest.TestCase):
    def series(self):
        return pd.Series(np.arange(50, dtype=float), index=pd.date_range('2020-01-01', periods=50, freq='h'))

    def test_strict_past_alignment(self):
        series = self.series()
        x_train, x_test, y_train, y_test, ledger = prepare(series, series.index[35], 3)
        self.assertLess(x_train.index.max(), x_test.index.min())
        np.testing.assert_array_equal(x_test['lag_1'], y_test-1)
        np.testing.assert_array_equal(x_test['lag_3'], y_test-3)
        self.assertEqual(ledger['train_rows']+ledger['test_rows']+ledger['excluded_rows'], 50)

    def test_holdout_values_cannot_fit_preprocessing(self):
        a = self.series()
        a.iloc[8] = np.nan
        b = a.copy()
        b.iloc[35:] = 1e9
        report_a, model_a = evaluate(a, a.index[35], 3)
        report_b, model_b = evaluate(b, b.index[35], 3)
        np.testing.assert_array_equal(model_a[0].statistics_, model_b[0].statistics_)
        np.testing.assert_array_equal(model_a[1].mean_, model_b[1].mean_)
        np.testing.assert_array_equal(model_a[2].coef_, model_b[2].coef_)

    def test_metrics_known_values(self):
        result = metrics([1, 3], [2, 1])
        self.assertEqual(result['mae'], 1.5)
        self.assertEqual(result['mse'], 2.5)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2.5))
        self.assertEqual(result['r2'], -1.5)

    def test_constant_target_r2_explicit(self):
        self.assertIsNone(metrics([1, 1], [1, 1])['r2'])

    def test_missing_target_never_imputed(self):
        series = self.series()
        series.iloc[40] = np.nan
        _, _, _, target, ledger = prepare(series, series.index[35], 3)
        self.assertNotIn(series.index[40], target.index)
        self.assertNotIn(series.index[41], target.index)
        self.assertEqual(ledger['excluded_rows'], 5)

    def test_reject_duplicate_irregular_and_reversed_times(self):
        a = self.series()
        for b in (a.iloc[::-1], a.drop(a.index[10]), pd.concat([a, a.iloc[:1]]).sort_index()):
            with self.assertRaises(ValueError): prepare(b, a.index[35], 3)

    def test_reject_nonfinite_and_misaligned_metrics(self):
        with self.assertRaises(ValueError): metrics([1, 2], [1])
        with self.assertRaises(ValueError): metrics([1, 2], [1, np.nan])

    def test_cli_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / 'observations.csv'
            report = root / 'report.json'
            series = self.series()
            pd.DataFrame({'date': series.index, 'CO': series.to_numpy(),
                          'station': 1}).to_csv(csv, index=False)
            command = [sys.executable, str(Path(__file__).resolve().parents[1]/'evaluate_forecast.py'),
                       str(csv), '--station', '1', '--test-start', str(series.index[35]),
                       '--lags', '3', '--output', str(report)]
            subprocess.run(command, check=True, capture_output=True)
            payload = json.loads(report.read_text())
            self.assertEqual(payload['sample_ledger']['test_rows'], 15)
            self.assertEqual(len(payload['input_sha256']), 64)
            self.assertEqual(len(payload['source_sha256']), 64)
            before = report.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, report.read_bytes())


if __name__ == '__main__': unittest.main()
