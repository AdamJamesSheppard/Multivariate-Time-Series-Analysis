import json
from pathlib import Path
import unittest
class NotebookTests(unittest.TestCase):
    def test_notebook_uses_audited_cli_and_has_no_stale_scores(self):
        notebook = json.loads((Path(__file__).resolve().parents[1]/'TimeSeries.ipynb').read_text())
        cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        code = '\n'.join(''.join(c['source']) for c in cells)
        self.assertIn('evaluate_forecast.py', code)
        self.assertNotIn('fit_transform', code)
        self.assertNotIn('/home/', code)
        for cell in cells:
            compile(''.join(cell['source']), '<notebook>', 'exec')
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])
