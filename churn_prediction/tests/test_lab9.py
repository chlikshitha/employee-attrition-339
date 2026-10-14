"""Unit tests for schema and preprocessing failure detection."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_data", ROOT / "src/mlops_pipeline/02_validate_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data/raw/churn.csv")

    def test_current_data_passes(self):
        self.assertEqual(module.validate(self.data), [])

    def test_missing_column_fails(self):
        self.assertTrue(any("Missing columns" in e for e in module.validate(self.data.drop(columns=["Contract"]))))

    def test_non_numeric_charge_fails(self):
        bad = self.data.copy()
        bad["MonthlyCharges"] = bad["MonthlyCharges"].astype(object)
        bad.loc[bad.index[0], "MonthlyCharges"] = "INVALID_NUMBER"
        self.assertTrue(any("MonthlyCharges" in e for e in module.validate(bad)))

if __name__ == "__main__":
    unittest.main()
