"""Make the Churn test app helper importable in any pytest import mode."""
import importlib.util
from pathlib import Path
import sys

_helper_path = Path(__file__).with_name("churn_app.py")
_spec = importlib.util.spec_from_file_location("churn_api_test_support", _helper_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Churn test helper from {_helper_path}")
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
sys.modules.setdefault("churn_app", _helper)
