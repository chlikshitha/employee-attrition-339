"""Load the Churn FastAPI app without polluting the shared ``api`` namespace."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys


def _load_churn_app():
    project_root = Path(__file__).resolve().parents[1]
    saved_api_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "api" or name.startswith("api.")
    }
    for name in saved_api_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(project_root))
    try:
        churn_main = importlib.import_module("api.main")
        return churn_main.app
    finally:
        for name in tuple(sys.modules):
            if name == "api" or name.startswith("api."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_api_modules)
        sys.path.remove(str(project_root))


app = _load_churn_app()
