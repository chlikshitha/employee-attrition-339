"""Compatibility entry point for configured model evaluation."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "mlops_pipeline" / "06_model_evaluation.py"), run_name="__main__")
