"""Compatibility entry point for Lab 9's isolated corruption simulations."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "mlops_pipeline" / "10_failure_injection.py"), run_name="__main__")
