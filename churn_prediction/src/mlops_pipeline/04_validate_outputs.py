"""Validate processed matrices and produce a preprocessing summary."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.mlops_pipeline.validate_outputs import validate_preprocessing_outputs
if __name__=="__main__":
    if not validate_preprocessing_outputs(): raise SystemExit(1)
