"""Create deterministic train/test splits and persist fitted preprocessing."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.mlops_pipeline.preprocessing import run_preprocessing
if __name__=="__main__":
    if not run_preprocessing(): raise SystemExit(1)
