"""Stop the lifecycle when the configured metric misses its threshold."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.mlops_pipeline.quality_gate import run_quality_gate
if __name__=="__main__":
    if not run_quality_gate(): raise SystemExit(1)
