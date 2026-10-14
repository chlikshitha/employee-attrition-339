"""Load the configured raw data and emit a reproducible load report."""
import json
from pathlib import Path
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[2]
def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data_path = ROOT / config["data"]["raw_path"]
    if not data_path.is_file(): raise FileNotFoundError(f"Raw dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    if df.empty: raise ValueError(f"Raw dataset is empty: {data_path}")
    out = ROOT / "artifacts" / "data_load_report.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stage":"data_loading","source_file":str(data_path.relative_to(ROOT)),"rows":len(df),"columns":len(df.columns),"column_names":df.columns.tolist(),"dtypes":df.dtypes.astype(str).to_dict()}, indent=2), encoding="utf-8")
    print(f"Loaded {len(df)} rows; report: {out.relative_to(ROOT)}")
if __name__ == "__main__": main()
