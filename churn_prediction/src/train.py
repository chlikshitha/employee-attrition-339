"""Train the Lab 3 baseline from deterministic processed arrays."""
from pathlib import Path
import yaml, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
ROOT=Path(__file__).resolve().parents[1]
def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); d=ROOT/cfg["data"]["processed_dir"]
    model=RandomForestClassifier(**cfg["model"]); model.fit(np.load(d/"X_train_final.npy"),np.load(d/"y_train.npy"))
    out=ROOT/"models"/"random_forest_baseline.pkl"; out.parent.mkdir(exist_ok=True); joblib.dump(model,out); print(f"Saved baseline to {out}")
if __name__=="__main__": main()
