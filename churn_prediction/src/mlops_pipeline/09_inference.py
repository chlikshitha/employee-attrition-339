"""Run production inference on the separate, unlabeled new data file."""
from pathlib import Path
import os, yaml, mlflow, mlflow.sklearn, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[2]
def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",cfg["mlflow"]["tracking_uri"]))
    path=ROOT/cfg["data"]["inference_path"]
    if not path.exists(): raise FileNotFoundError(f"Inference dataset not found: {path}")
    df=pd.read_csv(path); target=cfg["data"]["target_column"]; ids=cfg["data"]["id_column"]
    X=df.drop(columns=[c for c in [target,ids] if c in df.columns])
    train_columns=set(pd.read_csv(ROOT/cfg["data"]["processed_dir"]/"X_train_raw.csv").columns)
    errors=[]
    if set(X.columns)!=train_columns:
        errors.append("Feature mismatch: missing="+str(sorted(train_columns-set(X.columns)))+"; extra="+str(sorted(set(X.columns)-train_columns)))
    validation=ROOT/"artifacts"/"new_data_validation.csv"; validation.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"error":errors}).to_csv(validation,index=False)
    if errors: raise ValueError("; ".join(errors))
    model=mlflow.sklearn.load_model(f"models:/{cfg['registry']['model_name']}@{cfg['lab9']['production_alias']}")
    pred=model.predict(X); prob=model.predict_proba(X)[:,1]
    result=df.copy(); result["Predicted_Churn"]=pred; result["Churn_Probability"]=np.round(prob,6)
    out=ROOT/cfg["paths"]["inference_output"]; out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False)
    result.to_csv(ROOT/"outputs"/"lab7_inference_predictions.csv",index=False)
    print(f"Wrote {len(result)} predictions to {out.relative_to(ROOT)}")
if __name__=="__main__": main()
