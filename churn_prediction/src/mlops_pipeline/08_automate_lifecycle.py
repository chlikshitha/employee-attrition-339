"""Assign the best eligible registered version the production alias."""
from pathlib import Path
import json, os, yaml, mlflow
from mlflow.tracking import MlflowClient
ROOT=Path(__file__).resolve().parents[2]
def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",cfg["mlflow"]["tracking_uri"]))
    client=MlflowClient(); name=cfg["registry"]["model_name"]; metric=cfg["registry"]["decision_metric"]
    versions=client.search_model_versions(f"name='{name}'"); eligible=[]
    for v in versions:
        run=client.get_run(v.run_id)
        score=run.data.metrics.get(metric)
        if score is not None and run.data.params.get("model_family")=="RandomForestPipeline":
            eligible.append((float(score),int(v.version)))
    if not eligible: raise RuntimeError(f"No registered version has metric {metric}")
    score,version=max(eligible); alias=cfg["lab9"]["production_alias"]; client.set_registered_model_alias(name,alias,str(version))
    report={"model_name":name,"alias":alias,"version":version,"decision_metric":metric,"score":score,"registered_versions_seen":len(versions)}
    out=ROOT/"artifacts"/"production_model_report.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Production alias points to {name} version {version} ({metric}={score:.4f})")
if __name__=="__main__": main()
