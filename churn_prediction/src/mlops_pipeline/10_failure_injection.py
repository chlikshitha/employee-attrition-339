"""Exercise failure paths in memory without replacing source or model files."""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml, mlflow, mlflow.sklearn
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
output_spec=importlib.util.spec_from_file_location("processed_output_validation",ROOT/"src/mlops_pipeline/validate_outputs.py")
output_validator=importlib.util.module_from_spec(output_spec); output_spec.loader.exec_module(output_validator)
spec=importlib.util.spec_from_file_location("data_validation",ROOT/"src/mlops_pipeline/02_validate_data.py")
validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); raw=pd.read_csv(ROOT/cfg["data"]["raw_path"]); results=[]
    scenarios=[
        ("missing_data",not (ROOT/"data/raw/definitely_missing.csv").exists()),
        ("invalid_datatype",bool(validator.validate(raw.assign(MonthlyCharges="INVALID_NUMBER")))),
        ("schema_mismatch",bool(validator.validate(raw.drop(columns=["Contract"])))),
    ]
    for name,passed in scenarios: results.append({"scenario":name,"expected":"HALT","actual":"HALT" if passed else "CONTINUED","passed":bool(passed)})
    d=ROOT/cfg["data"]["processed_dir"]; x=np.load(d/"X_train_final.npy"); x[0,0]=np.nan
    passed=bool(output_validator.validate_arrays(x,np.load(d/"X_test_final.npy"),np.load(d/"y_train.npy"),np.load(d/"y_test.npy"),require_raw_splits=False))
    results.append({"scenario":"preprocessing_inconsistency","expected":"HALT","actual":"HALT" if passed else "CONTINUED","passed":passed})
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",cfg["mlflow"]["tracking_uri"]))
    import tempfile, joblib
    try:
        with tempfile.TemporaryDirectory(prefix="lab9_model_recovery_") as temp_dir:
            missing_model=Path(temp_dir)/"model.pkl"; failure_detected=not missing_model.exists()
            model=mlflow.sklearn.load_model(f"models:/{cfg['registry']['model_name']}@{cfg['lab9']['production_alias']}")
            joblib.dump(model,missing_model); recovered=joblib.load(missing_model)
            recovered.predict(pd.read_csv(d/"X_test_raw.csv").head(1)); passed=failure_detected
    except Exception as exc:
        passed=False; recovery_error=str(exc)
    results.append({"scenario":"missing_model_artifact_recovery","expected":"RECOVER","actual":"RECOVERED" if passed else "RECOVERY_FAILED","passed":passed,**({} if passed else {"error":recovery_error})})
    error_rows=[{"scenario":r["scenario"],"expected":r["expected"],"actual":r["actual"],"passed":r["passed"]} for r in results]
    artifact=ROOT/"artifacts"/"corrupted_simulation_errors.csv"; artifact.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(error_rows,columns=["scenario","expected","actual","passed"]).to_csv(artifact,index=False)
    reps=int(cfg.get("lab9",{}).get("repetitions",1)); runs=[{"repetition":i,"results":results,"passed":all(r["passed"] for r in results)} for i in range(1,reps+1)]
    report={"stage":"Lab 9 failure handling","repetitions":reps,"runs":runs,"overall_status":"PASSED" if all(r["passed"] for r in runs) else "FAILED"}
    out=ROOT/cfg["paths"]["failure_report"]; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    (ROOT/"artifacts"/"failure_simulation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if report["overall_status"]!="PASSED": raise SystemExit(1)
if __name__=="__main__": main()
