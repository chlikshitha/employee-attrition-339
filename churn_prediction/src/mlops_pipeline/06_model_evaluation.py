"""Evaluate the saved full raw-input pipeline and create evaluation visuals."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
ROOT=Path(__file__).resolve().parents[2]
def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8")); d=ROOT/cfg["data"]["processed_dir"]; model=joblib.load(ROOT/cfg["paths"]["full_pipeline"])
    X=pd.read_csv(d/"X_test_raw.csv"); y=np.load(d/"y_test.npy"); pred=model.predict(X); proba=model.predict_proba(X)[:,1]
    metrics={"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1_score":float(f1_score(y,pred,zero_division=0)),"roc_auc":float(roc_auc_score(y,proba))}
    report={"stage":"Lab 8/9 model evaluation","test_rows":len(y),"metrics":metrics,"confusion_matrix":confusion_matrix(y,pred).tolist()}; out=ROOT/cfg["paths"]["evaluation_report"]; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    art=ROOT/"artifacts"; art.mkdir(exist_ok=True); fig,ax=plt.subplots(); ConfusionMatrixDisplay.from_predictions(y,pred,ax=ax); fig.savefig(art/"confusion_matrix.png",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(); RocCurveDisplay.from_predictions(y,proba,ax=ax); fig.savefig(art/"roc_curve.png",bbox_inches="tight"); plt.close(fig)
    fn=X.loc[(y==1)&(pred==0)].copy(); fp=X.loc[(y==0)&(pred==1)].copy(); (ROOT/"outputs").mkdir(exist_ok=True)
    fn_out=fn.assign(Actual_Churn=1,Predicted_Churn=0); fp_out=fp.assign(Actual_Churn=0,Predicted_Churn=1)
    fn_out.to_csv(ROOT/"outputs"/"false_negatives.csv",index=False); fp_out.to_csv(ROOT/"outputs"/"false_positives.csv",index=False)
    fn_out.to_csv(ROOT/"outputs"/"lab7_false_negatives.csv",index=False); fp_out.to_csv(ROOT/"outputs"/"lab7_false_positives.csv",index=False)
    print("Evaluation report:",out.relative_to(ROOT))
if __name__=="__main__": main()
