"""Train a raw-feature sklearn pipeline and log/register it in MLflow."""
import json, os
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd, yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[2]
def main():
    os.chdir(ROOT); cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",cfg["mlflow"]["tracking_uri"])); mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    d=ROOT/cfg["data"]["processed_dir"]; Xtr=pd.read_csv(d/"X_train_raw.csv"); Xte=pd.read_csv(d/"X_test_raw.csv"); ytr=np.load(d/"y_train.npy"); yte=np.load(d/"y_test.npy")
    cats=Xtr.select_dtypes(include=["object","category"]).columns.tolist(); nums=Xtr.select_dtypes(include=["number"]).columns.tolist()
    pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="constant",fill_value=0)),("scale",StandardScaler())]),nums),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)])
    m=cfg["model"]; estimator=RandomForestClassifier(n_estimators=int(m["n_estimators"]),max_depth=m["max_depth"],random_state=int(m["random_state"]),class_weight=m["class_weight"])
    model=Pipeline([("preprocessor",pre),("classifier",estimator)]); model.fit(Xtr,ytr); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
    metrics={"accuracy":float(accuracy_score(yte,pred)),"precision":float(precision_score(yte,pred,zero_division=0)),"recall":float(recall_score(yte,pred,zero_division=0)),"f1_score":float(f1_score(yte,pred,zero_division=0)),"roc_auc":float(roc_auc_score(yte,prob))}
    os.makedirs("models",exist_ok=True); joblib.dump(model,cfg["paths"]["full_pipeline"]); joblib.dump(model.named_steps["classifier"],cfg["paths"]["model"])
    Xtr_processed=np.load(d/"X_train_final.npy"); ytr_processed=np.load(d/"y_train.npy")
    joblib.dump(DecisionTreeClassifier(random_state=int(m["random_state"])).fit(Xtr_processed,ytr_processed),"models/decision_tree_baseline.pkl")
    joblib.dump(0.5,"models/churn_decision_threshold.pkl")
    from sklearn.base import clone
    xgb_pipeline=Pipeline([("preprocessor",clone(pre)),("classifier",XGBClassifier(n_estimators=150,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,objective="binary:logistic",eval_metric="logloss",n_jobs=1,random_state=int(m["random_state"])))])
    xgb_pipeline.fit(Xtr,ytr)
    joblib.dump(xgb_pipeline,"models/churn_xgboost_pipeline.pkl")
    with mlflow.start_run(run_name="Lab9_RandomForest_Pipeline") as run:
        mlflow.log_params({"n_estimators":m["n_estimators"],"max_depth":m["max_depth"],"random_state":m["random_state"],"class_weight":m["class_weight"]}); mlflow.log_metrics(metrics)
        mlflow.log_param("model_family","RandomForestPipeline")
        mlflow.sklearn.log_model(model, name="model", registered_model_name=cfg["registry"]["model_name"], skops_trusted_types=["sklearn.tree._tree.Tree","numpy.dtype"])
        (ROOT/"artifacts").mkdir(exist_ok=True); (ROOT/"artifacts"/"production_model_report.json").write_text(json.dumps({"run_id":run.info.run_id,"model_name":cfg["registry"]["model_name"],"metrics":metrics},indent=2),encoding="utf-8")
    print("Logged pipeline and registered model; metrics:",metrics)
if __name__=="__main__": main()
