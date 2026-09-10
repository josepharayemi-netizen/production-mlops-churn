from pathlib import Path
import json, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.data import generate

NUMERIC=["tenure_months","monthly_charges","support_tickets","usage_hours"]
CATEGORICAL=["contract_type","payment_method","internet_service"]
FEATURES=NUMERIC+CATEGORICAL


def train(model_dir: Path=Path("models")):
    data=generate(); train_df,test_df=train_test_split(data,test_size=.25,random_state=42,stratify=data["churn"])
    prep=ColumnTransformer([("numeric",StandardScaler(),NUMERIC),
        ("categorical",OneHotEncoder(handle_unknown="ignore"),CATEGORICAL)])
    model=Pipeline([("preprocessor",prep),("classifier",RandomForestClassifier(
        n_estimators=180,max_depth=9,class_weight="balanced",random_state=42,n_jobs=-1))])
    model.fit(train_df[FEATURES],train_df["churn"])
    probability=model.predict_proba(test_df[FEATURES])[:,1]; prediction=(probability>=.5).astype(int)
    metrics={"roc_auc":round(roc_auc_score(test_df["churn"],probability),4),
        "precision":round(precision_score(test_df["churn"],prediction),4),
        "recall":round(recall_score(test_df["churn"],prediction),4)}
    model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,model_dir/"churn_model.joblib")
    metadata={"model_version":"1.0.0","features":FEATURES,"metrics":metrics,
        "baseline":{c:train_df[c].astype(float).tolist() for c in NUMERIC}}
    (model_dir/"metadata.json").write_text(json.dumps(metadata))
    try:
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")
        with mlflow.start_run(run_name="churn-random-forest"):
            mlflow.log_params({"n_estimators":180,"max_depth":9})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model,"model")
    except ImportError:
        pass
    print(metrics)
    return metrics


if __name__=="__main__":
    train()
