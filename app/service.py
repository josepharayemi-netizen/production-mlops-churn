from pathlib import Path
import json, joblib
import pandas as pd
from app.schemas import Customer, Prediction


class ChurnService:
    def __init__(self,model_dir:Path=Path("models")):
        if not (model_dir/"churn_model.joblib").exists():
            from app.train import train
            train(model_dir)
        self.model=joblib.load(model_dir/"churn_model.joblib")
        self.metadata=json.loads((model_dir/"metadata.json").read_text())

    def predict(self,c:Customer)->Prediction:
        probability=round(float(self.model.predict_proba(pd.DataFrame([c.model_dump()]))[0,1]),4)
        reasons=[]
        if c.tenure_months<6: reasons.append("SHORT_TENURE")
        if c.contract_type=="monthly": reasons.append("MONTHLY_CONTRACT")
        if c.support_tickets>=4: reasons.append("HIGH_SUPPORT_DEMAND")
        if c.monthly_charges>=100: reasons.append("HIGH_MONTHLY_CHARGES")
        if c.usage_hours<15: reasons.append("LOW_USAGE")
        tier="HIGH" if probability>=.7 else "MEDIUM" if probability>=.4 else "LOW"
        return Prediction(churn_probability=probability,risk_tier=tier,
            model_version=self.metadata["model_version"],reasons=reasons or ["NO_MAJOR_RISK_DRIVER"])
