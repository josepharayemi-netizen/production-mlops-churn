from fastapi import FastAPI
from app.drift import report
from app.schemas import Customer, Prediction
from app.service import ChurnService

app=FastAPI(title="Customer Churn MLOps API",version="1.0.0")
service=ChurnService()


@app.get("/health")
def health():
    return {"status":"healthy","model_version":service.metadata["model_version"]}


@app.post("/predict",response_model=Prediction)
def predict(customer:Customer):
    return service.predict(customer)


@app.post("/monitor/drift")
def monitor(batch:list[Customer]):
    return report(service.metadata["baseline"],[item.model_dump() for item in batch])
