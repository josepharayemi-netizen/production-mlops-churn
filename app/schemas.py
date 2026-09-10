from typing import Literal
from pydantic import BaseModel, Field


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    support_tickets: int = Field(ge=0, le=100)
    usage_hours: float = Field(ge=0, le=744)
    contract_type: Literal["monthly","annual","two_year"]
    payment_method: Literal["automatic","manual"]
    internet_service: Literal["fiber","dsl","none"]


class Prediction(BaseModel):
    churn_probability: float
    risk_tier: str
    model_version: str
    reasons: list[str]
