from pathlib import Path
from app.data import generate
from app.drift import psi
from app.schemas import Customer
from app.service import ChurnService


def test_data_is_reproducible():
    assert generate(5,42).equals(generate(5,42))


def test_prediction_has_version_and_reasons(tmp_path:Path):
    service=ChurnService(tmp_path)
    customer=Customer(tenure_months=2,monthly_charges=130,support_tickets=7,
        usage_hours=5,contract_type="monthly",payment_method="manual",internet_service="fiber")
    result=service.predict(customer)
    assert 0<=result.churn_probability<=1
    assert result.model_version=="1.0.0"
    assert "SHORT_TENURE" in result.reasons


def test_psi_detects_shift():
    assert psi(list(range(100)),list(range(100)))<.01
    assert psi(list(range(100)),list(range(1000,1100)))>.25
