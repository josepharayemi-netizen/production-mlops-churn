from pathlib import Path
import numpy as np
import pandas as pd


def generate(rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, rows)
    charges = rng.normal(75, 28, rows).clip(15, 180)
    tickets = rng.poisson(1.8, rows)
    usage = rng.normal(45, 18, rows).clip(0, 120)
    contract = rng.choice(["monthly", "annual", "two_year"], rows, p=[.55, .3, .15])
    payment = rng.choice(["automatic", "manual"], rows, p=[.62, .38])
    internet = rng.choice(["fiber", "dsl", "none"], rows, p=[.58, .32, .10])
    logit = -1.8 - .035*tenure + .012*(charges-70) + .35*tickets - .018*usage
    logit += np.where(contract=="monthly", .9, np.where(contract=="annual", -.3, -.8))
    logit += np.where(payment=="manual", .35, -.15) + np.where(internet=="fiber", .25, 0)
    probability = 1/(1+np.exp(-logit))
    return pd.DataFrame({"tenure_months":tenure,"monthly_charges":charges.round(2),
        "support_tickets":tickets,"usage_hours":usage.round(1),"contract_type":contract,
        "payment_method":payment,"internet_service":internet,"churn":rng.binomial(1,probability)})


def main():
    path=Path("data/customers.csv"); path.parent.mkdir(exist_ok=True)
    data=generate(); data.to_csv(path,index=False)
    print(f"Wrote {len(data)} customers to {path}")


if __name__=="__main__":
    main()
