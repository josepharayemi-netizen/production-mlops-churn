# Model Card: Customer Churn Classifier

## Intended use

Prioritize customers for retention review. The output is decision support and must not be used to deny service or make consequential decisions without human oversight.

## Training data

The demonstration uses 4,000 reproducibly generated synthetic telecom customer records. It contains no real customer or personal data.

## Model

Random forest with class balancing, categorical one-hot encoding, numeric standardization, fixed random seed, and a versioned metadata artifact.

## Baseline evaluation

The deterministic baseline run produced ROC-AUC 0.7493. Precision and recall depend on the selected operating threshold and should be optimized against business costs using real, representative validation data.

## Limitations

- Synthetic relationships do not establish real-world performance.
- Reason codes describe operational risk indicators, not causal explanations.
- Data drift does not automatically imply model-performance degradation.
- Fairness, calibration, privacy, security, and regional regulatory reviews are required before production use.

## Monitoring

Monitor prediction distribution, data quality, PSI, calibration, false-positive rate, recall, latency, and model/version lineage. Require approval before threshold changes or retraining promotion.
