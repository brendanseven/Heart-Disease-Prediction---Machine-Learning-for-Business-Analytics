# Heart Disease Risk Prediction

End-to-end proof-of-concept ML system that predicts heart disease risk from
clinical and demographic features. Built for the MLBA course final project.

The pipeline trains a Logistic Regression model on the UCI Heart Disease
dataset, serves predictions through a Streamlit dashboard running on AWS EC2,
and persists every prediction to AWS RDS (MySQL) for audit and analytics.

## Final model

| Metric | Value |
|---|---|
| Model | Logistic Regression (class_weight=balanced) |
| Accuracy | 0.624 |
| Recall | 0.618 |
| Precision | 0.512 |
| F1 | 0.560 |
| ROC-AUC | 0.646 |

Metrics are reported on a held-out 20% stratified test set. We chose Logistic
Regression over Random Forest and Gradient Boosting because it had the best
recall and F1, and it is fully interpretable via its coefficients, which
matters for a clinical screening tool.

## Repository layout

```
.
├── data/                  Raw and expanded datasets
│   ├── heart_disease_uci.csv
│   └── heart_disease_dataset.csv
├── notebooks/             Exploratory analysis
│   └── heart_disease_model.ipynb
├── src/                   Production code
│   ├── train.py           Trains, evaluates, saves the pipeline
│   └── app.py             Streamlit dashboard
├── models/                Serialized model artifacts
│   └── heart_disease_pipeline.pkl
├── infrastructure/        Cloud + DB setup
│   ├── setup_db.sql       Creates the predictions table
│   └── deploy_runbook.md  Step-by-step AWS deployment guide
├── reports/               Generated figures
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── requirements.txt
├── .env.example           Template for local environment variables
└── README.md
```

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/brendanseven/Heart-Disease-Prediction---Machine-Learning-for-Business-Analytics.git
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# edit .env with your RDS credentials (or leave blank to run without DB)
```

The Streamlit app reads database credentials from environment variables
(`DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME`, `DB_PORT`). If `DB_HOST` is not
set, the app still runs but skips the prediction-log persistence.

### 3. Retrain the model (optional)

The repo ships with a pre-trained pipeline at `models/heart_disease_pipeline.pkl`.
To retrain from the dataset:

```bash
python src/train.py
```

This compares Random Forest, balanced Random Forest, Gradient Boosting, and
balanced Logistic Regression, selects the best by F1, and writes:

- `models/heart_disease_pipeline.pkl`
- `reports/model_comparison.png`
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`

### 4. Run the dashboard locally

```bash
# load env vars, then run Streamlit
set -a; source .env; set +a
streamlit run src/app.py
```

Open http://localhost:8501.

The dashboard has four tabs:

1. **Risk Calculator** – submit a patient, get a probability + per-feature
   contribution breakdown.
2. **Population Insights** – disease prevalence by age, chest pain type, and
   risk factors across the training set.
3. **Prediction Log** – every prediction written to RDS, with filters and
   CSV export.
4. **Model Insights** – global feature importance (coefficients vs. odds
   ratios) and a what-if sensitivity sweep.

## AWS deployment

See [`infrastructure/deploy_runbook.md`](infrastructure/deploy_runbook.md) for
the full S3 + EC2 + RDS setup, including security group rules, model upload,
and Streamlit launch.

Architecture:

```
Clinician browser
       |
       v
  EC2 (Streamlit) <----> RDS MySQL (predictions table)
       ^
       |
      S3 (model.pkl + dataset.csv)
```

## Reproducing results end-to-end

A reasonably technical reader can reproduce the project by:

1. Running `python src/train.py` to retrain the model and regenerate the
   figures from the dataset shipped in `data/`.
2. Running `streamlit run src/app.py` to launch the dashboard locally
   against the freshly trained model. Without `DB_HOST` set, predictions
   are returned but not persisted, which is fine for local validation.
3. Following `infrastructure/deploy_runbook.md` to provision the AWS
   resources and put the same code behind a public URL.

## Secrets

No credentials are committed to this repository. The `.env` file is
git-ignored and only `.env.example` ships with placeholder values.
