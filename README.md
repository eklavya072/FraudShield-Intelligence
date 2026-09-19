<h1 align="center">FraudShield Intelligence</h1>

<p align="center">
  Fraud detection on 6.3M mobile money transactions.<br>
  XGBoost and SHAP behind a FastAPI service, with a Streamlit dashboard.
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-1798e6?style=flat&logo=xgboost" alt="XGBoost"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker"></a>
</p>

<p align="center">
  <b><a href="https://fraudshield-intelligence-evci2hh3g7wzrytqmg848y.streamlit.app">Live demo</a></b>
</p>

---

A transaction is submitted to the API, which returns a fraud probability, an
alert decision against a cost-optimised threshold, and the five features that
contributed most to the score.

The model is trained on [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1),
a simulation of mobile money transfers containing 6,362,620 transactions of
which 8,213 are fraudulent (0.129%).

<img width="1708" alt="Risk console" src="https://github.com/user-attachments/assets/074bcbfe-bc18-4a4a-834a-e56afca6f262" />

---

## Results

<!-- METRICS:START -->
Held-out test partition: the final 1,248,736 transactions (all rows after hour 355), unseen during training and threshold selection.

| Metric | Value |
| --- | --- |
| PR-AUC (average precision) | 0.9998 |
| Recall | 100.0% |
| Precision | 100.0% |
| ROC-AUC | 1.0000 |
| Alerts per 1,000 transactions | 3.40 |
| Fraud value recovered | 100.0% ($6.68B of $6.68B) |
| Net benefit vs. no model | $6.68B |

**Operating point.** Alerts are raised at a probability of 0.0900 rather than 0.5, selected by minimising expected cost on the validation partition under a $25 review cost and a false-negative cost equal to the transaction amount. At the 0.5 default the same model records 100.0% recall and $6.68B net benefit, against 100.0% and $6.68B at the selected threshold.

**Rule baseline.** The two-line rule `oldbalanceOrg == amount and newbalanceOrig == 0` achieves 100.0% precision and 97.4% recall on the same partition (F1 0.987) against the model's 1.000, a margin of +0.013.

**Feature ablation.** Refitting without the three balance features (`--feature-set realistic`) yields PR-AUC 0.9781, precision 57.0%, recall 99.5%, and 5.9 alerts per 1,000 transactions against 3.4. The difference quantifies the contribution of the simulator's generation rule to the headline figures.

**Split control.** The same model trained on a random split scores 0.9928 PR-AUC against 0.9998: a difference of -0.0069. The random split does not inflate the result on this dataset: PaySim concentrates fraud in later hours, which constitute the temporal test window, making that partition denser in positives and marginally easier. The temporal split remains the correct design, as the only one that reflects deployment conditions. Reproduce with `python train.py --compare-random-split`.

<sub>Trained on 6,362,620 transactions (8,213 fraudulent) in 462s · xgboost 3.3.0 · scikit-learn 1.9.0 · scale_pos_weight 1</sub>
<!-- METRICS:END -->

---

## Label leakage in PaySim

PaySim generates a fraudulent transaction by transferring the sender's entire
balance. The resulting signature is close to deterministic:

| Condition | Fraudulent rows | Legitimate rows |
|---|---|---|
| `oldbalanceOrg == amount` and `newbalanceOrig == 0` | **97.7%** | **0.0%** |

The classes are therefore separable by a two-line rule, and any model given
features derived from that relationship will approach perfect scores without
learning a generalisable fraud pattern. Published results on this dataset that
report near-perfect metrics without a baseline comparison are usually
measuring this artefact.

Two controls run on every training pass to quantify it:

| Control | Purpose |
|---|---|
| **Rule baseline** | The `if` statement above, scored as a classifier, establishing the floor any model must clear |
| **Feature ablation** | The same model refit without the three balance features that encode the relationship |

Both appear in [Results](#results). The engineering that transfers to real
transaction data is the evaluation design and the threshold selection; the
reported accuracy does not.

---

## Methodology

**Temporal split.** The `step` column is an hour index across a 30-day
simulation. A random split permits training on hour 500 and evaluating on hour
200, which no deployed system can do. Partitions are taken in chronological
order: the earliest 60% of rows for training, the next 20% for validation, the
final 20% held out for test.

**Average precision as the primary metric.** At a 0.129% positive rate, a
classifier that never predicts fraud achieves 99.87% accuracy, and ROC-AUC
saturates. PR-AUC is the metric that responds to genuine improvement.

**Cost-based decision threshold.** A false negative costs the full transaction
amount; a false positive costs one analyst review, set at $25. Given that
asymmetry, 0.5 is arbitrary. Candidate thresholds are swept against the cost
function and the minimum is selected on the validation partition.

**Class imbalance.** `scale_pos_weight` is tuned against validation PR-AUC
rather than fixed by assumption.

---

## Model selection

Evaluated in the original notebook on a random split, which was later replaced
by the temporal split described above. Retained for comparison; not directly
comparable to the figures in [Results](#results).

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | 99.90% | 99.97% | **99.98%** |
| Precision | 65.06% | **98.02%** | 96.60% |
| Recall | 48.63% | 78.59% | **85.98%** |
| F1 | 55.66% | 87.24% | **90.98%** |
| False negatives | 1,070 | 446 | **292** |

XGBoost was selected on recall and F1, the metrics that matter at this level of
class imbalance.

---

## Architecture

```
Streamlit dashboard                FastAPI service
(Streamlit Cloud)  ──────────────► (Render)
:8501               POST /predict   :8000
                                        │
                                        ▼
                                 XGBoost + SHAP
                                        │
                                        ▼
                         probability · decision · top 5 factors
```

| Endpoint | Response |
|---|---|
| `POST /predict` | Fraud probability, alert decision, expected loss, SHAP attributions |
| `GET /health` | Loaded model metadata and explainer availability |

Feature construction lives in `backend/features.py` and is imported by both the
training script and the service, preventing divergence between training-time
and serving-time features; a test asserts the two code paths agree. The
serialised model records the columns it was fitted on, and the API constructs
exactly those.

---

## Features

Eight fields are supplied per request:

| Field | Description |
|---|---|
| Step | Hour index of the transaction |
| Transaction type | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Amount | Transaction amount |
| Sender balance before / after | Originating balance either side of the transaction |
| Receiver balance before / after | Destination balance either side of the transaction |
| Flagged fraud | Rule-based indicator from the source system |

Five further features are derived:

| Derived feature | Rationale |
|---|---|
| Hour of day | Raw `step` does not transfer across a temporal split, as every test value falls outside the training range |
| Sender ledger mismatch | Balances should satisfy `new = old - amount`; fraudulent rows frequently violate this |
| Receiver ledger mismatch | The equivalent check on the destination account |
| Sender emptied | Whether the originating balance reached exactly zero |
| Amount to balance ratio | Proportion of the available balance moved |

---

## Getting started

**Docker**

```bash
docker compose up --build
```

Dashboard on <http://localhost:8501>, API documentation on
<http://localhost:8000/docs>.

**Local**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt -r frontend/requirements.txt
```

```bash
cd backend && uvicorn app:app --reload
```

```bash
cd frontend && API_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py
```

**Retraining**

The trained model is committed, so the dataset is required only for retraining.
Download [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) into
`dataset/`, then:

```bash
cd backend && python train.py
```

| Flag | Effect |
|---|---|
| `--review-cost 50` | Adjusts the cost of a false positive; the threshold moves accordingly |
| `--feature-set realistic` | Excludes the balance features that encode the leakage |
| `--compare-random-split` | Additionally trains on a random split for comparison |
| `--nrows 500000` | Trains on a subset for a faster run |

**Tests**

```bash
cd backend && python -m pytest
```

27 tests covering feature construction, the temporal split, the threshold sweep
against a brute-force implementation, the rule baseline, and the API. GitHub
Actions runs the suite and ruff on every push, and builds both images.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8000` | API endpoint used by the dashboard |
| `MODEL_DIR` | `backend/model` | Location of the model and metrics |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-IP cap on `/predict`; in-process, single worker only |
| `LOG_LEVEL` | `INFO` | Backend log level |

Streamlit Cloud does not support environment variables, so the dashboard also
resolves `API_URL` from secrets:

```toml
API_URL = "https://your-backend.onrender.com"
```

---

## Limitations

- **Simulated data.** All reported figures are an upper bound rather than an
  estimate of production performance.
- **No entity history.** Transactions are scored in isolation. A production
  system would incorporate account age, recent activity, device fingerprints
  and counterparty relationships, which account for most real detection power.
- **Placeholder review cost.** The $25 per-alert figure is illustrative;
  changing it moves the operating point.
- **No fairness evaluation.** PaySim contains no demographic attributes.
  Disparate impact analysis would be required before deployment on real data.
- **Graceful explanation degradation.** Where SHAP cannot produce an
  attribution, the API returns `available: false` with a reason rather than a
  substitute value.

---

## Structure

```
backend/
  app.py             FastAPI service
  features.py        feature construction, shared by training and serving
  train.py           training, evaluation, threshold selection
  model/             serialised model and metrics
  tests/
frontend/
  streamlit_app.py   dashboard
notebook/            original exploratory analysis
scripts/             regenerates the results section of this file
dataset/             location for the Kaggle CSV
```
