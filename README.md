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

Enter a transaction, get back a fraud probability, an alert decision, and the
five features that drove it.

Trained on [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1): 6,362,620
simulated mobile money transactions, 8,213 of them fraud (0.129%).

<img width="1708" alt="Risk console" src="https://github.com/user-attachments/assets/074bcbfe-bc18-4a4a-834a-e56afca6f262" />

---

## Results

<!-- METRICS:START -->
Tested on the last 1,248,736 transactions (everything after hour 355), which the model never saw while training or while picking the threshold.

| Metric | Value |
| --- | --- |
| PR-AUC (average precision) | 0.9998 |
| Recall | 100.0% |
| Precision | 100.0% |
| ROC-AUC | 1.0000 |
| Alerts per 1,000 transactions | 3.40 |
| Fraud value caught | 100.0% ($6.68B of $6.68B) |
| Savings vs. no model | $6.68B |

The alert cutoff is 0.0900, not 0.5. It comes from minimising cost on the validation set, where a false alarm costs $25 of review time and a missed fraud costs the full amount. At 0.5 the same model gets 100.0% recall and saves $6.68B, against 100.0% and $6.68B at the chosen point.

How much of this is real: the two line rule `oldbalanceOrg == amount and newbalanceOrig == 0` on its own gets 100.0% precision and 97.4% recall on the same test set, F1 of 0.987. The model gets 1.000, so it adds +0.013.

Dropping the three balance features that encode that rule (`--feature-set realistic`) gives PR-AUC 0.9781, precision 57.0%, recall 99.5%, and 5.9 alerts per 1,000 rows instead of 3.4. That difference is how much of the headline number came from the simulator rather than from anything the model learned.

Same model trained on a random split instead scores 0.9928 PR-AUC against 0.9998 here: -0.0069, so the random split didn't inflate anything here. PaySim puts most of its fraud in the later hours, which are the ones the time based test set uses, so that window is denser in fraud and a bit easier. The time based split is still the right one, it just isn't where the optimism is on this dataset. Run `python train.py --compare-random-split` to reproduce.

<sub>Trained on 6,362,620 rows (8,213 fraud) in 462s, xgboost 3.3.0, scikit-learn 1.9.0, scale_pos_weight 1</sub>
<!-- METRICS:END -->

---

## The dataset has a shortcut in it

My first training run came back with 100% precision and 100% recall. That
isn't a thing that happens, so I went looking for the reason.

PaySim generates a fraudulent transaction by emptying the sender's account.
Across all 6.3M rows:

| `oldbalanceOrg == amount` and `newbalanceOrig == 0` | Fraud | Not fraud |
|---|---|---|
| Share of rows matching | **97.7%** | **0.0%** |

The classes separate with a two line `if` statement. Any model given features
built on that looks near perfect without having learned anything about fraud.

So every training run now reports two extra things:

1. **A rule baseline.** That `if` statement scored as if it were a model, so
   there is always something to beat.
2. **An ablation.** The same model refit without the three balance features
   that encode the rule, which shows how much of the score came from the
   simulator.

Both are in the [results](#results) above.

This is near the top rather than buried in limitations, because a near perfect
score on a public dataset is usually the dataset's doing. What would still
hold up on real transactions is the evaluation setup and the threshold logic,
not the accuracy.

---

## How it is trained

Three decisions, and why:

**Split on time, not at random.** `step` is an hour counter over a 30 day
simulation. A random split trains on hour 500 and tests on hour 200, which no
deployed system ever gets to do. Training uses the earliest 60% of rows,
validation the next 20%, test the last 20%.

**PR-AUC as the headline metric.** At a 0.129% fraud rate, a model that always
answers "not fraud" is 99.87% accurate, and ROC-AUC sits near the top of its
range regardless. Average precision is the one that moves when the model
improves.

**A threshold chosen by cost.** Missing a fraud costs the full transaction
amount. A false alarm costs one analyst review, set at $25. Those are nowhere
near equal, so `train.py` sweeps candidate thresholds against that cost
function and picks the cheapest instead of leaving it at 0.5.

Class imbalance is handled by tuning `scale_pos_weight` against validation
PR-AUC.

---

## Models tried

From the original notebook, on the random split I later replaced. Kept for the
comparison, but not comparable to the results above.

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | 99.90% | 99.97% | **99.98%** |
| Precision | 65.06% | **98.02%** | 96.60% |
| Recall | 48.63% | 78.59% | **85.98%** |
| F1 | 55.66% | 87.24% | **90.98%** |
| False negatives | 1,070 | 446 | **292** |

XGBoost won on recall and F1, which is what matters at this level of
imbalance.

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

| Endpoint | Returns |
|---|---|
| `POST /predict` | Fraud probability, alert decision, expected loss, SHAP factors |
| `GET /health` | Which model is loaded and whether SHAP is working |

`backend/features.py` builds the features and is imported by both training and
serving, so a feature cannot be computed one way at training time and another
at prediction time. A test checks the two paths agree. The saved model also
records which columns it was fitted on, and the API serves exactly those.

---

## Features

Eight fields go in:

| Field | Description |
|---|---|
| Step | Hour of the transaction |
| Transaction type | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Amount | Transaction amount |
| Sender balance before / after | Sender's balance either side of the transaction |
| Receiver balance before / after | Receiver's balance either side of the transaction |
| Flagged fraud | Rule based flag from the source system |

Five more are derived from them:

| Derived | Why it exists |
|---|---|
| Hour of day | Raw `step` doesn't survive a time based split, since every test value falls outside the training range |
| Sender ledger mismatch | Balances should satisfy `new = old - amount`. Fraud rows often don't |
| Receiver ledger mismatch | The same check on the receiving side |
| Sender emptied | Whether the account went to exactly zero |
| Amount vs sender balance | How much of the available balance moved |

---

## Getting started

**Docker**

```bash
docker compose up --build
```

Dashboard on <http://localhost:8501>, API docs on <http://localhost:8000/docs>.

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

The trained model is committed, so the dataset is only needed to retrain.
Download [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) into
`dataset/`, then:

```bash
cd backend && python train.py
```

| Flag | Effect |
|---|---|
| `--review-cost 50` | Change what a false alarm costs. The threshold moves with it |
| `--feature-set realistic` | Drop the balance features that encode the shortcut |
| `--compare-random-split` | Also train on a random split, for comparison |
| `--nrows 500000` | Train on a slice, for a quick run |

**Tests**

```bash
cd backend && python -m pytest
```

27 tests covering feature building, the time based split, the threshold sweep
against a brute force version, the rule baseline, and the API. GitHub Actions
runs these plus ruff on every push and builds both images.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8000` | Where the dashboard looks for the API |
| `MODEL_DIR` | `backend/model` | Where the model and metrics live |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per IP cap on `/predict`. In memory, single worker only |
| `LOG_LEVEL` | `INFO` | Backend logging |

Streamlit Cloud can't set environment variables, so the dashboard also reads
`API_URL` from secrets:

```toml
API_URL = "https://your-backend.onrender.com"
```

---

## Limitations

- **The data is simulated.** Every number here is a ceiling, not an estimate.
- **No account history.** Each transaction is scored alone. A real system would
  have account age, recent activity, device fingerprints and counterparty
  information, which is most of what actually catches fraud.
- **The review cost is a placeholder.** $25 per alert is made up. Change it and
  the operating point changes with it.
- **No fairness testing.** PaySim has no demographic columns. Real data would
  need this before deployment.
- **Explanations degrade rather than fail.** If SHAP can't run, the API returns
  `available: false` with a reason. An earlier version invented attribution
  numbers when SHAP was missing, which is worse than returning nothing.

---

## Structure

```
backend/
  app.py             FastAPI service
  features.py        feature building, shared by training and serving
  train.py           training, evaluation, threshold selection
  model/             trained model and metrics
  tests/
frontend/
  streamlit_app.py   the dashboard
notebook/            the original exploratory notebook
scripts/             regenerates the results section of this file
dataset/             where the Kaggle CSV goes
```
