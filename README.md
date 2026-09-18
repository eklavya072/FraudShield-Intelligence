<h1 align="center">FraudShield Intelligence</h1>

<p align="center">
  Real-time fraud detection on 6.3M mobile money transactions.<br>
  XGBoost behind a FastAPI service, with a Streamlit dashboard for scoring and explanations.
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

## Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Results](#results)
- [What I found in the dataset](#what-i-found-in-the-dataset)
- [What I changed after the first version](#what-i-changed-after-the-first-version)
- [Model comparison](#model-comparison)
- [Architecture](#architecture)
- [Input features](#input-features)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Tests](#tests)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Project structure](#project-structure)
- [About me](#about-me)

---

## Overview

This started as a Kaggle notebook and turned into something I kept working on,
mostly because the notebook version kept giving me results that looked too
good to be true. They were, and finding out why taught me more than the model
did.

What's here now:

- An XGBoost classifier trained on the PaySim dataset, 6.3M transactions with
  a 0.129% fraud rate
- A FastAPI service that scores a transaction and returns SHAP values for it
- A Streamlit dashboard for trying it out
- Docker for both services, and GitHub Actions running the tests

PaySim is simulated data rather than real transactions, so the numbers below
are a ceiling rather than an estimate. The [dataset section](#what-i-found-in-the-dataset)
explains why that matters more than I expected.

---

## Screenshots

**Scoring a transaction**

<img width="1708" alt="Risk console" src="https://github.com/user-attachments/assets/074bcbfe-bc18-4a4a-834a-e56afca6f262" />

**Result and risk score**

<img width="1709" alt="Prediction result" src="https://github.com/user-attachments/assets/fb83b3f0-26b2-428a-b0f2-98038d0fea22" />

**SHAP feature attribution**

<img width="680" alt="SHAP explanation" src="https://github.com/user-attachments/assets/1562f901-f5af-436e-9cd3-b8260fae4c1f" />

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

## What I found in the dataset

This is the part I'd most want someone to read.

My first proper training run came back with 100% precision and 100% recall,
which is not something that happens. So I went looking for the reason.

PaySim creates a fraudulent transaction by emptying the sender's account.
Across all 6.3M rows:

| Condition | Fraud | Not fraud |
|---|---|---|
| `oldbalanceOrg == amount` and `newbalanceOrig == 0` | **97.7%** | **0.0%** |

The two classes come apart with a two line `if` statement. Any model given
features built on top of that will look close to perfect without having
learned anything about fraud.

So `train.py` now does two extra things on every run:

1. Scores that rule on its own and reports it next to the model, so there's
   always something to compare against.
2. Refits without the three balance features that encode the rule, which shows
   how much of the score came from the simulator rather than the model.

Both numbers are in the [results](#results) above.

I'm keeping this near the top instead of in a limitations section, because a
near perfect score on a public dataset is usually the dataset's doing. The
parts of this project that would still hold up on real transactions are the
threshold logic and the evaluation setup, not the accuracy.

---

## What I changed after the first version

The original notebook is still in `notebook/`, but it isn't how the model gets
trained anymore. Three things in it were wrong for this problem.

**The split was random.** `step` is an hour counter over a 30 day simulation,
so a random split trains on hour 500 and tests on hour 200. No real fraud
system gets to see the future. `train.py` splits on time.

**Accuracy and F1 at a 0.5 threshold.** Only 0.129% of rows are fraud, so a
model that always answers "not fraud" is 99.87% accurate. The main metric is
now PR-AUC, which actually moves when the model improves.

**The imbalance was never handled.** The notebook says a workaround is needed
and then doesn't apply one. `train.py` tunes `scale_pos_weight` against
validation PR-AUC.

I also stopped using 0.5 as the cutoff. Missing a fraud costs the full
transaction amount and a false alarm costs one analyst review, which are
nowhere near equal, so `train.py` sweeps thresholds against a cost function
and picks the cheapest.

---

## Model comparison

From the notebook, trained on the original random split. I'm keeping this
because the comparison is still useful, but these numbers come from the split
I later replaced, so they don't line up with the results section above.

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| Accuracy | 99.90% | 99.97% | **99.98%** |
| Precision | 65.06% | **98.02%** | 96.60% |
| Recall | 48.63% | 78.59% | **85.98%** |
| F1 | 55.66% | 87.24% | **90.98%** |
| False negatives | 1,070 | 446 | **292** |

XGBoost won on recall and F1, which is what matters when the classes are this
imbalanced, so that's what the app uses.

---

## Architecture

```
  Streamlit dashboard                FastAPI service
  (Streamlit Cloud)      ------->    (Render)
  :8501                  POST        :8000
                         /predict         |
                                          v
                                   XGBoost model
                                   + SHAP explainer
                                          |
                                          v
                            probability, decision, top 5 factors
```

`backend/features.py` builds the features, and both training and serving
import it, so a feature can't get built one way during training and another
way at prediction time. There's a test that checks the two paths agree.

The model bundle also records which columns it was fitted on, and the API
serves exactly those.

---

## Input features

Eight fields go in:

| Field | Description |
|---|---|
| Step | Hour of the transaction |
| Transaction type | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Amount | Transaction amount |
| Sender balance before / after | Sender's balance either side of the transaction |
| Receiver balance before / after | Receiver's balance either side of the transaction |
| Flagged fraud | Rule based flag from the source system |

From those, the model builds five more:

| Derived feature | Why |
|---|---|
| Hour of day | `step` raw doesn't survive a time based split, since every test value sits outside the training range |
| Sender ledger mismatch | Balances should satisfy `new = old - amount`. Fraud rows often don't |
| Receiver ledger mismatch | Same on the receiving side |
| Sender emptied | Whether the account went to exactly zero |
| Amount vs sender balance | How much of the balance the transaction moved |

---

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Modelling | XGBoost, scikit-learn, pandas, NumPy |
| Explainability | SHAP |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Containers | Docker, Docker Compose |
| CI | GitHub Actions, pytest, ruff |
| Hosting | Render (API), Streamlit Community Cloud (dashboard) |

---

## Getting started

### Docker

```bash
docker compose up --build
```

Dashboard on <http://localhost:8501>, API docs on <http://localhost:8000/docs>.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt -r frontend/requirements.txt
```

Start the API:

```bash
cd backend && uvicorn app:app --reload
```

Start the dashboard in a second terminal:

```bash
cd frontend && API_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py
```

### Retraining

The trained model is committed, so you only need the dataset if you want to
retrain. Download [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
into `dataset/`, then:

```bash
cd backend && python train.py
python ../scripts/update_readme_metrics.py
```

Useful flags:

| Flag | Effect |
|---|---|
| `--review-cost 50` | Change what a false alarm costs. The threshold moves with it |
| `--feature-set realistic` | Drop the balance features that give the simulator away |
| `--compare-random-split` | Also train on a random split, to compare |
| `--nrows 500000` | Train on a slice, for a quick run |

---

## Tests

```bash
cd backend && python -m pytest
```

27 tests covering feature building, the time based split, the threshold sweep
against a brute force version, the rule baseline, and the API. GitHub Actions
runs these plus ruff on every push, and builds both Docker images.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8000` | Where the dashboard looks for the API |
| `MODEL_DIR` | `backend/model` | Where the model and metrics live |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per IP cap on `/predict`. In memory, single worker only |
| `LOG_LEVEL` | `INFO` | Backend logging |

On Streamlit Cloud there's no way to set environment variables, so the
dashboard also reads `API_URL` from secrets:

```toml
API_URL = "https://your-backend.onrender.com"
```

---

## Limitations

- **The data is simulated.** Everything above is a ceiling, not an estimate.
- **No account history.** Each transaction is scored on its own. A real system
  would have account age, recent activity, device fingerprints and who the
  money is going to, which is most of what actually catches fraud.
- **The review cost is made up.** $25 per alert is a placeholder. Change it and
  the operating point changes with it.
- **No fairness testing.** PaySim has no demographic columns, so there was
  nothing to test. Real data would need this before going near production.
- **Rate limiting is in process**, so it only holds with a single worker.
- **SHAP explanations degrade rather than fail.** If the explainer can't run,
  the API returns `available: false` with a reason instead of a number. An
  earlier version made up attribution values when SHAP was missing, which is
  worse than returning nothing.

---

## Project structure

```
backend/
  app.py             FastAPI service
  features.py        feature building, shared by training and serving
  train.py           training, evaluation, threshold selection
  model/             trained model and metrics
  tests/
frontend/
  streamlit_app.py   the dashboard, one page
notebook/            the original exploratory notebook
scripts/             regenerates the results section of this file
dataset/             where the Kaggle CSV goes
```

---

## About me

I'm Eklavya, and I built this while teaching myself machine learning.

I started it as a straightforward classification exercise and ended up
spending most of my time on the parts around the model: how to evaluate it
honestly, where to put the decision threshold, and why a suspiciously good
score usually means something is wrong with the data rather than right with
the model. Working that out was the most useful thing I got from the project,
and it's why the dataset problem is documented near the top of this README
instead of quietly left out.

I'm interested in applied machine learning, particularly problems where the
cost of being wrong isn't symmetric.

- GitHub: [@eklavya072](https://github.com/eklavya072)
- Email: eklavyasingh2106@gmail.com

Feedback and issues are welcome.

---

<p align="center">
  Dataset: <a href="https://www.kaggle.com/datasets/ealaxi/paysim1">PaySim1</a>,
  6,362,620 transactions, 8,213 of them fraud. Not committed, see <code>dataset/README.md</code>.
</p>
