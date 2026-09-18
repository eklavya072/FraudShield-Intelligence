# FraudShield Intelligence

Fraud detection on the PaySim dataset, wrapped in an actual app instead of
just a notebook. XGBoost model behind a FastAPI service, with a Streamlit
dashboard on top for scoring transactions, working through alerts, batch
scoring a file, and checking for drift.

I started this as a Kaggle-style notebook project and kept going, mostly
because the notebook version kept giving me results that felt too good. More
on that below.

Note that PaySim is simulated data, not real transactions, so treat the
numbers here as a best case.

## Things I changed after the first version

The notebook is still in `notebook/` but it isn't how the model gets trained
anymore. Three things in it were wrong for this problem.

**The train/test split was random.** The `step` column is an hour counter over
a 30 day simulation, so a random split trains on hour 500 and tests on hour
200. No real fraud system gets to see the future. `train.py` splits on time
instead.

**Accuracy and F1 at a 0.5 threshold.** Only 0.13% of the rows are fraud, so a
model that always says "not fraud" is 99.87% accurate. I switched the main
metric to PR-AUC (average precision), which actually moves when the model gets
better.

**The imbalance never got handled.** The notebook says a workaround is needed
and then doesn't apply one. `train.py` tunes `scale_pos_weight` against
validation PR-AUC.

I also stopped using 0.5 as the cutoff. Missing a fraud costs you the whole
transaction amount, a false alarm costs one analyst review, and those aren't
close to equal. `train.py` sweeps thresholds against a cost function and picks
the cheapest one.

The same idea carries into the app: the review queue sorts alerts by expected
loss (probability x amount) rather than by probability, so a 60% chance on
$80,000 gets looked at before a 98% chance on $40.

## The problem with this dataset

This is the part I'd want someone to read.

My first proper training run came back with 100% precision and 100% recall,
which is not a thing that happens. So I went looking for why.

PaySim creates a fraudulent transaction by emptying the sender's account.
Across all 6.3 million rows:

| | `oldbalanceOrg == amount` and `newbalanceOrig == 0` |
| --- | --- |
| Fraud | 97.7% |
| Not fraud | 0.0% |

So you can separate the classes with a two line if statement. Any model that
gets features built on that will look perfect without having learned anything
about fraud.

`train.py` now reports that rule as a baseline next to the model, and refits
without those features so you can see the difference. Both numbers are below.

I'm leaving this at the top rather than in a limitations section because a
near perfect score on a public dataset is usually a property of the dataset.
The parts of this project that would survive contact with real data are the
threshold logic, the queue, and the drift monitoring, not the accuracy.

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

## How it fits together

```
Streamlit dashboard (:8501)          FastAPI service (:8000)
  Risk console                 --->    POST /predict
  Review queue                         POST /predict/batch
  Batch scoring                        GET  /alerts
  Drift monitor                        POST /alerts/{id}/decision
  Model card                           GET  /drift, /metrics, /health
                                              |
                                   XGBoost + SHAP
                                   SQLite alert queue
                                   PSI reference profile
```

`backend/features.py` builds the features, and training, single scoring and
batch scoring all import it. That way a feature can't get built one way during
training and another way at serving time. There's a test for it.

## Running it

With Docker:

```bash
docker compose up --build
```

Dashboard on <http://localhost:8501>, API docs on
<http://localhost:8000/docs>.

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt -r frontend/requirements.txt
```

The trained model is committed, so you only need the dataset if you want to
retrain. If you do, download [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
into `dataset/` and run:

```bash
cd backend && python train.py
```

Then start both services:

```bash
cd backend && uvicorn app:app --reload
```

```bash
cd frontend && API_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py
```

The dashboard gets its endpoint from `API_URL`.

## Tests

```bash
cd backend && python -m pytest
```

```bash
cd frontend && python -m pytest
```

53 tests. GitHub Actions runs them plus ruff on every push, and builds both
Docker images.

## Settings

| Variable | Default | What it does |
| --- | --- | --- |
| `API_URL` | `http://127.0.0.1:8000` | Where the dashboard looks for the API |
| `FRAUDSHIELD_API_KEY` | unset | If set, calls need an `X-API-Key` header |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per IP limit. In memory, so single instance only |
| `MAX_BATCH_SIZE` | `5000` | Row limit on `/predict/batch` |
| `MODEL_DIR` | `backend/model` | Model, metrics and profile location |
| `FRAUDSHIELD_DB` | `backend/model/fraudshield.db` | Alert database |
| `LOG_LEVEL` | `INFO` | Backend logging |

If your review costs aren't $25, retrain with
`python train.py --review-cost 50` and the threshold moves with it.

## A note on the SHAP explanations

`/predict` returns SHAP values when it can, and `"available": false` with a
reason when it can't. The first version of this had a fallback that made up
attribution numbers from hardcoded constants whenever SHAP failed to import,
and returned them in the same format as the real ones. That's worse than
returning nothing, because you can't tell them apart and you'd act on them.
There's a test covering it now.

## What this doesn't do

- The data is simulated, so the numbers are a ceiling, not an estimate.
- Every transaction is scored on its own. No account history, no velocity
  features, no device or counterparty information, which is most of what real
  fraud detection actually runs on.
- The $25 review cost is made up. Change it and the operating point changes.
- No fairness testing. PaySim has no demographic columns, so there was nothing
  to test. Real data would need this before going anywhere near production.
- Rate limiting is in process, so it only works with one worker. Redis if you
  run more.
- PSI catches distribution shift, not accuracy drops. Real labels come back
  weeks later from chargebacks. The realised precision number on the queue
  page is the closest thing here to a live signal.

## Files

```
backend/
  features.py    feature building, shared by everything
  train.py       training, evaluation, threshold selection
  app.py         FastAPI service
  store.py       SQLite alert queue
  drift.py       PSI calculation
  tests/
frontend/
  streamlit_app.py   entry point
  fraudshield/       theme, API client, shared bits
  pages/             the five dashboard pages
  tests/
notebook/        the original notebook
scripts/         regenerates the results section of this file
```

Dataset: [PaySim1](https://www.kaggle.com/datasets/ealaxi/paysim1), 6,362,620
transactions, 8,213 of them fraud. Not committed, see `dataset/README.md`.
