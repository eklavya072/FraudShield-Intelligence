"""Trains and evaluates the model.

Differences from the notebook version:

1. Splits on time, not randomly. `step` is an hour counter, so a random split
   trains on hour 500 and tests on hour 200. --compare-random-split shows what
   that does to the numbers.
2. Reports PR-AUC rather than accuracy. At 0.13% fraud both accuracy and
   ROC-AUC sit near the top of their range and don't tell you much.
3. Picks the threshold by cost instead of leaving it at 0.5. A missed fraud
   costs the transaction amount, a false alarm costs one review.

    python train.py --data ../dataset/PS_20174392719_1491204439457_log.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from features import DEGENERATE_FEATURES, FEATURE_SETS, build_features, load_raw_csv
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).parent / "model"

# What it costs to send one transaction to a human: their time plus annoying a
# real customer. Made up number, change it for your own setup.
DEFAULT_REVIEW_COST = 25.0


def temporal_split(frame: pd.DataFrame, train_frac: float, val_frac: float):
    """Splits on `step` so validation and test are always later than training.

    Cutoffs come from row counts, not step values. Step is skewed badly (94% of
    rows are at or below step 446), so splitting on the values themselves left
    me with a test set of under 2%.
    """
    train_end = int(frame["step"].quantile(train_frac))
    val_end = int(frame["step"].quantile(train_frac + val_frac))
    if val_end <= train_end:
        val_end = train_end + 1

    train = frame[frame["step"] <= train_end]
    val = frame[(frame["step"] > train_end) & (frame["step"] <= val_end)]
    test = frame[frame["step"] > val_end]
    return train, val, test, int(train_end), int(val_end)


def expected_cost(y_true, scores, amounts, threshold, review_cost):
    """What it costs to run at `threshold`.

    Every alert costs a review either way. Every fraud we miss costs the amount.
    """
    flagged = scores >= threshold
    missed_fraud = (y_true == 1) & ~flagged
    return float(flagged.sum() * review_cost + amounts[missed_fraud].sum())


def choose_threshold(y_true, scores, amounts, review_cost):
    """Tries every threshold and keeps the cheapest.

    Done with a sort and a searchsorted instead of recomputing the cost per row
    for each of ~1000 candidates. On 1.3M validation rows the naive version is
    over a billion operations and was taking longer than the actual training.
    test_fast_threshold_sweep_matches_brute_force checks they agree.
    """
    candidates = np.unique(np.round(np.linspace(0.001, 0.999, 999), 4))

    order = np.argsort(scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    sorted_amounts = np.asarray(amounts)[order]
    sorted_truth = np.asarray(y_true)[order]

    # Running total of fraud value below each position, so the amount missed at
    # a threshold is one lookup.
    fraud_value = np.where(sorted_truth == 1, sorted_amounts, 0.0)
    cumulative_missed = np.concatenate([[0.0], np.cumsum(fraud_value)])

    first_flagged = np.searchsorted(sorted_scores, candidates, side="left")
    n_flagged = len(sorted_scores) - first_flagged
    costs = review_cost * n_flagged + cumulative_missed[first_flagged]

    best = int(np.argmin(costs))
    curve = [
        {"threshold": float(t), "cost": float(c)}
        for t, c in zip(candidates[::20], costs[::20], strict=True)
    ]
    return float(candidates[best]), float(costs[best]), curve


def evaluate(y_true, scores, amounts, threshold, review_cost) -> dict:
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fraud_amount = float(amounts[y_true == 1].sum())
    caught_amount = float(amounts[(y_true == 1) & (y_pred == 1)].sum())
    model_cost = expected_cost(y_true, scores, amounts, threshold, review_cost)

    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n_transactions": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        "fraud_rate": float(y_true.mean()),
        "alert_rate": float(y_pred.mean()),
        "alerts_per_1000": float(y_pred.mean() * 1000),
        "economics": {
            "review_cost_per_alert": review_cost,
            "fraud_value_at_risk": fraud_amount,
            "fraud_value_caught": caught_amount,
            "fraud_value_missed": fraud_amount - caught_amount,
            "value_detection_rate": (float(caught_amount / fraud_amount) if fraud_amount else 0.0),
            "review_spend": float(y_pred.sum() * review_cost),
            "total_cost_with_model": model_cost,
            "total_cost_no_model": fraud_amount,
            "net_savings": float(fraud_amount - model_cost),
        },
    }


def trivial_rule_baseline(frame: pd.DataFrame) -> dict:
    """Scores `oldbalanceOrg == amount and newbalanceOrig == 0` as if it were a model.

    This is what any model here has to beat, and it's a high bar for reasons
    that have nothing to do with fraud: the simulator empties the sender's
    account whenever it makes a fraudulent row. Without this comparison the
    model's numbers look like a result when they're mostly an artifact.
    """
    y_true = frame["isfraud"].to_numpy()
    y_pred = (
        ((frame["oldbalanceorig"] == frame["amount"]) & (frame["newbalanceorig"] == 0))
        .to_numpy()
        .astype(int)
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "rule": "oldbalanceOrg == amount and newbalanceOrig == 0",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "alert_rate": float(y_pred.mean()),
    }


def fit_model(X_train, y_train, scale_pos_weight, seed):
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def build_reference_profile(X_train: pd.DataFrame, train_scores: np.ndarray) -> dict:
    """Decile edges from the training data, so PSI has something to compare to."""
    profile = {"features": {}}
    for column in X_train.columns:
        values = X_train[column].to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        counts, _ = np.histogram(values, bins=edges)
        profile["features"][column] = {
            "edges": [float(e) for e in edges],
            "proportions": [float(c) for c in counts / max(counts.sum(), 1)],
            "mean": float(values.mean()),
        }

    score_edges = np.linspace(0, 1, 11)
    score_counts, _ = np.histogram(train_scores, bins=score_edges)
    profile["score"] = {
        "edges": [float(e) for e in score_edges],
        "proportions": [float(c) for c in score_counts / max(score_counts.sum(), 1)],
    }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../dataset/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--nrows", type=int, default=None, help="cap rows (for CI)")
    parser.add_argument("--review-cost", type=float, default=DEFAULT_REVIEW_COST)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--compare-random-split", action="store_true", default=True)
    parser.add_argument(
        "--no-compare-random-split", dest="compare_random_split", action="store_false"
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="full",
        help=(
            "`full` includes the ledger-identity features that PaySim makes "
            "trivially separable; `realistic` drops them to measure what the "
            "model learns without reading the simulator's generation rule."
        ),
    )
    parser.add_argument("--out", default=str(MODEL_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print(f"Loading {args.data} ...")
    raw = load_raw_csv(args.data, nrows=args.nrows)
    print(
        f"  {len(raw):,} transactions, {int(raw['isfraud'].sum()):,} fraudulent "
        f"({raw['isfraud'].mean():.4%})"
    )

    train, val, test, train_end, val_end = temporal_split(raw, args.train_frac, args.val_frac)
    print(f"Temporal split on `step`: train <= {train_end}, val <= {val_end}, test > {val_end}")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s} {len(part):>9,} rows  {int(part['isfraud'].sum()):>6,} fraud")

    columns = FEATURE_SETS[args.feature_set]
    print(f"Feature set: {args.feature_set} ({len(columns)} features)")
    if args.feature_set == "full":
        print(f"  includes ledger-identity features: {', '.join(DEGENERATE_FEATURES)}")

    X_train, y_train = build_features(train, columns), train["isfraud"].to_numpy()
    X_val, y_val = build_features(val, columns), val["isfraud"].to_numpy()
    X_test, y_test = build_features(test, columns), test["isfraud"].to_numpy()
    amt_val = val["amount"].to_numpy()
    amt_test = test["amount"].to_numpy()

    # The imbalance handling the notebook mentioned and never did. Picked on
    # validation PR-AUC rather than guessed.
    imbalance = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    grid = sorted(
        {1.0, round(np.sqrt(imbalance), 2), round(imbalance / 10, 2), round(imbalance, 2)}
    )
    print(f"\nTuning scale_pos_weight (dataset imbalance ratio {imbalance:,.0f}:1)")

    best = None
    tuning = []
    for spw in grid:
        model = fit_model(X_train, y_train, spw, args.seed)
        val_scores = model.predict_proba(X_val)[:, 1]
        ap = float(average_precision_score(y_val, val_scores))
        tuning.append({"scale_pos_weight": float(spw), "val_average_precision": ap})
        print(f"  scale_pos_weight={spw:<10} val PR-AUC={ap:.4f}")
        if best is None or ap > best["ap"]:
            best = {"spw": spw, "ap": ap, "model": model, "val_scores": val_scores}

    model, val_scores = best["model"], best["val_scores"]
    print(f"  -> selected scale_pos_weight={best['spw']} (val PR-AUC {best['ap']:.4f})")

    threshold, val_cost, cost_curve = choose_threshold(y_val, val_scores, amt_val, args.review_cost)
    print(
        f"\nCost-optimal threshold on validation: {threshold:.4f} "
        f"(cost ${val_cost:,.0f} vs ${amt_val[y_val == 1].sum():,.0f} unprotected)"
    )

    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_scores, amt_test, threshold, args.review_cost)
    at_half = evaluate(y_test, test_scores, amt_test, 0.5, args.review_cost)

    print("\n--- Held-out test (future hours the model never saw) ---")
    print(f"  PR-AUC (average precision) : {test_metrics['average_precision']:.4f}")
    print(f"  ROC-AUC                    : {test_metrics['roc_auc']:.4f}")
    print(
        f"  Precision / Recall / F1    : {test_metrics['precision']:.3f} / "
        f"{test_metrics['recall']:.3f} / {test_metrics['f1']:.3f}"
    )
    print(f"  Alerts per 1,000 txns      : {test_metrics['alerts_per_1000']:.2f}")
    econ = test_metrics["economics"]
    print(
        f"  Fraud value caught         : ${econ['fraud_value_caught']:,.0f} of "
        f"${econ['fraud_value_at_risk']:,.0f} ({econ['value_detection_rate']:.1%})"
    )
    print(f"  Net savings vs no model    : ${econ['net_savings']:,.0f}")
    print(
        f"  (at default 0.5 threshold  : recall {at_half['recall']:.3f}, "
        f"net savings ${at_half['economics']['net_savings']:,.0f})"
    )

    rule = trivial_rule_baseline(test)
    print("\n--- Control: a two-line rule, no model at all ---")
    print(f"  {rule['rule']}")
    print(
        f"  Precision / Recall / F1    : {rule['precision']:.3f} / "
        f"{rule['recall']:.3f} / {rule['f1']:.3f}"
    )
    lift = test_metrics["f1"] - rule["f1"]
    print(f"  Model F1 minus rule F1     : {lift:+.4f}")
    if lift < 0.02:
        print("  NOTE: the model barely beats the rule. On PaySim this is expected --")
        print("        the simulator drains the sender's account exactly when it")
        print("        generates fraud, so the classes are separable by construction.")

    # Refit on the other feature set, so the model card can show what happens
    # without the balance features that give the simulator away.
    other = "realistic" if args.feature_set == "full" else "full"
    other_columns = FEATURE_SETS[other]
    print(f"\n--- Ablation: feature set '{other}' ({len(other_columns)} features) ---")
    ablation_model = fit_model(
        build_features(train, other_columns), y_train, best["spw"], args.seed
    )
    ablation_val = ablation_model.predict_proba(build_features(val, other_columns))[:, 1]
    ablation_threshold, _, _ = choose_threshold(y_val, ablation_val, amt_val, args.review_cost)
    ablation_scores = ablation_model.predict_proba(build_features(test, other_columns))[:, 1]
    ablation_metrics = evaluate(
        y_test, ablation_scores, amt_test, ablation_threshold, args.review_cost
    )
    ablation_metrics["feature_set"] = other
    print(f"  PR-AUC                     : {ablation_metrics['average_precision']:.4f}")
    print(
        f"  Precision / Recall / F1    : {ablation_metrics['precision']:.3f} / "
        f"{ablation_metrics['recall']:.3f} / {ablation_metrics['f1']:.3f}"
    )

    random_split_metrics = None
    if args.compare_random_split:
        # Redo it the wrong way on purpose, to see how much difference it makes.
        from sklearn.model_selection import train_test_split

        print("\n--- Control: the same model under a random split ---")
        X_all, y_all = build_features(raw, columns), raw["isfraud"].to_numpy()
        Xr_tr, Xr_te, yr_tr, yr_te, _, amt_r_te = train_test_split(
            X_all,
            y_all,
            raw["amount"].to_numpy(),
            test_size=0.25,
            random_state=args.seed,
            stratify=y_all,
        )
        rmodel = fit_model(Xr_tr, yr_tr, best["spw"], args.seed)
        r_scores = rmodel.predict_proba(Xr_te)[:, 1]
        random_split_metrics = evaluate(yr_te, r_scores, amt_r_te, threshold, args.review_cost)
        print(f"  PR-AUC under random split  : {random_split_metrics['average_precision']:.4f}")
        print(f"  PR-AUC under temporal split: {test_metrics['average_precision']:.4f}")
        inflation = random_split_metrics["average_precision"] - test_metrics["average_precision"]
        print(f"  Optimism from random split : {inflation:+.4f} PR-AUC")

    train_scores = model.predict_proba(X_train)[:, 1]
    profile = build_reference_profile(X_train, train_scores)

    bundle = {
        "model": model,
        "threshold": threshold,
        "feature_names": columns,
        "trained_at": datetime.now(UTC).isoformat(),
        "review_cost": args.review_cost,
    }
    joblib.dump(bundle, out_dir / "fraud_model.joblib")

    importances = dict(
        sorted(
            zip(columns, (float(v) for v in model.feature_importances_), strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    metrics = {
        "trained_at": bundle["trained_at"],
        "training_seconds": round(time.perf_counter() - started, 1),
        "dataset": {
            "path": args.data,
            "rows": int(len(raw)),
            "fraud": int(raw["isfraud"].sum()),
            "fraud_rate": float(raw["isfraud"].mean()),
            "source": "Kaggle PaySim1 (simulated mobile money transactions)",
        },
        "split": {
            "strategy": "temporal on `step` (hour index)",
            "train_max_step": train_end,
            "val_max_step": val_end,
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        },
        "model": {
            "algorithm": "XGBClassifier",
            "scale_pos_weight": float(best["spw"]),
            "params": model.get_params(),
            "feature_names": columns,
            "feature_importances": importances,
        },
        "threshold_selection": {
            "method": "minimise expected cost on validation",
            "review_cost_per_alert": args.review_cost,
            "missed_fraud_cost": "full transaction amount",
            "chosen_threshold": threshold,
            "validation_cost": val_cost,
            "cost_curve": cost_curve,
        },
        "tuning": tuning,
        "test": test_metrics,
        "test_at_default_threshold_0.5": at_half,
        "trivial_rule_baseline": rule,
        "model_lift_over_rule_f1": round(test_metrics["f1"] - rule["f1"], 4),
        "feature_set": args.feature_set,
        "feature_set_ablation": ablation_metrics,
        "random_split_control": random_split_metrics,
        "environment": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out_dir / "reference_profile.json").write_text(json.dumps(profile, indent=2))

    print(f"\nWrote {out_dir / 'fraud_model.joblib'}, metrics.json, reference_profile.json")
    print(f"Total {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
