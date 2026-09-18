"""Rewrites the results section of README.md from backend/model/metrics.json.

Saves me copying numbers by hand and forgetting to update them.
Run after training: python scripts/update_readme_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "backend" / "model" / "metrics.json"
README = ROOT / "README.md"

START = "<!-- METRICS:START -->"
END = "<!-- METRICS:END -->"


def money(value: float) -> str:
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"${value / limit:,.2f}{suffix}"
    return f"${value:,.0f}"


def render(card: dict) -> str:
    test = card["test"]
    econ = test["economics"]
    split = card["split"]
    threshold = card["threshold_selection"]
    default = card["test_at_default_threshold_0.5"]
    control = card.get("random_split_control")
    rule = card.get("trivial_rule_baseline")
    ablation = card.get("feature_set_ablation")

    lines = [
        f"Tested on the last {split['test_rows']:,} transactions (everything after "
        f"hour {split['val_max_step']}), which the model never saw while training or "
        f"while picking the threshold.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| PR-AUC (average precision) | {test['average_precision']:.4f} |",
        f"| Recall | {test['recall']:.1%} |",
        f"| Precision | {test['precision']:.1%} |",
        f"| ROC-AUC | {test['roc_auc']:.4f} |",
        f"| Alerts per 1,000 transactions | {test['alerts_per_1000']:.2f} |",
        f"| Fraud value caught | {econ['value_detection_rate']:.1%} "
        f"({money(econ['fraud_value_caught'])} of "
        f"{money(econ['fraud_value_at_risk'])}) |",
        f"| Savings vs. no model | {money(econ['net_savings'])} |",
        "",
        f"The alert cutoff is {threshold['chosen_threshold']:.4f}, not 0.5. It comes "
        f"from minimising cost on the validation set, where a false alarm costs "
        f"${threshold['review_cost_per_alert']:,.0f} of review time and a missed "
        f"fraud costs the full amount. At 0.5 the same model gets "
        f"{default['recall']:.1%} recall and saves "
        f"{money(default['economics']['net_savings'])}, against "
        f"{test['recall']:.1%} and {money(econ['net_savings'])} at the chosen point.",
    ]

    if rule:
        lift = card.get("model_lift_over_rule_f1", 0.0)
        lines += [
            "",
            "How much of this is real: the two line rule "
            "`oldbalanceOrg == amount and newbalanceOrig == 0` on its own gets "
            f"{rule['precision']:.1%} precision and {rule['recall']:.1%} recall on the "
            f"same test set, F1 of {rule['f1']:.3f}. The model gets "
            f"{test['f1']:.3f}, so it adds {lift:+.3f}.",
        ]
        if ablation:
            lines += [
                "",
                "Dropping the three balance features that encode that rule "
                f"(`--feature-set {ablation['feature_set']}`) gives PR-AUC "
                f"{ablation['average_precision']:.4f}, precision "
                f"{ablation['precision']:.1%}, recall {ablation['recall']:.1%}, and "
                f"{ablation['alerts_per_1000']:.1f} alerts per 1,000 rows instead of "
                f"{test['alerts_per_1000']:.1f}. That difference is how much of the "
                "headline number came from the simulator rather than from anything "
                "the model learned.",
            ]

    if control:
        gap = control["average_precision"] - test["average_precision"]
        if gap > 0.001:
            verdict = (
                f"{gap:+.4f} higher, which is the usual result and the reason not to split randomly"
            )
        else:
            verdict = (
                f"{gap:+.4f}, so the random split didn't inflate anything here. PaySim "
                "puts most of its fraud in the later hours, which are the ones the "
                "time based test set uses, so that window is denser in fraud and a bit "
                "easier. The time based split is still the right one, it just isn't "
                "where the optimism is on this dataset"
            )
        lines += [
            "",
            f"Same model trained on a random split instead scores "
            f"{control['average_precision']:.4f} PR-AUC against "
            f"{test['average_precision']:.4f} here: {verdict}. "
            "Run `python train.py --compare-random-split` to reproduce.",
        ]

    lines += [
        "",
        f"<sub>Trained on {card['dataset']['rows']:,} rows "
        f"({card['dataset']['fraud']:,} fraud) in "
        f"{card['training_seconds']:.0f}s, xgboost "
        f"{card['environment']['xgboost']}, scikit-learn "
        f"{card['environment']['scikit_learn']}, scale_pos_weight "
        f"{card['model']['scale_pos_weight']:.0f}</sub>",
    ]
    return "\n".join(lines)


def main() -> None:
    if not METRICS.exists():
        raise SystemExit(f"{METRICS} not found. Run backend/train.py first.")

    readme = README.read_text()
    if START not in readme or END not in readme:
        raise SystemExit(f"README.md is missing the {START} / {END} markers.")

    before = readme.split(START)[0]
    after = readme.split(END)[1]
    block = render(json.loads(METRICS.read_text()))
    README.write_text(f"{before}{START}\n{block}\n{END}{after}")
    print(f"Updated the metrics block in {README}")


if __name__ == "__main__":
    main()
