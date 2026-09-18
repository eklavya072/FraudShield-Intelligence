"""Model card: how it was trained, how it scored, and where it falls over."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from fraudshield import api, ui

ui.topbar("Model card")

st.markdown(
    '<div class="page-title"><div><h1>Model <span>Card</span></h1>'
    "<p>Methodology, held-out performance and known limitations</p></div></div>",
    unsafe_allow_html=True,
)

if ui.connection_banner() is None:
    st.stop()

try:
    card = api.metrics()
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

test = card["test"]
econ = test["economics"]
split = card["split"]
threshold = card["threshold_selection"]

ui.metric_cards(
    [
        {
            "label": "PR-AUC (held out)",
            "value": f"{test['average_precision']:.3f}",
            "hint": "average precision",
        },
        {"label": "Recall", "value": f"{test['recall']:.1%}", "hint": "of fraudulent transactions"},
        {"label": "Precision", "value": f"{test['precision']:.1%}", "hint": "of raised alerts"},
        {
            "label": "Value caught",
            "value": f"{econ['value_detection_rate']:.1%}",
            "hint": ui.money(econ["fraud_value_caught"])
            + " of "
            + ui.money(econ["fraud_value_at_risk"]),
        },
        {
            "label": "Alerts / 1,000",
            "value": f"{test['alerts_per_1000']:.2f}",
            "hint": "analyst workload",
        },
        {
            "label": "Net savings",
            "value": ui.money(econ["net_savings"]),
            "hint": "vs. no model, on the test window",
            "tone": "good",
        },
    ]
)

st.markdown('<div class="panel-title">Why the cutoff is not 0.5</div>', unsafe_allow_html=True)
default = card.get("test_at_default_threshold_0.5", {})
ui.note(
    f"Alerts fire at <strong>{threshold['chosen_threshold']:.4f}</strong>. That comes "
    f"from minimising cost on the validation set, where a false alarm costs "
    f"<strong>${threshold['review_cost_per_alert']:,.0f}</strong> of review time and a "
    f"missed fraud costs the <strong>whole transaction amount</strong>. Those are very "
    f"different numbers, so 0.5 is an arbitrary place to draw the line. At 0.5 this "
    f"same model gets <strong>{default.get('recall', 0):.1%}</strong> recall and saves "
    f"<strong>{ui.money(default.get('economics', {}).get('net_savings', 0))}</strong>, "
    f"against <strong>{test['recall']:.1%}</strong> and "
    f"<strong>{ui.money(econ['net_savings'])}</strong> at the chosen cutoff. "
    + (
        f"What it costs is precision, down from "
        f"<strong>{default.get('precision', 0):.1%}</strong> to "
        f"<strong>{test['precision']:.1%}</strong>, which is worth it when the two "
        f"kinds of mistake are this far apart in cost."
        if default.get("precision", 0) - test["precision"] > 0.01
        else "On this data the classes separate cleanly enough that the lower cutoff "
        "barely costs any precision, so the two points end up almost the same. The "
        "logic still matters: on data where recall and precision actually trade off, "
        "this is what picks between them."
    )
)

curve = pd.DataFrame(threshold["cost_curve"])
if not curve.empty:
    st.line_chart(curve.set_index("threshold")["cost"], height=240)
    st.caption("Total cost at each candidate threshold, on the validation set.")

st.markdown(
    '<div class="panel-title">How much of this is real</div>',
    unsafe_allow_html=True,
)
rule = card.get("trivial_rule_baseline")
ablation = card.get("feature_set_ablation")
if rule:
    lift = card.get("model_lift_over_rule_f1", 0.0)
    verdict = (
        "the model adds basically nothing over the rule"
        if abs(lift) < 0.02
        else f"the model adds {lift:+.3f} F1 over the rule"
    )
    ui.note(
        "Worth reading alongside the numbers above. PaySim empties the sender's "
        "account whenever it generates a fraud, so "
        "<strong>97.7% of fraud rows have "
        "<code>oldbalanceOrg == amount and newbalanceOrig == 0</code>, and no "
        "legitimate row does</strong>. That two line rule on its own gets "
        f"<strong>{rule['precision']:.1%}</strong> precision and "
        f"<strong>{rule['recall']:.1%}</strong> recall on this same test set, an F1 of "
        f"<strong>{rule['f1']:.3f}</strong> against the model's "
        f"<strong>{test['f1']:.3f}</strong>. So on this data, {verdict}."
        + (
            "<br><br>Refitting without the three balance features that encode that "
            f"rule (<em>{ablation['feature_set']}</em> set) gives PR-AUC "
            f"<strong>{ablation['average_precision']:.3f}</strong>, precision "
            f"<strong>{ablation['precision']:.1%}</strong>, recall "
            f"<strong>{ablation['recall']:.1%}</strong>, and "
            f"{ablation['alerts_per_1000']:.1f} alerts per 1,000 rows instead of "
            f"{test['alerts_per_1000']:.1f}. That difference is how much of the "
            "headline number came from the simulator rather than from anything the "
            "model learned."
            if ablation
            else ""
        )
        + "<br><br>Putting this on the page rather than hiding it because a near "
        "perfect score on a public dataset is usually the dataset's doing. The parts "
        "that would still work on real transactions are the cost based cutoff, the "
        "expected loss queue and the drift check. The accuracy would not."
    )

st.markdown('<div class="panel-title">How it was evaluated</div>', unsafe_allow_html=True)
control = card.get("random_split_control")
gap = control["average_precision"] - test["average_precision"] if control else None
ui.note(
    f"Split on <strong>time</strong>: trained on hours up to "
    f"{split['train_max_step']}, validated through {split['val_max_step']}, tested on "
    f"everything after that. A random split would let the model train on hour 500 and "
    f"test on hour 200, which nothing in production ever gets to do."
    + (
        (
            f" The same model on a random split scores "
            f"<strong>{control['average_precision']:.4f}</strong> PR-AUC against "
            f"<strong>{test['average_precision']:.4f}</strong> here, so the random "
            f"version is <strong>{gap:+.4f}</strong> too optimistic."
            if gap > 0.001
            else f" The same model on a random split scores "
            f"<strong>{control['average_precision']:.4f}</strong> PR-AUC against "
            f"<strong>{test['average_precision']:.4f}</strong> here, a gap of "
            f"<strong>{gap:+.4f}</strong>. So the random split didn't inflate "
            "anything in this case. PaySim puts most of its fraud in the later "
            "hours, which are the ones the time based test set uses, so that window "
            "is denser in fraud and slightly easier. Splitting on time is still the "
            "right call, it just isn't where the optimism is here."
        )
        if control
        else ""
    )
    + f" Main metric is average precision rather than accuracy, because at a "
    f"{card['dataset']['fraud_rate']:.4%} fraud rate a model that always says no is "
    f"{1 - card['dataset']['fraud_rate']:.4%} accurate and useless."
)

left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        '<div class="panel-title">Confusion matrix (held out)</div>', unsafe_allow_html=True
    )
    matrix = test["confusion_matrix"]
    st.dataframe(
        pd.DataFrame(
            [
                [matrix["true_negative"], matrix["false_positive"]],
                [matrix["false_negative"], matrix["true_positive"]],
            ],
            index=["Actually legitimate", "Actually fraud"],
            columns=["Predicted clear", "Predicted alert"],
        ),
        use_container_width=True,
    )
with right:
    st.markdown('<div class="panel-title">Feature importance</div>', unsafe_allow_html=True)
    importances = card["model"]["feature_importances"]
    st.bar_chart(pd.Series(importances).sort_values(ascending=False).head(8), height=260)

st.markdown('<div class="panel-title">Limitations</div>', unsafe_allow_html=True)
ui.note(
    "<strong>The data is simulated.</strong> PaySim is a generator, not real "
    "transactions. Fraud in it is cleaner and easier to separate than anything real, "
    "so everything above is a ceiling rather than an estimate.<br><br>"
    "<strong>No account history.</strong> Every transaction is scored on its own. A "
    "real system would have account age, how often the account has been used recently, "
    "device information and who the money is going to, which is most of what actually "
    "catches fraud.<br><br>"
    "<strong>The review cost is made up.</strong> $"
    f"{threshold['review_cost_per_alert']:,.0f} per alert is a placeholder. Change it "
    "and the cutoff moves with it."
    "<br><br>"
    "<strong>No fairness testing.</strong> PaySim has no demographic columns so there "
    "was nothing to test. Real data would need this before going near production."
)

st.caption(
    f"Model trained {card['trained_at']} on {card['dataset']['rows']:,} transactions "
    f"({card['dataset']['fraud']:,} fraudulent) in "
    f"{card['training_seconds']:.0f}s · xgboost {card['environment']['xgboost']} · "
    f"scikit-learn {card['environment']['scikit_learn']}"
)
