"""PSI between the training data and whatever the model is seeing now.

Nothing in a prediction tells you the model has gone stale, so this compares
each feature's current distribution against the one it was fitted on.

Usual reading: under 0.1 stable, 0.1 to 0.25 some shift, over 0.25 worth
retraining.
"""

from __future__ import annotations

import numpy as np

STABLE = 0.1
MODERATE = 0.25
_EPSILON = 1e-6


def band(psi: float) -> str:
    if psi < STABLE:
        return "stable"
    if psi < MODERATE:
        return "moderate"
    return "significant"


def psi(reference_proportions, live_values, edges) -> float:
    """PSI of `live_values` against reference proportions over fixed `edges`."""
    live_values = np.asarray(live_values, dtype="float64")
    if live_values.size == 0 or len(edges) < 2:
        return 0.0

    counts, _ = np.histogram(live_values, bins=edges)
    live = counts / max(counts.sum(), 1)
    reference = np.asarray(reference_proportions, dtype="float64")

    if live.shape != reference.shape:
        return 0.0

    live = np.clip(live, _EPSILON, None)
    reference = np.clip(reference, _EPSILON, None)
    return float(np.sum((live - reference) * np.log(live / reference)))


def compute(profile: dict, live_rows: list[dict], min_rows: int = 100) -> dict:
    """PSI per feature, plus one for the score distribution.

    Returns sufficient_data: False on thin traffic instead of a zero. Saying
    "stable" off 12 rows would be worse than saying nothing.
    """
    if len(live_rows) < min_rows:
        return {
            "sufficient_data": False,
            "n_observations": len(live_rows),
            "min_observations": min_rows,
            "features": [],
            "score_psi": None,
            "overall": None,
        }

    features = []
    for name, spec in profile.get("features", {}).items():
        values = [row[name] for row in live_rows if name in row]
        value = psi(spec["proportions"], values, spec["edges"])
        live_mean = float(np.mean(values)) if values else 0.0
        features.append(
            {
                "feature": name,
                "psi": round(value, 4),
                "band": band(value),
                "reference_mean": round(spec.get("mean", 0.0), 4),
                "live_mean": round(live_mean, 4),
            }
        )
    features.sort(key=lambda item: item["psi"], reverse=True)

    score_spec = profile.get("score", {})
    scores = [row["_score"] for row in live_rows if "_score" in row]
    score_psi = (
        round(psi(score_spec["proportions"], scores, score_spec["edges"]), 4)
        if score_spec and scores
        else None
    )

    worst = features[0]["psi"] if features else 0.0
    return {
        "sufficient_data": True,
        "n_observations": len(live_rows),
        "min_observations": min_rows,
        "features": features,
        "score_psi": score_psi,
        "score_band": band(score_psi) if score_psi is not None else None,
        "overall": {"max_feature_psi": worst, "band": band(worst)},
    }
