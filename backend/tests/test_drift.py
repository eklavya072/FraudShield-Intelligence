import numpy as np
from drift import band, compute, psi


def _profile():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 10_000)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    counts, _ = np.histogram(values, bins=edges)
    return {
        "features": {
            "amount": {
                "edges": [float(e) for e in edges],
                "proportions": [float(c) for c in counts / counts.sum()],
                "mean": 0.0,
            }
        },
        "score": {
            "edges": list(np.linspace(0, 1, 11)),
            "proportions": [0.1] * 10,
        },
    }


def test_identical_distribution_has_near_zero_psi():
    profile = _profile()["features"]["amount"]
    rng = np.random.default_rng(1)
    assert psi(profile["proportions"], rng.normal(0, 1, 10_000), profile["edges"]) < 0.1


def test_shifted_distribution_is_detected():
    profile = _profile()["features"]["amount"]
    rng = np.random.default_rng(1)
    shifted = psi(profile["proportions"], rng.normal(3, 1, 10_000), profile["edges"])
    assert shifted > 0.25
    assert band(shifted) == "significant"


def test_bands():
    assert band(0.05) == "stable"
    assert band(0.15) == "moderate"
    assert band(0.9) == "significant"


def test_compute_says_nothing_on_thin_traffic():
    rows = [{"amount": 1.0, "_score": 0.1} for _ in range(10)]
    result = compute(_profile(), rows, min_rows=100)

    assert result["sufficient_data"] is False
    assert result["features"] == []
    assert result["overall"] is None


def test_compute_ranks_the_worst_drifting_feature_first():
    rng = np.random.default_rng(2)
    rows = [
        {"amount": float(v), "_score": float(s)}
        for v, s in zip(rng.normal(4, 1, 500), rng.random(500), strict=True)
    ]
    result = compute(_profile(), rows)

    assert result["sufficient_data"] is True
    assert result["features"][0]["feature"] == "amount"
    assert result["features"][0]["band"] == "significant"
    assert result["overall"]["max_feature_psi"] == result["features"][0]["psi"]
