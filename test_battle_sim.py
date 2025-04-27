import math
import pytest

# ---------------------------------------------------------------------------
# Adjust the import below so that it points to *your* implementation module.
# For example, if your function lives in `battle_sim.py`, do:
#     from battle_sim import win_probability
# ---------------------------------------------------------------------------
from battle_sim_numba import win_probability   # <-- change if needed

# ---------------------------------------------------------------------------
# Helper – keep the Monte‑Carlo run time sane but still accurate.
# 100 000 sims give <±0.5 % 1‑σ error on probabilities ~0.5.
# ---------------------------------------------------------------------------
DEFAULT_SIMS = 100_000
REL_TOL      = 0.02    # ±2 % of exact value is plenty for a unit‑test

# ---------------------------------------------------------------------------
# Base‑case scenarios derived analytically in the conversation
# ---------------------------------------------------------------------------
TEST_CASES = [
    # 1. Sudden‑death: one 1‑dmg cannon, hull = 1 each.
    #    P(A wins) = 3/4.
    (
        {
            "initiative": 5, "hull": 1, "computer": 2, "shield": 1,
            "missiles": [], "cannons": [1],
        },
        {
            "initiative": 4, "hull": 1, "computer": 1, "shield": 1,
            "missiles": [], "cannons": [1],
        },
        3/4,
        "sudden_death_1_cannon",
    ),
    # 2. 1‑missile vs 1 killer missile, hull = 2.  P(A) = 85 / 128.
    (
        {
            "initiative": 4, "hull": 2, "computer": 2, "shield": 1,
            "missiles": [1], "cannons": [1],
        },
        {
            "initiative": 5, "hull": 2, "computer": 1, "shield": 1,
            "missiles": [2], "cannons": [1],
        },
        85/128,
        "missile_vs_killer_missile",
    ),
    # 3. Two vs three 1‑dmg missiles, hull = 2.  P(A) = 23181 / 31104.
    (
        {
            "initiative": 5, "hull": 2, "computer": 2, "shield": 1,
            "missiles": [1, 1], "cannons": [1],
        },
        {
            "initiative": 4, "hull": 2, "computer": 1, "shield": 1,
            "missiles": [1, 1, 1], "cannons": [1],
        },
        23181/31104,
        "two_vs_three_missiles",
    ),
    # 4. Variable‑damage missiles + multi‑damage cannons, hull = 3.  P(A) = 233 948 607 / 275 365 888.
    (
        {
            "initiative": 6, "hull": 3, "computer": 2, "shield": 1,
            "missiles": [2, 1], "cannons": [2, 1],
        },
        {
            "initiative": 4, "hull": 3, "computer": 1, "shield": 1,
            "missiles": [2, 1, 1], "cannons": [1, 1],
        },
        233_948_607 / 275_365_888,
        "multi_damage_combo",
    ),
]

# ---------------------------------------------------------------------------
# Parametrised pytest – runs each scenario once.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "spec_a, spec_b, exact_p, _id",
    [(a, b, p, _id) for a, b, p, _id in TEST_CASES],
    ids=[_id for *_rest, _id in TEST_CASES],
)
def test_win_probability(spec_a, spec_b, exact_p, _id):
    """Monte‑Carlo estimate should be within REL_TOL of the analytical value."""
    est = win_probability(spec_a, spec_b, simulations=DEFAULT_SIMS, seed=123)
    assert math.isclose(est, exact_p, rel_tol=REL_TOL), (
        f"{_id}: estimate={est:.4%}, expected={exact_p:.4%}")


# ---------------------------------------------------------------------------
# Optional: mark a *slow* test that runs 1 000 000 sims to catch regressions
# in performance as well as accuracy.  Disabled by default (need -m slow).
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "spec_a, spec_b, exact_p, _id",
    [(a, b, p, _id) for a, b, p, _id in TEST_CASES],
    ids=[_id for *_rest, _id in TEST_CASES],
)
def test_win_probability_million_sims(spec_a, spec_b, exact_p, _id):
    est = win_probability(spec_a, spec_b, simulations=10_000_000, seed=456)
    assert math.isclose(est, exact_p, rel_tol=0.001), (
        f"{_id} (1M sims): estimate={est:.4%}, expected={exact_p:.4%}")
