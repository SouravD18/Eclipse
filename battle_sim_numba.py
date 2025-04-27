"""
fast_battle.py
==============

High-performance Monte-Carlo simulator for the Eclipse-style ship duel.

• Pure-Python wrapper  – always available
• Numba JIT core      – auto-enabled when numba is installed
• Automatically falls back to the slower (but still vectorised) pure-Python
  engine if numba is missing or you’re on PyPy, etc.

---------------------------------------------------------------------------
Public API
----------
win_probability(spec_a: dict, spec_b: dict, simulations=100_000, seed=None)
    → float   # P(ship-A wins)

Ship specification dict
-----------------------
{
    "initiative": int,
    "hull":        int,          # starting HP  (0–6 in original rules)
    "computer":    int,
    "shield":      int,
    "missiles":   [int, …],      # list of damage values (one-shot each)
    "cannons":    [int, …],      # list of damage values (fires every turn)
}
"""

from __future__ import annotations
import os
import math
import random
from typing import List, Sequence

try:
    import numba as nb
    import numpy as np

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:       # pragma: no cover
    NUMBA_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────
# Internal helpers – compiled version
# ──────────────────────────────────────────────────────────────────────────

if NUMBA_AVAILABLE:
    # constant dice mapping for JIT function
    _VALUES = np.array([-1, 2, 3, 4, 5, -2], dtype=np.int8)
    #   -1 → “No damage”     (always miss)
    #   -2 → “Full hit”      (always hit)

    @nb.njit(inline="always", fastmath=True)
    def _roll_hit(comp: int, shield: int) -> bool:
        """Return True if a single shot hits."""
        r = np.random.randint(0, 6)        # 0 … 5
        val = _VALUES[r]
        if val == -1:                      # "No damage"
            return False
        if val == -2:                      # "Full hit"
            return True
        return val + comp - shield >= 6

    @nb.njit(inline="always", fastmath=True)
    def _fire_sequence(dmg_arr,  # 1-D int32 array
                       comp: int,
                       enemy_shield: int,
                       enemy_hp: int) -> int:
        """Apply an entire missile/cannon list; return new enemy_hp (≥ 0)."""
        for dmg in dmg_arr:
            if _roll_hit(comp, enemy_shield):
                enemy_hp -= dmg
                if enemy_hp <= 0:
                    return enemy_hp
        return enemy_hp

    @nb.njit(fastmath=True)
    def _battle_once(a_init, a_hp0, a_comp, a_shield,
                     a_miss, a_can,
                     b_init, b_hp0, b_comp, b_shield,
                     b_miss, b_can) -> int:
        """
        Simulate one duel.

        Returns
        -------
        1  → A wins
        0  → B wins   (draws are impossible with current rules)
        """
        a_hp = a_hp0
        b_hp = b_hp0

        # who fires first
        a_first = a_init > b_init

        # ---------- Missile phase ----------
        if a_first:
            b_hp = _fire_sequence(a_miss, a_comp, b_shield, b_hp)
            if b_hp <= 0:
                return 1
            a_hp = _fire_sequence(b_miss, b_comp, a_shield, a_hp)
            if a_hp <= 0:
                return 0
        else:
            a_hp = _fire_sequence(b_miss, b_comp, a_shield, a_hp)
            if a_hp <= 0:
                return 0
            b_hp = _fire_sequence(a_miss, a_comp, b_shield, b_hp)
            if b_hp <= 0:
                return 1

        # ---------- Cannon phase ----------
        while True:
            # A fires (if has initiative) or B fires depending on a_first
            if a_first:
                b_hp = _fire_sequence(a_can, a_comp, b_shield, b_hp)
                if b_hp <= 0:
                    return 1
                a_hp = _fire_sequence(b_can, b_comp, a_shield, a_hp)
                if a_hp <= 0:
                    return 0
            else:
                a_hp = _fire_sequence(b_can, b_comp, a_shield, a_hp)
                if a_hp <= 0:
                    return 0
                b_hp = _fire_sequence(a_can, a_comp, b_shield, b_hp)
                if b_hp <= 0:
                    return 1

    @nb.njit(parallel=True, fastmath=True)
    def _batch_simulate(n,
                        a_init, a_hp, a_comp, a_shield,
                        a_miss, a_can,
                        b_init, b_hp, b_comp, b_shield,
                        b_miss, b_can) -> int:
        wins = 0
        for _ in nb.prange(n):
            wins += _battle_once(a_init, a_hp, a_comp, a_shield,
                                 a_miss, a_can,
                                 b_init, b_hp, b_comp, b_shield,
                                 b_miss, b_can)
        return wins

# ──────────────────────────────────────────────────────────────────────────
# Pure-Python fallback  (≈ 8–10 × slower)
# ──────────────────────────────────────────────────────────────────────────

def _roll_hit_py(comp: int, shield: int) -> bool:
    r = random.randint(0, 5)
    if r == 0:                 # "No damage"
        return False
    if r == 5:                 # "Full hit"
        return True
    val = r + 1               # 1→2, 4→5
    return val + comp - shield >= 6

def _fire_sequence_py(damage_seq: Sequence[int],
                      comp: int,
                      enemy_shield: int,
                      enemy_hp: int) -> int:
    for dmg in damage_seq:
        if _roll_hit_py(comp, enemy_shield):
            enemy_hp -= dmg
            if enemy_hp <= 0:
                return enemy_hp
    return enemy_hp

def _battle_once_py(spec_a, spec_b) -> bool:
    a_hp = spec_a["hull"]
    b_hp = spec_b["hull"]
    a_first = spec_a["initiative"] > spec_b["initiative"]

    # missile phase
    if a_first:
        b_hp = _fire_sequence_py(spec_a["missiles"], spec_a["computer"],
                                 spec_b["shield"], b_hp)
        if b_hp <= 0:
            return True
        a_hp = _fire_sequence_py(spec_b["missiles"], spec_b["computer"],
                                 spec_a["shield"], a_hp)
        if a_hp <= 0:
            return False
    else:
        a_hp = _fire_sequence_py(spec_b["missiles"], spec_b["computer"],
                                 spec_a["shield"], a_hp)
        if a_hp <= 0:
            return False
        b_hp = _fire_sequence_py(spec_a["missiles"], spec_a["computer"],
                                 spec_b["shield"], b_hp)
        if b_hp <= 0:
            return True

    # cannon phase
    while True:
        if a_first:
            b_hp = _fire_sequence_py(spec_a["cannons"], spec_a["computer"],
                                     spec_b["shield"], b_hp)
            if b_hp <= 0:
                return True
            a_hp = _fire_sequence_py(spec_b["cannons"], spec_b["computer"],
                                     spec_a["shield"], a_hp)
            if a_hp <= 0:
                return False
        else:
            a_hp = _fire_sequence_py(spec_b["cannons"], spec_b["computer"],
                                     spec_a["shield"], a_hp)
            if a_hp <= 0:
                return False
            b_hp = _fire_sequence_py(spec_a["cannons"], spec_a["computer"],
                                     spec_b["shield"], b_hp)
            if b_hp <= 0:
                return True

# ──────────────────────────────────────────────────────────────────────────
# Public function
# ──────────────────────────────────────────────────────────────────────────

def win_probability(spec_a: dict,
                    spec_b: dict,
                    simulations: int = 100_000,
                    seed: int | None = None) -> float:
    """
    Monte-Carlo estimate of P(ship-A wins).  Draws are impossible.

    Parameters
    ----------
    spec_a, spec_b : mapping
        Same schema as documented at top of file.
    simulations    : int
        Number of independent battles.
    seed           : int or None
        RNG seed for reproducibility.

    Returns
    -------
    float
    """
    if seed is not None:
        random.seed(seed)
        if NUMBA_AVAILABLE:
            np.random.seed(seed)

    # Fast path with Numba
    if NUMBA_AVAILABLE:
        # Convert variable-length lists to 1-D numpy int32 arrays
        a_miss = np.array(spec_a["missiles"], dtype=np.int32)
        a_can  = np.array(spec_a["cannons"],   dtype=np.int32)
        b_miss = np.array(spec_b["missiles"], dtype=np.int32)
        b_can  = np.array(spec_b["cannons"],   dtype=np.int32)

        wins = _batch_simulate(simulations,
                               spec_a["initiative"], spec_a["hull"],
                               spec_a["computer"],   spec_a["shield"],
                               a_miss, a_can,
                               spec_b["initiative"], spec_b["hull"],
                               spec_b["computer"],   spec_b["shield"],
                               b_miss, b_can)
        return wins / simulations

    # Pure-Python fallback
    wins = 0
    for _ in range(simulations):
        if _battle_once_py(spec_a, spec_b):
            wins += 1
    return wins / simulations


# ──────────────────────────────────────────────────────────────────────────
# CLI / quick benchmark
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SHIP_A = dict(initiative=6, hull=3, computer=2, shield=1,
                  missiles=[2, 1], cannons=[2, 1])
    SHIP_B = dict(initiative=4, hull=3, computer=1, shield=1,
                  missiles=[2, 1, 1], cannons=[1, 1])

    import time, sys
    N = 1_000_000
    t0 = time.perf_counter()
    p = win_probability(SHIP_A, SHIP_B, simulations=N, seed=42)
    dt = time.perf_counter() - t0
    mode = "Numba" if NUMBA_AVAILABLE else "Pure-Python"
    print(f"{mode}:  P(A wins) ≈ {p:.4%}   ({N:_} sims in {dt:.2f}s)")
    if not NUMBA_AVAILABLE:
        print("\nTip:  pip / conda install numba  →  >20× speed-up on Apple silicon.")
