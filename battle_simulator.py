import random
from typing import List, Dict, Optional, Union, Tuple

# ────────────────────  core model  ────────────────────
class Ship:
    def __init__(
        self,
        initiative: int,
        hull: int,
        computer: int,
        shield: int,
        missiles: List[int],
        cannons: List[int],
    ) -> None:
        self.initiative = initiative
        self.hull = hull
        self.computer = computer
        self.shield = shield
        self.missiles = missiles       # each entry = fixed damage of one shot
        self.cannons = cannons         # idem

    # ── helpers ────────────────────────────────────────
    def _destroyed(self) -> bool:
        return self.hull <= 0

    def _take(self, dmg: int) -> None:
        self.hull -= dmg

    def _roll(self) -> Union[str, int]:
        """Uniform pick from six outcomes."""
        return random.choice(["No damage", 2, 3, 4, 5, "Full hit"])

    def _hit(self, roll: Union[str, int], target_shield: int) -> bool:
        if roll == "No damage":
            return False
        if roll == "Full hit":
            return True
        return roll + self.computer - target_shield >= 6

    def _fire(self, target: "Ship", weapon: List[int]) -> None:
        for dmg in weapon:
            if self._hit(self._roll(), target.shield):
                target._take(dmg)
                if target._destroyed():
                    break


def _simulate_once(a: Ship, b: Ship) -> int:
    """Run one battle; return 1 if A wins, -1 if B wins, 0 for draw."""
    first, second = (a, b) if a.initiative > b.initiative else (b, a)

    # missile phase
    if first.missiles:
        first._fire(second, first.missiles)
        if second._destroyed():
            return 1 if first is a else -1
    if second.missiles:
        second._fire(first, second.missiles)
        if first._destroyed():
            return 1 if second is a else -1

    # cannon phase
    while not a._destroyed() and not b._destroyed():
        first._fire(second, first.cannons)
        if second._destroyed():
            break
        second._fire(first, second.cannons)

    if a._destroyed() and b._destroyed():
        return 0
    return 1 if not a._destroyed() else -1


# ────────────────────  public API  ─────────────────────
def win_probability(
    spec_a: Dict[str, Union[int, List[int]]],
    spec_b: Dict[str, Union[int, List[int]]],
    simulations: int = 100_000,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate P(Ship A wins) by running `simulations` independent battles.

    Parameters
    ----------
    spec_a, spec_b : mapping with keys
        initiative, hull, computer, shield, missiles, cannons
    simulations    : how many Monte-Carlo trials (default 100 000)
    seed           : optional RNG seed for reproducibility

    Returns
    -------
    float   probability that Ship A wins
    """
    if seed is not None:
        random.seed(seed)

    wins = 0
    for _ in range(simulations):
        ship_a = Ship(**spec_a)
        ship_b = Ship(**spec_b)
        outcome = _simulate_once(ship_a, ship_b)
        wins += 1 if outcome == 1 else 0
    return wins / simulations


# ────────────────────  example usage  ──────────────────
if __name__ == "__main__":
    a_spec = dict(
        initiative=5,
        hull=2,
        computer=2,
        shield=1,
        missiles=[1, 1],   # two 1-dmg missiles
        cannons=[1],       # one 1-dmg cannon
    )
    b_spec = dict(
        initiative=4,
        hull=2,
        computer=1,
        shield=1,
        missiles=[1, 1, 1],  # three missiles
        cannons=[1],
    )

    p = win_probability(a_spec, b_spec, simulations=200_000, seed=42)
    print(f"Estimated P(A wins) ≈ {p:.3%}")
