"""
Game environments: 4 canonical 2x2 repeated games.
Each game defines a payoff matrix for (self_action, opp_action) -> (self_payoff, opp_payoff).
Actions are "C" (cooperate) or "D" (defect).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PayoffMatrix:
    """Payoff matrix for a 2x2 game. Entries: (self_payoff, opp_payoff)."""
    cc: tuple[float, float]  # both cooperate
    cd: tuple[float, float]  # self cooperates, opp defects
    dc: tuple[float, float]  # self defects, opp cooperates
    dd: tuple[float, float]  # both defect

    def get_payoffs(self, self_action: str, opp_action: str) -> tuple[float, float]:
        key = self_action.upper() + opp_action.upper()
        return {"CC": self.cc, "CD": self.cd, "DC": self.dc, "DD": self.dd}[key]

    @property
    def min_payoff(self) -> float:
        return min(p[0] for p in [self.cc, self.cd, self.dc, self.dd])

    @property
    def max_payoff(self) -> float:
        return max(p[0] for p in [self.cc, self.cd, self.dc, self.dd])

    def format_matrix(self) -> str:
        lines = [
            f"  C,C -> {self.cc[0]},{self.cc[1]}  |  C,D -> {self.cd[0]},{self.cd[1]}",
            f"  D,C -> {self.dc[0]},{self.dc[1]}  |  D,D -> {self.dd[0]},{self.dd[1]}",
        ]
        return "\n".join(lines)

    def format_matrix_verbose(self) -> str:
        lines = [
            f"  You=C, Opponent=C  ->  you earn {self.cc[0]}, opponent earns {self.cc[1]}",
            f"  You=C, Opponent=D  ->  you earn {self.cd[0]}, opponent earns {self.cd[1]}",
            f"  You=D, Opponent=C  ->  you earn {self.dc[0]}, opponent earns {self.dc[1]}",
            f"  You=D, Opponent=D  ->  you earn {self.dd[0]}, opponent earns {self.dd[1]}",
        ]
        return "\n".join(lines)


# ── Game definitions ──

PRISONERS_DILEMMA = PayoffMatrix(
    cc=(3.0, 3.0),
    cd=(0.0, 5.0),
    dc=(5.0, 0.0),
    dd=(1.0, 1.0),
)

STAG_HUNT = PayoffMatrix(
    cc=(4.0, 4.0),
    cd=(0.0, 3.0),
    dc=(3.0, 0.0),
    dd=(2.0, 2.0),
)

CHICKEN = PayoffMatrix(
    cc=(3.0, 3.0),
    cd=(1.0, 5.0),
    dc=(5.0, 1.0),
    dd=(0.0, 0.0),
)

BATTLE_OF_SEXES = PayoffMatrix(
    cc=(3.0, 2.0),
    cd=(0.0, 0.0),
    dc=(0.0, 0.0),
    dd=(2.0, 3.0),
)

GAME_REGISTRY: dict[str, PayoffMatrix] = {
    "prisoners_dilemma": PRISONERS_DILEMMA,
    "stag_hunt": STAG_HUNT,
    "chicken": CHICKEN,
    "battle_of_sexes": BATTLE_OF_SEXES,
}

GAME_DESCRIPTIONS: dict[str, str] = {
    "prisoners_dilemma": "Two players simultaneously choose to Cooperate (C) or Defect (D). Mutual cooperation yields a good outcome for both, but each is tempted to defect for a higher individual payoff.",
    "stag_hunt": "Two players choose to Cooperate (C) for a high mutual payoff (hunting stag) or Defect (D) for a safe but lower payoff (hunting hare). Cooperation requires trust.",
    "chicken": "Two players choose to Cooperate (C, swerve) or Defect (D, go straight). Mutual defection is the worst outcome. Each player prefers the other to swerve.",
    "battle_of_sexes": "Two players prefer to coordinate but disagree on which outcome is better. C,C favors player 1; D,D favors player 2. Miscoordination yields zero.",
}


def get_game(name: str) -> PayoffMatrix:
    if name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game: {name}. Choose from: {list(GAME_REGISTRY.keys())}")
    return GAME_REGISTRY[name]
