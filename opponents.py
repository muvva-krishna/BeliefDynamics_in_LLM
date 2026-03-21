"""
Nine opponent-policy types for the repeated 2x2 game experiments.
Each opponent is a function: (history, game_name) -> action ("C" or "D").
History is a list of (self_action, opp_action) tuples from the opponent's perspective
(i.e., the opponent is "self" in its own history).
"""
import random
from typing import Callable


def always_cooperate(history: list[tuple[str, str]], **kw) -> str:
    return "C"


def always_defect(history: list[tuple[str, str]], **kw) -> str:
    return "D"


def tit_for_tat(history: list[tuple[str, str]], **kw) -> str:
    """Cooperate first, then copy opponent's last action."""
    if not history:
        return "C"
    return history[-1][1]  # opponent's last action (from opp's perspective, [1] is the LLM's action)


def grim_trigger(history: list[tuple[str, str]], **kw) -> str:
    """Cooperate until opponent defects, then defect forever."""
    if not history:
        return "C"
    if any(opp_act == "D" for _, opp_act in history):
        return "D"
    return "C"


def win_stay_lose_shift(history: list[tuple[str, str]], **kw) -> str:
    """
    Start with C. If last round was a 'win' (payoff >= 3), repeat action.
    Otherwise switch. Uses a simple threshold.
    """
    if not history:
        return "C"
    from games import get_game
    game_name = kw.get("game_name", "prisoners_dilemma")
    game = get_game(game_name)
    last_self, last_opp = history[-1]
    payoff, _ = game.get_payoffs(last_self, last_opp)
    threshold = (game.max_payoff + game.min_payoff) / 2.0
    if payoff >= threshold:
        return last_self  # win → stay
    return "D" if last_self == "C" else "C"  # lose → shift


def noisy_tit_for_tat(history: list[tuple[str, str]], **kw) -> str:
    """Tit-for-tat with 10% noise (random flip)."""
    base = tit_for_tat(history, **kw)
    seed = kw.get("rng", random)
    if seed.random() < 0.1:
        return "D" if base == "C" else "C"
    return base


def deceptive_opportunist(history: list[tuple[str, str]], **kw) -> str:
    """
    Cooperate for the first 3 rounds to build trust, then mostly defect.
    Occasionally cooperates (20%) to prevent pure-defect detection.
    """
    seed = kw.get("rng", random)
    if len(history) < 3:
        return "C"
    if seed.random() < 0.2:
        return "C"
    return "D"


def gradual_defector(history: list[tuple[str, str]], **kw) -> str:
    """
    Cooperate initially. After each opponent defection, retaliate with
    an increasing number of defections, then return to cooperation.
    """
    if not history:
        return "C"

    # Count opponent defections so far
    opp_defections = sum(1 for _, opp_act in history if opp_act == "D")

    # Track retaliation state via history length pattern
    # Simple implementation: defect for `opp_defections` rounds after each new defection
    if opp_defections == 0:
        return "C"

    # Count our own recent consecutive defections
    consecutive_d = 0
    for self_act, _ in reversed(history):
        if self_act == "D":
            consecutive_d += 1
        else:
            break

    # If the last opponent action was D and we haven't retaliated enough
    last_opp = history[-1][1]
    if last_opp == "D" and consecutive_d < opp_defections:
        return "D"
    if consecutive_d > 0 and consecutive_d < opp_defections:
        return "D"

    return "C"


def adaptive_mirror(history: list[tuple[str, str]], **kw) -> str:
    """
    Tracks opponent's cooperation rate over a sliding window (last 5 rounds).
    If opponent cooperates > 60%, cooperate. Otherwise defect.
    """
    if not history:
        return "C"
    window = history[-5:]
    coop_rate = sum(1 for _, opp_act in window if opp_act == "C") / len(window)
    return "C" if coop_rate > 0.6 else "D"


# ── Registry ──
OPPONENT_REGISTRY: dict[str, Callable] = {
    "always_cooperate": always_cooperate,
    "always_defect": always_defect,
    "tit_for_tat": tit_for_tat,
    "grim_trigger": grim_trigger,
    "win_stay_lose_shift": win_stay_lose_shift,
    "noisy_tit_for_tat": noisy_tit_for_tat,
    "deceptive_opportunist": deceptive_opportunist,
    "gradual_defector": gradual_defector,
    "adaptive_mirror": adaptive_mirror,
}


def get_opponent_action(
    opponent_type: str,
    history: list[tuple[str, str]],
    game_name: str = "prisoners_dilemma",
    rng: random.Random | None = None,
) -> str:
    """
    Get the opponent's action given history.
    History is from the OPPONENT's perspective: each entry is (opp_action, llm_action).
    """
    if opponent_type not in OPPONENT_REGISTRY:
        raise ValueError(f"Unknown opponent type: {opponent_type}. Choose from: {list(OPPONENT_REGISTRY.keys())}")
    fn = OPPONENT_REGISTRY[opponent_type]
    return fn(history, game_name=game_name, rng=rng or random)
