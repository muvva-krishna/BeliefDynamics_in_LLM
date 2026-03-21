"""
Oracle Bayesian posterior computation.
Computes P(θ | h_t) ∝ P(θ) * ∏ P_θ(a_opp_τ | h_τ) for each opponent type θ.
Uses log-space for numerical stability.
"""
import math
import random as _random
from typing import Optional

import config
from games import get_game
from opponents import OPPONENT_REGISTRY, get_opponent_action


def compute_opponent_likelihood(
    opponent_type: str,
    opp_action: str,
    history_before: list[tuple[str, str]],
    game_name: str,
    seed: int,
    noise_epsilon: float = 0.01,
) -> float:
    """
    Compute P_θ(a_opp | h_t) — the probability that opponent type θ
    would play `opp_action` given history_before.

    For deterministic policies, returns (1 - ε) if action matches, ε otherwise.
    For stochastic policies (noisy_tit_for_tat, deceptive_opportunist),
    returns the actual probability.
    """
    rng = _random.Random(seed)

    # Get what this opponent type would deterministically do
    predicted = get_opponent_action(opponent_type, history_before, game_name, rng=rng)

    # Stochastic opponents: compute actual probability
    if opponent_type == "noisy_tit_for_tat":
        base = "C" if not history_before else history_before[-1][1]
        if opp_action == base:
            return 0.9
        else:
            return 0.1

    if opponent_type == "deceptive_opportunist":
        if len(history_before) < 3:
            return 1.0 - noise_epsilon if opp_action == "C" else noise_epsilon
        if opp_action == "D":
            return 0.8
        else:
            return 0.2

    # Deterministic opponents: near-1 if match, near-0 if not
    if opp_action == predicted:
        return 1.0 - noise_epsilon
    else:
        return noise_epsilon


def compute_oracle_posterior(
    history: list[dict],
    game_name: str,
    seed: int,
    prior: Optional[dict[str, float]] = None,
    epsilon_smooth: float = 1e-6,
) -> dict[str, float]:
    """
    Compute the oracle Bayesian posterior P*(θ | h_t) over all 9 opponent types.

    Args:
        history: List of round dicts with keys: self_action, opp_action
        game_name: Name of the game
        seed: Random seed for deterministic opponent policies
        prior: Prior probabilities. Defaults to uniform 1/9.
        epsilon_smooth: Smoothing parameter to avoid zero probabilities.

    Returns:
        Dict mapping opponent_type -> posterior probability (sums to 1.0).
    """
    types = config.OPPONENT_TYPES

    if prior is None:
        prior = {t: 1.0 / len(types) for t in types}

    # Compute in log-space: log P(θ|h) = log P(θ) + Σ log P_θ(a_opp_τ | h_τ)
    log_posterior = {}
    for opp_type in types:
        log_p = math.log(prior.get(opp_type, 1.0 / len(types)))

        # Build opponent's history incrementally
        opp_history: list[tuple[str, str]] = []
        for rnd in history:
            self_act = rnd["self_action"]
            opp_act = rnd["opp_action"]

            # Likelihood of this opponent playing opp_act given history so far
            lik = compute_opponent_likelihood(
                opp_type, opp_act, opp_history, game_name, seed
            )
            log_p += math.log(max(lik, 1e-300))

            # Update opponent's history (from opponent's perspective: opp is "self")
            opp_history.append((opp_act, self_act))

        log_posterior[opp_type] = log_p

    # Normalize via log-sum-exp
    max_log = max(log_posterior.values())
    exp_sum = sum(math.exp(lp - max_log) for lp in log_posterior.values())
    log_norm = max_log + math.log(exp_sum)

    posterior = {}
    for opp_type in types:
        p = math.exp(log_posterior[opp_type] - log_norm)
        # Apply ε-smoothing
        p = (1 - epsilon_smooth) * p + epsilon_smooth / len(types)
        posterior[opp_type] = p

    # Renormalize after smoothing
    total = sum(posterior.values())
    posterior = {k: v / total for k, v in posterior.items()}

    return posterior


def is_identifiable(posterior: dict[str, float], tau: float = 0.60) -> bool:
    """Check if max posterior probability exceeds threshold τ."""
    return max(posterior.values()) >= tau


def get_oracle_prediction(
    posterior: dict[str, float],
    history: list[dict],
    game_name: str,
    seed: int,
) -> str:
    """
    Get the oracle's predicted next opponent action (most likely under posterior).
    Weighted vote across opponent types.
    """
    rng = _random.Random(seed)
    # Build opponent history
    opp_history = [(r["opp_action"], r["self_action"]) for r in history]

    c_prob = 0.0
    for opp_type, prob in posterior.items():
        predicted = get_opponent_action(opp_type, opp_history, game_name, rng=rng)
        if predicted == "C":
            c_prob += prob
        # else d_prob += prob

    return "C" if c_prob >= 0.5 else "D"
