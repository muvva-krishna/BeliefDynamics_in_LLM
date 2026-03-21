"""
All preregistered metrics for the 5 experiments.
Includes: normalized payoff, cooperation rate, KL divergence, Brier score,
ECE, conditional agreement, EU-consistent coherence, directional accuracy,
sensitivity, JS divergence, rigidity index, and more.
"""
import math
import numpy as np
from typing import Optional

import config
from games import get_game
from oracle import compute_oracle_posterior


# ═══════════════════════════════════════════════════════
# Exp 1: Baseline Strategic Play
# ═══════════════════════════════════════════════════════

def normalized_episode_payoff(payoffs: list[float], game_name: str) -> float:
    """R̄^norm = (1/T) Σ (r_t - r_min) / (r_max - r_min)"""
    game = get_game(game_name)
    r_min = game.min_payoff
    r_max = game.max_payoff
    if r_max == r_min:
        return 0.0
    return sum((p - r_min) / (r_max - r_min) for p in payoffs) / len(payoffs)


def cooperation_rate(actions: list[str]) -> float:
    """CR = (1/T) Σ 𝕀[a_t = C]"""
    if not actions:
        return 0.0
    return sum(1 for a in actions if a == "C") / len(actions)


# ═══════════════════════════════════════════════════════
# Exp 2: Belief Elicitation
# ═══════════════════════════════════════════════════════

def kl_divergence(p_star: dict[str, float], p_hat: dict[str, float],
                  epsilon: float = 1e-6) -> float:
    """
    D_KL(P* || P̂) with ε-smoothing.
    P̂_ε = (1-ε)P̂ + ε/|Θ|
    """
    types = config.OPPONENT_TYPES
    n = len(types)
    total = 0.0
    for t in types:
        ps = p_star.get(t, 0.0)
        ph = p_hat.get(t, 0.0)
        ph_smooth = (1 - epsilon) * ph + epsilon / n
        if ps > 0:
            total += ps * math.log(ps / ph_smooth)
    return total


def brier_score(p_hat: dict[str, float], true_type: str) -> float:
    """
    Multiclass Brier score: BS = Σ (P̂(θ) - 𝕀[θ = θ_true])²
    Range [0, 2].
    """
    total = 0.0
    for t in config.OPPONENT_TYPES:
        ph = p_hat.get(t, 0.0)
        indicator = 1.0 if t == true_type else 0.0
        total += (ph - indicator) ** 2
    return total


def expected_calibration_error(
    predictions: list[tuple[float, bool]],
    n_bins: int = 10,
) -> float:
    """
    ECE with fixed equal-width bins.
    predictions: list of (confidence, was_correct) tuples.
    """
    if not predictions:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    ece = 0.0
    n_total = len(predictions)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, correct in b if correct) / len(b)
        ece += len(b) / n_total * abs(avg_acc - avg_conf)
    return ece


def maximum_calibration_error(
    predictions: list[tuple[float, bool]],
    n_bins: int = 10,
) -> float:
    """MCE: max bin gap."""
    if not predictions:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    mce = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, correct in b if correct) / len(b)
        mce = max(mce, abs(avg_acc - avg_conf))
    return mce


# ═══════════════════════════════════════════════════════
# Exp 3: Belief-Action Coupling
# ═══════════════════════════════════════════════════════

def eu_optimal_action(posterior: dict[str, float], game_name: str,
                      history: list[dict], seed: int) -> str:
    """
    Compute the EU-optimal action under a given posterior.
    a^EU = argmax_a E_θ~P̂[Q(a, θ)]
    """
    from opponents import get_opponent_action
    import random as _random

    game = get_game(game_name)
    rng = _random.Random(seed)
    opp_history = [(r["opp_action"], r["self_action"]) for r in history]

    eu_c = 0.0
    eu_d = 0.0
    for opp_type, prob in posterior.items():
        # What would this opponent type do next?
        opp_act = get_opponent_action(opp_type, opp_history, game_name, rng=_random.Random(seed))
        payoff_c, _ = game.get_payoffs("C", opp_act)
        payoff_d, _ = game.get_payoffs("D", opp_act)
        eu_c += prob * payoff_c
        eu_d += prob * payoff_d

    return "C" if eu_c >= eu_d else "D"


def belief_use_capability(norm_payoff_c: float, norm_payoff_a: float) -> float:
    """Δ payoff (C - A): oracle-injected vs baseline."""
    return norm_payoff_c - norm_payoff_a


def conditional_action_agreement(
    actions_a: list[str],
    actions_b: list[str],
    informative: list[bool],
) -> float:
    """
    Agreement rate conditioned on informative rounds
    (where EU-optimal action under reference posterior differs from baseline's implied action).
    """
    if not any(informative):
        return float('nan')
    matches = sum(
        1 for a, b, inf in zip(actions_a, actions_b, informative)
        if inf and a == b
    )
    total = sum(1 for inf in informative if inf)
    return matches / total if total > 0 else float('nan')


def eu_consistent_coherence(
    actions: list[str],
    posteriors: list[dict[str, float]],
    game_name: str,
    history_list: list[list[dict]],
    seed: int,
) -> float:
    """
    Coherence = (1/N) Σ 𝕀[a_self = argmax E_θ~P̂[Q(a,θ)]]
    """
    if not actions:
        return 0.0
    matches = 0
    for action, post, hist in zip(actions, posteriors, history_list):
        optimal = eu_optimal_action(post, game_name, hist, seed)
        if action == optimal:
            matches += 1
    return matches / len(actions)


# ═══════════════════════════════════════════════════════
# Exp 4: Posterior Intervention
# ═══════════════════════════════════════════════════════

def directional_accuracy(
    actions_cf: list[str],
    eu_actions_cf: list[str],
) -> float:
    """DirAcc = (1/N) Σ 𝕀[a_cf = a^EU(P_cf)]"""
    if not actions_cf:
        return 0.0
    return sum(1 for a, e in zip(actions_cf, eu_actions_cf) if a == e) / len(actions_cf)


def sensitivity_to_posterior(
    actions_cf: list[str],
    actions_ctl: list[str],
) -> float:
    """Sens = (1/N) Σ 𝕀[a_cf ≠ a_ctl]"""
    if not actions_cf:
        return 0.0
    return sum(1 for a, b in zip(actions_cf, actions_ctl) if a != b) / len(actions_cf)


def adversarial_flip_posterior(
    oracle_posterior: dict[str, float],
    eta: float = 0.01,
) -> dict[str, float]:
    """
    Adversarial flip: redistribute mass to the least likely type.
    P^cf(θ_min) = 1 - η, P^cf(θ ≠ θ_min) = η/(K-1)
    This creates the maximal posterior change — strongest possible intervention.
    """
    types = config.OPPONENT_TYPES
    min_type = min(oracle_posterior, key=oracle_posterior.get)
    k = len(types)
    cf = {}
    for t in types:
        if t == min_type:
            cf[t] = 1.0 - eta
        else:
            cf[t] = eta / (k - 1)
    return cf


def plausible_perturbation_posterior(
    oracle_posterior: dict[str, float],
    lam: float = 0.80,
) -> dict[str, float]:
    """
    Plausible perturbation: P^cf = (1-λ)P* + λU
    λ=0.80: strong enough to shift EU margins in non-dominant games.
    Note: for near-uniform oracles in dominant-strategy games (PD),
    mixing toward uniform barely changes EU margins — adversarial_flip
    compensates for those cases.
    """
    types = config.OPPONENT_TYPES
    k = len(types)
    cf = {}
    for t in types:
        cf[t] = (1 - lam) * oracle_posterior.get(t, 0.0) + lam / k
    # Normalize
    total = sum(cf.values())
    return {t: v / total for t, v in cf.items()}


def eu_margin(posterior: dict[str, float], game_name: str,
              history: list[dict], seed: int) -> float:
    """
    EU(D) - EU(C) under a given posterior.
    Positive = D is better, Negative = C is better.
    Used as a continuous validity signal for Exp4 (complements binary eu_optimal_action).
    """
    from opponents import get_opponent_action
    import random as _random

    game = get_game(game_name)
    opp_history = [(r["opp_action"], r["self_action"]) for r in history]

    eu_c = 0.0
    eu_d = 0.0
    for opp_type, prob in posterior.items():
        opp_act = get_opponent_action(opp_type, opp_history, game_name,
                                     rng=_random.Random(seed))
        payoff_c, _ = game.get_payoffs("C", opp_act)
        payoff_d, _ = game.get_payoffs("D", opp_act)
        eu_c += prob * payoff_c
        eu_d += prob * payoff_d

    return eu_d - eu_c  # positive → D is EU-optimal


def margin_threshold(game_name: str, alpha: float = 0.10) -> float:
    """
    Principled validity threshold for EU-margin shift.
    threshold = alpha * (max_payoff - min_payoff)
    Scales with each game's payoff range so the threshold is never arbitrary.
    Default alpha=0.10 means a 10% shift in the game's payoff range counts as valid.
    """
    game = get_game(game_name)
    return alpha * (game.max_payoff - game.min_payoff)


# ═══════════════════════════════════════════════════════
# Exp 6: Cognitive Theory Metrics (offline)
# ═══════════════════════════════════════════════════════

def js_divergence(p_star: dict[str, float], p_hat: dict[str, float]) -> float:
    """
    Jensen-Shannon divergence: JS(P*, P̂) = ½ KL(P*||M) + ½ KL(P̂||M)
    where M = ½(P* + P̂). Range [0, ln2].
    """
    types = config.OPPONENT_TYPES
    m = {t: 0.5 * (p_star.get(t, 0.0) + p_hat.get(t, 0.0)) for t in types}

    kl_star_m = 0.0
    kl_hat_m = 0.0
    for t in types:
        ps = p_star.get(t, 0.0)
        ph = p_hat.get(t, 0.0)
        mt = m[t]
        if ps > 0 and mt > 0:
            kl_star_m += ps * math.log(ps / mt)
        if ph > 0 and mt > 0:
            kl_hat_m += ph * math.log(ph / mt)

    return 0.5 * kl_star_m + 0.5 * kl_hat_m


def rigidity_index(cooperation_rates: dict[str, float]) -> float:
    """
    RI = 1 - SD(CR_g) / 0.57735
    Higher RI = more rigid (less cross-game adaptation).
    """
    if len(cooperation_rates) < 2:
        return 0.0
    values = list(cooperation_rates.values())
    sd = float(np.std(values, ddof=0))
    return 1.0 - sd / 0.57735


def surprisal(predicted_prob: float) -> float:
    """δ_t = -log P_model(a^opp_{t+1} | h_t)"""
    if predicted_prob <= 0:
        return float('inf')
    return -math.log(predicted_prob)


def belief_update_magnitude(
    posterior_t: dict[str, float],
    posterior_t_prev: dict[str, float],
) -> float:
    """L1 norm of posterior change between consecutive probe rounds."""
    types = config.OPPONENT_TYPES
    return sum(abs(posterior_t.get(t, 0.0) - posterior_t_prev.get(t, 0.0)) for t in types)


def compute_all_exp1_metrics(episode: dict) -> dict:
    """Compute all Exp1 metrics for a completed episode."""
    rounds = episode["rounds"]
    game_name = episode["game_name"]
    payoffs = [r["self_payoff"] for r in rounds]
    actions = [r["self_action"] for r in rounds]

    return {
        "normalized_payoff": normalized_episode_payoff(payoffs, game_name),
        "cooperation_rate": cooperation_rate(actions),
        "total_payoff": sum(payoffs),
        "mean_payoff": sum(payoffs) / len(payoffs) if payoffs else 0,
    }


def compute_all_exp2_metrics(episode: dict, true_type: str) -> dict:
    """Compute all Exp2 metrics for a completed episode with probe data."""
    probes = [r for r in episode["rounds"] if r.get("is_probe")]
    if not probes:
        return {}

    kl_values = []
    brier_values = []
    calibration_data = []

    for probe in probes:
        oracle_post = probe.get("oracle_posterior", {})
        model_post = probe.get("posterior_over_types", {})
        confidence = probe.get("confidence", 0.0)
        predicted = probe.get("predicted_next_action", "")
        actual_next = probe.get("actual_next_opp_action", "")

        if model_post and oracle_post:
            kl_values.append(kl_divergence(oracle_post, model_post))
            brier_values.append(brier_score(model_post, true_type))

        if predicted and actual_next:
            calibration_data.append((confidence, predicted == actual_next))

    result = {}
    if kl_values:
        result["mean_kl"] = sum(kl_values) / len(kl_values)
        result["kl_values"] = kl_values
    if brier_values:
        result["mean_brier"] = sum(brier_values) / len(brier_values)
    if calibration_data:
        result["ece"] = expected_calibration_error(calibration_data)
        result["mce"] = maximum_calibration_error(calibration_data)

    return result
