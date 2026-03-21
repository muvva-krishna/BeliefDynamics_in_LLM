"""
Unified Pipeline: runs gameplay ONCE per condition, derives all experiment data.

Stage 1: Base Gameplay + Probes  (13 calls/episode) → Exp1, Exp2, Exp3-A, Exp3-B
Stage 2: B2 Two-Stage Probes    (6 calls/episode)  → Exp3-B2
Stage 3: Oracle-Injected Game   (10 calls/episode)  → Exp3-C
Stage 4: Posterior Intervention  (0-12 calls/episode)→ Exp4
Stage 5: Cognitive Metrics       (0 calls)           → Exp6
"""
import random
import logging
import math
from typing import Any

import config
from games import get_game
from opponents import get_opponent_action
from oracle import compute_oracle_posterior, is_identifiable
from api_clients import call_llm, parse_json_response
from prompts import (
    build_system_prompt, build_oracle_system_prompt, build_intervention_system_prompt,
    ACTION_SCHEMA, PROBE_SCHEMA, BELIEF_ONLY_SCHEMA,
    action_user_prompt, probe_user_prompt,
    b2_belief_prompt, b2_action_prompt,
    oracle_user_prompt, intervention_user_prompt,
)
from metrics import (
    normalized_episode_payoff, cooperation_rate,
    eu_optimal_action, eu_margin, margin_threshold,
    adversarial_flip_posterior, plausible_perturbation_posterior,
    js_divergence, kl_divergence, expected_calibration_error, maximum_calibration_error,
    rigidity_index, surprisal, belief_update_magnitude,
)
from checkpoint import save_pipeline_checkpoint, load_pipeline_checkpoint, clear_pipeline_checkpoint

logger = logging.getLogger(__name__)


def normalize_posterior(raw: dict) -> dict[str, float]:
    """Normalize a raw posterior dict to sum to 1.0, fill missing types with 0."""
    result = {}
    for t in config.OPPONENT_TYPES:
        try:
            result[t] = max(float(raw.get(t, 0.0)), 0.0)
        except (ValueError, TypeError):
            result[t] = 0.0
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    else:
        n = len(config.OPPONENT_TYPES)
        result = {k: 1.0 / n for k in config.OPPONENT_TYPES}
    return result


def run_game_round(game_name: str, opponent_type: str,
                   opp_history: list[tuple[str, str]],
                   llm_action: str, game_name_kw: str = "",
                   rng: random.Random | None = None) -> tuple[str, float, float]:
    """Execute one round: get opponent action and compute payoffs."""
    game = get_game(game_name)
    opp_action = get_opponent_action(opponent_type, opp_history, game_name, rng=rng)
    self_payoff, opp_payoff = game.get_payoffs(llm_action, opp_action)
    return opp_action, self_payoff, opp_payoff


class UnifiedPipeline:
    """
    Runs gameplay ONCE per condition, then derives all experiment data.
    A condition is a (game, opponent_type, episode_idx, seed) tuple.
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        games: list[str] | None = None,
        opponent_types: list[str] | None = None,
        episodes_per_condition: int = 10,
        rounds_per_episode: int = 10,
        base_seed: int = 42,
        resume: bool = True,
        stages: list[str] | None = None,
        on_condition_complete=None,
        tag: str = "",
    ):
        self.provider = provider
        self.model = model or config.DEFAULT_MODELS.get(provider, "")
        self.tag = tag
        # Optional callback: called after each condition with (completed_list, provider, model)
        self.on_condition_complete = on_condition_complete
        self.games = games or config.GAME_NAMES
        self.opponent_types = opponent_types or config.OPPONENT_TYPES
        self.episodes_per_condition = episodes_per_condition
        self.rounds_per_episode = rounds_per_episode
        self.base_seed = base_seed
        self.resume = resume
        # Which stages to run: default all
        self.stages = set(stages or ["stage1", "stage2", "stage3", "stage4", "stage5"])

        self.completed: list[dict] = []
        self.pending: list[dict] = []

    def _build_conditions(self) -> list[dict]:
        conditions = []
        for game_name in self.games:
            for opp_type in self.opponent_types:
                for ep_idx in range(self.episodes_per_condition):
                    seed = self.base_seed + hash((game_name, opp_type, ep_idx)) % (2**31)
                    conditions.append({
                        "game_name": game_name,
                        "opponent_type": opp_type,
                        "episode_idx": ep_idx,
                        "seed": seed,
                    })
        return conditions

    def run(self) -> list[dict]:
        """Run the full unified pipeline with checkpointing."""
        # Try resume
        if self.resume:
            ckpt = load_pipeline_checkpoint(self.provider, self.model, self.tag)
            if ckpt:
                self.completed = ckpt.get("completed", [])
                self.pending = ckpt.get("pending", [])
                logger.info(f"Resumed: {len(self.completed)} done, {len(self.pending)} pending")

        if not self.pending and not self.completed:
            self.pending = self._build_conditions()

        total = len(self.completed) + len(self.pending)
        logger.info(f"Pipeline: {total} conditions, {len(self.completed)} done, "
                     f"{len(self.pending)} pending, stages={self.stages}")

        while self.pending:
            cond = self.pending[0]
            n = len(self.completed) + 1
            logger.info(f"[{n}/{total}] {cond['game_name']} vs {cond['opponent_type']} "
                         f"ep={cond['episode_idx']} seed={cond['seed']}")

            try:
                result = self._run_condition(cond)
                self.completed.append(result)
                self.pending.pop(0)
                save_pipeline_checkpoint(
                    self.provider, self.model, self.completed, self.pending, self.tag
                )
                # Live CSV/TXT update after every condition
                if self.on_condition_complete:
                    try:
                        self.on_condition_complete(self.completed, self.provider, self.model)
                    except Exception as cb_err:
                        logger.warning(f"Live analyzer callback failed: {cb_err}")
            except KeyboardInterrupt:
                logger.warning("Interrupted — saving checkpoint")
                save_pipeline_checkpoint(
                    self.provider, self.model, self.completed, self.pending, self.tag
                )
                raise
            except Exception as e:
                logger.error(f"Error in condition {n}: {e}")
                save_pipeline_checkpoint(
                    self.provider, self.model, self.completed, self.pending, self.tag
                )
                raise

        # Save final checkpoint instead of deleting — needed for --merge-tags
        save_pipeline_checkpoint(
            self.provider, self.model, self.completed, [], self.tag
        )
        logger.info(f"Pipeline complete: {len(self.completed)} conditions (checkpoint preserved)")
        return self.completed

    def _run_condition(self, condition: dict) -> dict:
        """Run all enabled stages for a single condition."""
        game_name = condition["game_name"]
        opp_type = condition["opponent_type"]
        seed = condition["seed"]
        ep_id = f"{game_name}_{opp_type}_{condition['episode_idx']}"

        result = {
            "condition": condition,
            "episode_id": ep_id,
            "game_name": game_name,
            "opponent_type": opp_type,
            "seed": seed,
        }

        # ── Stage 1: Base Gameplay + Probes ──
        if "stage1" in self.stages:
            s1 = self._stage1(game_name, opp_type, seed, ep_id)
            result["stage1"] = s1
        else:
            s1 = None

        # ── Stage 2: B2 Two-Stage Probes ──
        if "stage2" in self.stages and s1:
            result["stage2"] = self._stage2(game_name, seed, s1, ep_id)

        # ── Stage 3: Oracle-Injected Game ──
        if "stage3" in self.stages:
            result["stage3"] = self._stage3(game_name, opp_type, seed, ep_id)

        # ── Stage 4: Posterior Intervention ──
        if "stage4" in self.stages and s1:
            result["stage4"] = self._stage4(game_name, opp_type, seed, s1, ep_id)

        # ── Stage 5: Cognitive Metrics (offline) ──
        if "stage5" in self.stages and s1:
            result["stage5"] = self._stage5(game_name, opp_type, seed, s1)

        return result

    # ──────────────────────────────────────────────────
    # Stage 1: Base Gameplay + Probes (13 calls)
    # ──────────────────────────────────────────────────
    def _stage1(self, game_name: str, opp_type: str, seed: int, ep_id: str) -> dict:
        rng = random.Random(seed)
        sys_prompt = build_system_prompt(game_name)
        rounds: list[dict] = []
        opp_history: list[tuple[str, str]] = []
        probe_data: list[dict] = []
        api_log: list[dict] = []

        for t in range(1, self.rounds_per_episode + 1):
            # Gameplay call: action only
            user_msg = action_user_prompt(t, self.rounds_per_episode, rounds)
            resp = call_llm(
                self.provider, sys_prompt, user_msg, self.model,
                json_schema=ACTION_SCHEMA,
                metadata={"stage": "s1_action", "ep": ep_id, "t": t},
            )
            parsed = parse_json_response(resp)
            action = parsed.get("action", "C").upper()
            if action not in ("C", "D"):
                action = "C"

            # Execute round
            opp_action, self_payoff, opp_payoff = run_game_round(
                game_name, opp_type, opp_history, action, rng=rng
            )

            round_rec = {
                "t": t,
                "self_action": action,
                "opp_action": opp_action,
                "self_payoff": self_payoff,
                "opp_payoff": opp_payoff,
            }
            rounds.append(round_rec)
            opp_history.append((opp_action, action))
            _pf = (parsed == {})
            api_log.append({
                "stage": "s1_action", "t": t,
                "usage": resp.get("usage", {}),
                "latency": resp.get("latency", 0),
                "parse_failure": _pf,
                "schema_failure": not _pf and not bool(parsed.get("action")),
            })

            # Probe call at probe rounds (ADDITIONAL call, does not affect gameplay)
            if t in config.PROBE_ROUNDS:
                probe_msg = probe_user_prompt(t, self.rounds_per_episode, rounds)
                probe_resp = call_llm(
                    self.provider, sys_prompt, probe_msg, self.model,
                    json_schema=PROBE_SCHEMA,
                    max_tokens=config.MAX_TOKENS_PROBE,
                    metadata={"stage": "s1_probe", "ep": ep_id, "t": t},
                )
                probe_parsed = parse_json_response(probe_resp)

                # Oracle posterior at this point
                history_simple = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                                  for r in rounds]
                oracle_post = compute_oracle_posterior(history_simple, game_name, seed)
                model_post = normalize_posterior(probe_parsed.get("posterior_over_types", {}))

                probe_rec = {
                    "t": t,
                    "oracle_posterior": oracle_post,
                    "posterior_over_types": model_post,
                    "confidence": float(probe_parsed.get("confidence", 0.0)),
                    "predicted_next_action": probe_parsed.get("predicted_next_action", ""),
                    "probe_action": probe_parsed.get("action", ""),
                    "identifiable": is_identifiable(oracle_post),
                }
                probe_data.append(probe_rec)
                _probe_pf = (probe_parsed == {})
                api_log.append({
                    "stage": "s1_probe", "t": t,
                    "usage": probe_resp.get("usage", {}),
                    "latency": probe_resp.get("latency", 0),
                    "parse_failure": _probe_pf,
                    "schema_failure": not _probe_pf and not bool(probe_parsed.get("posterior_over_types")),
                })

        # Back-fill actual_next_opp_action for probes
        for pd in probe_data:
            pt = pd["t"]
            if pt < len(rounds):
                pd["actual_next_opp_action"] = rounds[pt]["opp_action"]  # rounds is 0-indexed, t is 1-indexed

        return {
            "rounds": rounds,
            "probe_data": probe_data,
            "api_log": api_log,
        }

    # ──────────────────────────────────────────────────
    # Stage 2: B2 Two-Stage Probes (6 calls)
    # ──────────────────────────────────────────────────
    def _stage2(self, game_name: str, seed: int, s1: dict, ep_id: str) -> dict:
        sys_prompt = build_system_prompt(game_name)
        rounds = s1["rounds"]
        b2_data: list[dict] = []
        api_log: list[dict] = []

        for t in config.PROBE_ROUNDS:
            if t > len(rounds):
                continue
            history_so_far = rounds[:t - 1]  # history before this round

            # Call 1: belief only
            belief_msg = b2_belief_prompt(t, self.rounds_per_episode, history_so_far)
            resp1 = call_llm(
                self.provider, sys_prompt, belief_msg, self.model,
                json_schema=BELIEF_ONLY_SCHEMA,
                max_tokens=config.MAX_TOKENS_PROBE,
                metadata={"stage": "s2_belief", "ep": ep_id, "t": t},
            )
            parsed1 = parse_json_response(resp1)
            posterior = normalize_posterior(parsed1.get("posterior_over_types", {}))
            confidence = float(parsed1.get("confidence", 0.0))
            predicted = parsed1.get("predicted_next_action", "")

            _s2_pf = (parsed1 == {})
            api_log.append({"stage": "s2_belief", "t": t,
                            "usage": resp1.get("usage", {}), "latency": resp1.get("latency", 0),
                            "parse_failure": _s2_pf,
                            "schema_failure": not _s2_pf and not bool(parsed1.get("posterior_over_types"))})

            # Call 2: action from posterior
            action_msg = b2_action_prompt(t, self.rounds_per_episode, history_so_far, posterior)
            resp2 = call_llm(
                self.provider, sys_prompt, action_msg, self.model,
                json_schema=ACTION_SCHEMA,
                metadata={"stage": "s2_action", "ep": ep_id, "t": t},
            )
            parsed2 = parse_json_response(resp2)
            b2_action = parsed2.get("action", "C").upper()
            if b2_action not in ("C", "D"):
                b2_action = "C"

            _s2a_pf = (parsed2 == {})
            api_log.append({"stage": "s2_action", "t": t,
                            "usage": resp2.get("usage", {}), "latency": resp2.get("latency", 0),
                            "parse_failure": _s2a_pf,
                            "schema_failure": not _s2a_pf and not bool(parsed2.get("action"))})

            b2_data.append({
                "t": t,
                "posterior_over_types": posterior,
                "confidence": confidence,
                "predicted_next_action": predicted,
                "b2_action": b2_action,
            })

        return {"b2_data": b2_data, "api_log": api_log}

    # ──────────────────────────────────────────────────
    # Stage 3: Oracle-Injected Game (10 calls)
    # ──────────────────────────────────────────────────
    def _stage3(self, game_name: str, opp_type: str, seed: int, ep_id: str) -> dict:
        rng = random.Random(seed)  # same seed = same opponent sequence
        sys_prompt = build_oracle_system_prompt(game_name)
        rounds: list[dict] = []
        opp_history: list[tuple[str, str]] = []
        api_log: list[dict] = []

        for t in range(1, self.rounds_per_episode + 1):
            # Compute oracle posterior from history so far
            history_simple = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                              for r in rounds]
            oracle_post = compute_oracle_posterior(history_simple, game_name, seed)

            user_msg = oracle_user_prompt(t, self.rounds_per_episode, rounds, oracle_post)
            resp = call_llm(
                self.provider, sys_prompt, user_msg, self.model,
                json_schema=ACTION_SCHEMA,
                metadata={"stage": "s3_oracle", "ep": ep_id, "t": t},
            )
            parsed = parse_json_response(resp)
            action = parsed.get("action", "C").upper()
            if action not in ("C", "D"):
                action = "C"

            opp_action, self_payoff, opp_payoff = run_game_round(
                game_name, opp_type, opp_history, action, rng=rng
            )

            rounds.append({
                "t": t,
                "self_action": action,
                "opp_action": opp_action,
                "self_payoff": self_payoff,
                "opp_payoff": opp_payoff,
                "oracle_posterior": oracle_post,
            })
            opp_history.append((opp_action, action))
            _s3_pf = (parsed == {})
            api_log.append({"stage": "s3_oracle", "t": t,
                            "usage": resp.get("usage", {}), "latency": resp.get("latency", 0),
                            "parse_failure": _s3_pf,
                            "schema_failure": not _s3_pf and not bool(parsed.get("action"))})

        return {"rounds": rounds, "api_log": api_log}

    # ──────────────────────────────────────────────────
    # Stage 4: Posterior Intervention (0-12 calls)
    # ──────────────────────────────────────────────────
    def _stage4(self, game_name: str, opp_type: str, seed: int,
                s1: dict, ep_id: str) -> dict:
        sys_prompt = build_intervention_system_prompt(game_name)
        rounds = s1["rounds"]
        probe_results: list[dict] = []
        api_log: list[dict] = []

        # Principled margin threshold: alpha * payoff_range (game-specific).
        # alpha=0.05: 5% of payoff range. Balances sensitivity vs triviality:
        # too high (0.10) → near-uniform oracle conditions (PD cooperative play)
        # always fail plausible_perturbation; too low → trivial.
        m_thresh = margin_threshold(game_name, alpha=0.05)

        for t in config.PROBE_ROUNDS:
            if t > len(rounds):
                continue
            h_before = rounds[:t - 1]
            h_simple = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                        for r in h_before]

            oracle_post = compute_oracle_posterior(h_simple, game_name, seed)

            for transform_name, transform_fn in [
                ("adversarial_flip", adversarial_flip_posterior),
                ("plausible_perturbation", plausible_perturbation_posterior),
            ]:
                cf_post = transform_fn(oracle_post)

                # Sanity: verify the transform actually changed the posterior
                js_ctl_cf = js_divergence(oracle_post, cf_post)
                if js_ctl_cf < 1e-6:
                    logger.warning(f"Stage4 {ep_id} t={t} {transform_name}: "
                                   f"JS(ctl,cf)={js_ctl_cf:.6f} — transform had no effect")

                # EU-optimal actions and continuous margins
                eu_ctl = eu_optimal_action(oracle_post, game_name, h_simple, seed)
                eu_cf = eu_optimal_action(cf_post, game_name, h_simple, seed)
                margin_ctl = eu_margin(oracle_post, game_name, h_simple, seed)
                margin_cf = eu_margin(cf_post, game_name, h_simple, seed)
                margin_delta = abs(margin_cf - margin_ctl)

                # Validity: EU-optimal action flips OR EU-margin shifts by
                # >= alpha * payoff_range.  The margin check catches dominant-strategy
                # games (PD) where the argmax never changes but the decision surface
                # meaningfully shifts.
                eu_action_flip = eu_ctl != eu_cf
                eu_margin_shift = margin_delta >= m_thresh
                is_valid = eu_action_flip or eu_margin_shift

                # 3-way validity classification: flip_only, margin_only, both
                validity_type = "invalid"
                if eu_action_flip and eu_margin_shift:
                    validity_type = "both"
                elif eu_action_flip:
                    validity_type = "flip_only"
                elif eu_margin_shift:
                    validity_type = "margin_only"

                # Sensitivity analysis: validity at alternative alpha thresholds
                # strict=0.10 (old default), default=0.05 (current), lenient=0.03
                valid_strict  = eu_action_flip or margin_delta >= margin_threshold(game_name, 0.10)
                valid_default = eu_action_flip or margin_delta >= margin_threshold(game_name, 0.05)
                valid_lenient = eu_action_flip or margin_delta >= margin_threshold(game_name, 0.03)

                if not is_valid:
                    probe_results.append({
                        "t": t, "transform": transform_name, "is_valid": False,
                        "validity_type": "invalid",
                        "action_ctl": "", "action_cf": "",
                        "eu_ctl": eu_ctl, "eu_cf": eu_cf,
                        "eu_margin_ctl": margin_ctl, "eu_margin_cf": margin_cf,
                        "margin_delta": margin_delta,
                        "eu_action_flip": False,
                        "js_ctl_cf": js_ctl_cf,
                        "valid_strict": valid_strict, "valid_default": valid_default,
                        "valid_lenient": valid_lenient,
                        "dir_acc_cf": 0, "sensitive": 0,
                    })
                    continue

                # Randomize order
                order_rng = random.Random(seed + t + hash(transform_name))
                ctl_first = order_rng.random() > 0.5

                queries = [(oracle_post, "control"), (cf_post, "counterfactual")]
                if not ctl_first:
                    queries = queries[::-1]

                actions = {}
                for posterior, label in queries:
                    msg = intervention_user_prompt(t, self.rounds_per_episode, h_before, posterior)
                    resp = call_llm(
                        self.provider, sys_prompt, msg, self.model,
                        json_schema=ACTION_SCHEMA,
                        metadata={"stage": "s4", "ep": ep_id, "t": t,
                                  "label": label, "transform": transform_name},
                    )
                    parsed = parse_json_response(resp)
                    act = parsed.get("action", "C").upper()
                    if act not in ("C", "D"):
                        act = "C"
                    actions[label] = act
                    _s4_pf = (parsed == {})
                    api_log.append({
                        "stage": "s4", "t": t, "label": label, "transform": transform_name,
                        "usage": resp.get("usage", {}), "latency": resp.get("latency", 0),
                        "parse_failure": _s4_pf,
                        "schema_failure": not _s4_pf and not bool(parsed.get("action")),
                    })

                probe_results.append({
                    "t": t,
                    "transform": transform_name,
                    "is_valid": True,
                    "validity_type": validity_type,
                    "action_ctl": actions["control"],
                    "action_cf": actions["counterfactual"],
                    "eu_ctl": eu_ctl,
                    "eu_cf": eu_cf,
                    "eu_margin_ctl": margin_ctl,
                    "eu_margin_cf": margin_cf,
                    "margin_delta": margin_delta,
                    "eu_action_flip": eu_action_flip,
                    "js_ctl_cf": js_ctl_cf,
                    "valid_strict": valid_strict, "valid_default": valid_default,
                    "valid_lenient": valid_lenient,
                    "dir_acc_cf": 1 if actions["counterfactual"] == eu_cf else 0,
                    "sensitive": 1 if actions["control"] != actions["counterfactual"] else 0,
                })

        return {"probe_results": probe_results, "api_log": api_log}

    # ──────────────────────────────────────────────────
    # Stage 5: Cognitive Metrics (0 calls, offline)
    # ──────────────────────────────────────────────────
    def _stage5(self, game_name: str, opp_type: str, seed: int, s1: dict) -> dict:
        probe_data = s1.get("probe_data", [])
        if not probe_data:
            return {}

        # H1: Bayesian belief updating — JS and KL divergence
        js_values, kl_values = [], []
        for pd in probe_data:
            op = pd.get("oracle_posterior", {})
            mp = pd.get("posterior_over_types", {})
            if op and mp:
                js_values.append(js_divergence(op, mp))
                kl_values.append(kl_divergence(op, mp))

        # H4: Metacognitive calibration
        calibration_data = []
        for pd in probe_data:
            conf = pd.get("confidence", 0.0)
            pred = pd.get("predicted_next_action", "")
            actual = pd.get("actual_next_opp_action", "")
            if pred and actual:
                calibration_data.append((conf, pred == actual))

        # H7: Surprisal-update coupling
        surprisal_values, update_magnitudes = [], []
        sorted_probes = sorted(probe_data, key=lambda p: p["t"])
        for i, pd in enumerate(sorted_probes):
            pred = pd.get("predicted_next_action", "")
            actual = pd.get("actual_next_opp_action", "")
            if pred and actual:
                pred_prob = pd.get("confidence", 0.5) if pred == actual else (1 - pd.get("confidence", 0.5))
                pred_prob = max(pred_prob, 0.01)
                surprisal_values.append(surprisal(pred_prob))
            if i > 0:
                prev_p = sorted_probes[i - 1].get("posterior_over_types", {})
                curr_p = pd.get("posterior_over_types", {})
                if prev_p and curr_p:
                    update_magnitudes.append(belief_update_magnitude(curr_p, prev_p))

        metrics = {}
        if js_values:
            metrics["h1_mean_js"] = sum(js_values) / len(js_values)
            metrics["h1_mean_kl"] = sum(kl_values) / len(kl_values)
        if calibration_data:
            metrics["h4_ece"] = expected_calibration_error(calibration_data)
            metrics["h4_mce"] = maximum_calibration_error(calibration_data)
        if surprisal_values:
            metrics["h7_mean_surprisal"] = sum(surprisal_values) / len(surprisal_values)
        if update_magnitudes:
            metrics["h7_mean_update_magnitude"] = sum(update_magnitudes) / len(update_magnitudes)
        # Surprisal-update correlation
        if len(surprisal_values) > 1 and len(update_magnitudes) > 1:
            min_len = min(len(surprisal_values) - 1, len(update_magnitudes))
            if min_len > 1:
                s = surprisal_values[:min_len]
                u = update_magnitudes[:min_len]
                metrics["h7_surprisal_update_corr"] = _pearson_corr(s, u)

        return metrics


def _pearson_corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)
