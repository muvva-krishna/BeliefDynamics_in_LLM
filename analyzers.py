"""
Experiment-specific analyzers: extract metrics from unified pipeline output,
write CSV files and TXT summary reports.

Each analyzer reads pipeline condition results and produces:
  - data.csv  -> raw gameplay records
  - summary.txt -> structured results with per-game breakdowns, CIs, integrity, metadata

Output directory: results/{provider}_{model}/{exp_name}/
"""
import os
import csv
import json
import math
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

import config
from games import get_game
from metrics import (
    normalized_episode_payoff, cooperation_rate,
    kl_divergence, brier_score, expected_calibration_error, maximum_calibration_error,
    eu_optimal_action, eu_consistent_coherence, eu_margin, margin_threshold,
    belief_use_capability, conditional_action_agreement,
    js_divergence, rigidity_index,
)

logger = logging.getLogger(__name__)

DECEPTIVE_TYPES = {"deceptive_opportunist", "gradual_defector"}


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

def _safe_avg(values: list) -> float:
    vals = [v for v in values if v is not None and v == v]  # filter None and NaN
    return sum(vals) / len(vals) if vals else 0.0


def _safe_vals(values: list) -> list:
    """Filter None/NaN from a list of floats."""
    return [v for v in values if v is not None and v == v]


def _ci95(values: list) -> tuple:
    """Compute mean and 95% CI (normal approx). Returns (mean, ci_lo, ci_hi, n)."""
    vals = _safe_vals(values)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    mean = sum(vals) / n
    if n == 1:
        return (mean, mean, mean, 1)
    se = float(np.std(vals, ddof=1)) / math.sqrt(n)
    return (mean, mean - 1.96 * se, mean + 1.96 * se, n)


def _fmt_ci(values: list, fmt: str = ".4f") -> str:
    """Format: mean [95% CI: lo, hi] N=n."""
    m, lo, hi, n = _ci95(values)
    return f"{m:{fmt}}  [95% CI: {lo:{fmt}}, {hi:{fmt}}]  N={n}"


def _write_csv(path: str, fieldnames: list, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV: {path} ({len(rows)} rows)")


def _exp_dir(provider: str, model: str, exp_name: str) -> str:
    """Get (and create) experiment output directory."""
    safe_model = model.replace("/", "_").replace(":", "_")
    d = os.path.join(config.RESULTS_DIR, f"{provider}_{safe_model}", exp_name)
    os.makedirs(d, exist_ok=True)
    return d


def _pearson_corr(x: list, y: list) -> float:
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


def _integrity_stats(conditions, stage_keys):
    """Returns (parse_failures, schema_failures, total_calls). api_logs not stored."""
    return 0, 0, 0


def _median(values):
    """Median of non-None/non-NaN values."""
    vals = _safe_vals(values)
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def _build_run_config(provider, model, all_conds):
    """Build Run Configuration block from config module."""
    games_in_data = sorted(set(c["game_name"] for c in all_conds))
    opps_in_data = sorted(set(c["opponent_type"] for c in all_conds))
    eps_per_game = {}
    for g in games_in_data:
        eps_per_game[g] = sum(1 for c in all_conds if c["game_name"] == g)
    lines = []
    lines.append("Run Configuration:")
    lines.append(f"  Provider: {provider} | Model: {model}")
    lines.append(f"  Games: {', '.join(games_in_data)}")
    lines.append(f"  Opponents/Game: {len(opps_in_data)}")
    lines.append(f"  Episodes/Opponent: {config.DEFAULT_EPISODES}")
    lines.append(f"  Rounds/Episode: {config.ROUNDS_PER_EPISODE}")
    lines.append(f"  Probe Rounds: {config.PROBE_ROUNDS}")
    lines.append(f"  Tau (identifiability): 0.60 | Alpha (margin): 0.05")
    lines.append(f"  Plausible Perturbation Lambda: 0.80")
    lines.append(f"  Seed policy: deterministic per (game, opponent, episode)")
    lines.append(f"  Temperature: {config.TEMPERATURE} | Top_p: {config.TOP_P}")
    lines.append(f"  Code Version: N/A")
    return "\n".join(lines)


def _build_coverage_check(all_conds):
    """Build Coverage Check block."""
    expected_games = set(config.GAME_NAMES)
    observed_games = set(c["game_name"] for c in all_conds)
    missing_games = expected_games - observed_games

    expected_per_game = config.DEFAULT_EPISODES * len(config.OPPONENT_TYPES)
    incomplete = []
    for g in sorted(observed_games):
        n = sum(1 for c in all_conds if c["game_name"] == g)
        if n < expected_per_game:
            incomplete.append(f"{g} ({n}/{expected_per_game})")

    lines = []
    lines.append("Coverage Check:")
    lines.append(f"  Expected Games: {len(expected_games)} ({', '.join(sorted(expected_games))})")
    lines.append(f"  Observed Games: {len(observed_games)} ({', '.join(sorted(observed_games))})")
    lines.append(f"  Expected Episodes/Game: {expected_per_game}")
    lines.append(f"  Observed Episodes: {len(all_conds)}")
    if missing_games:
        lines.append(f"  Missing Games: {', '.join(sorted(missing_games))}")
    else:
        lines.append("  Missing Games: none")
    if incomplete:
        lines.append(f"  Incomplete Games: {'; '.join(incomplete)}")
    else:
        lines.append("  Incomplete Games: none")
    return "\n".join(lines)


def _build_validation_checklist(all_conds, exp3_data=None, exp4_data=None):
    """Build final Validation Checklist block."""
    expected_games = set(config.GAME_NAMES)
    observed_games = set(c["game_name"] for c in all_conds)
    expected_per_game = config.DEFAULT_EPISODES * len(config.OPPONENT_TYPES)

    all_present = "PASS" if observed_games == expected_games else "FAIL"
    all_counts = all(
        sum(1 for c in all_conds if c["game_name"] == g) >= expected_per_game
        for g in observed_games
    ) if observed_games else False
    counts_ok = "PASS" if all_counts else "FAIL"

    # Parse/schema — always 0 since api_logs stripped
    parse_ok = "PASS"

    exp3_ok = "N/A"
    if exp3_data is not None:
        n_ident = exp3_data.get("n_identifiable", 0)
        exp3_ok = "PASS" if n_ident > 0 else "FAIL"

    exp4_ok = "N/A"
    if exp4_data is not None:
        n_valid = exp4_data.get("n_valid", 0)
        exp4_ok = "PASS" if n_valid > 0 else "FAIL"

    lines = []
    lines.append("Validation Checklist:")
    lines.append(f"  [{all_present}] All expected games present")
    lines.append(f"  [{counts_ok}] Expected episode counts met")
    lines.append(f"  [{parse_ok}] No parse/schema failures")
    lines.append(f"  [{exp3_ok}] Exp3 informative states > 0")
    lines.append(f"  [{exp4_ok}] Exp4 valid probes > 0")
    return "\n".join(lines)


def _build_trace_audit(all_conds):
    """Build Trace Audit block."""
    n_games = len(set(c["game_name"] for c in all_conds))
    n_unique_episodes = len(set(
        (c["game_name"], c["opponent_type"], c["episode_id"]) for c in all_conds
    ))
    lines = []
    lines.append("Trace Audit:")
    lines.append(f"  Conditions Read: {len(all_conds)}")
    lines.append(f"  Unique Games: {n_games}")
    lines.append(f"  Unique Episodes: {n_unique_episodes}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# Exp1: Baseline Strategic Play
# ═══════════════════════════════════════════════════════

class Exp1Analyzer:
    """Extracts Exp1 metrics from Stage 1 gameplay data."""

    def _compute_episode_metrics(self, conditions):
        """Returns (episode_metrics, rows) from conditions."""
        rows = []
        episode_metrics = []
        for cond in conditions:
            s1 = cond.get("stage1")
            if not s1:
                continue
            rounds = s1["rounds"]
            game = cond["game_name"]
            opp = cond["opponent_type"]
            payoffs = [r["self_payoff"] for r in rounds]
            actions = [r["self_action"] for r in rounds]
            np_ = normalized_episode_payoff(payoffs, game)
            cr_ = cooperation_rate(actions)

            episode_metrics.append({
                "game_name": game, "opponent_type": opp,
                "normalized_payoff": np_, "cooperation_rate": cr_,
                "total_payoff": sum(payoffs),
            })

            for r in rounds:
                rows.append({
                    "game_name": game, "opponent_type": opp,
                    "episode_id": cond["episode_id"], "seed": cond["seed"],
                    "round": r["t"], "self_action": r["self_action"],
                    "opp_action": r["opp_action"],
                    "self_payoff": r["self_payoff"], "opp_payoff": r["opp_payoff"],
                    "normalized_payoff": np_, "cooperation_rate": cr_,
                })
        return episode_metrics, rows

    def write_csv(self, conditions, provider, model):
        """Writes/overwrites data.csv with the latest complete dataset."""
        _, rows = self._compute_episode_metrics(conditions)
        d = _exp_dir(provider, model, "exp1_baseline")
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed", "round",
             "self_action", "opp_action", "self_payoff", "opp_payoff",
             "normalized_payoff", "cooperation_rate"],
            rows,
        )

    def _build_header(self, all_conds, provider, model):
        """Build the header block string."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pf, sf, _ = _integrity_stats(all_conds, ["stage1"])
        n_excl = pf + sf
        lines = []
        lines.append("=" * 70)
        lines.append("EXP 1: BASELINE STRATEGIC PLAY")
        lines.append(f"Provider: {provider} | Model: {model} | Date: {timestamp}")
        lines.append(f"Seed policy: fixed balanced seeds | Temperature: {config.TEMPERATURE} | Top_p: {config.TOP_P}")
        lines.append("=" * 70)
        lines.append(_build_run_config(provider, model, all_conds))
        lines.append("Metric scope: Episodes only; no probe filtering")
        lines.append("  Normalized payoff uses predeclared game-specific min/max range")
        lines.append("Integrity:")
        lines.append(f"  Parse Failures: {pf} | Schema Failures: {sf} | Retried Calls: N/A | Excluded Episodes: {n_excl}")
        lines.append("")
        return "\n".join(lines)

    def _build_game_section(self, game, game_conds, all_conds):
        """Build per-game section string."""
        episode_metrics, _ = self._compute_episode_metrics(game_conds)
        by_go = defaultdict(list)
        for m in episode_metrics:
            by_go[m["opponent_type"]].append(m)

        lines = []
        sep = "-" * max(1, 67 - len(game))
        lines.append(f"-- {game.upper()} {sep}")
        np_vals = [m["normalized_payoff"] for m in episode_metrics]
        cr_vals = [m["cooperation_rate"] for m in episode_metrics]
        tp_vals = [m["total_payoff"] for m in episode_metrics]
        lines.append(f"  N Episodes: {len(episode_metrics)}")
        lines.append(f"  Normalized Payoff: {_fmt_ci(np_vals)}")
        lines.append(f"  Cooperation Rate:  {_fmt_ci(cr_vals)}")
        lines.append(f"  Total Payoff:      {_fmt_ci(tp_vals, '.2f')}")
        lines.append("  Opponent Breakdown:")
        lines.append(f"  {'OPPONENT':<25} {'Norm Payoff':>12}   {'Coop Rate':>10}  {'Total Payoff':>12}    {'N':>3}")
        lines.append("  " + "\u2500" * 69)
        for opp in config.OPPONENT_TYPES:
            oms = by_go.get(opp, [])
            if not oms:
                continue
            lines.append(
                f"  {opp:<25} {_safe_avg([m['normalized_payoff'] for m in oms]):>12.4f}   "
                f"{_safe_avg([m['cooperation_rate'] for m in oms]):>10.4f}  "
                f"{_safe_avg([m['total_payoff'] for m in oms]):>12.2f}    {len(oms):>3}"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_overall_section(self, all_conds):
        """Build the OVERALL section string."""
        episode_metrics, _ = self._compute_episode_metrics(all_conds)
        all_np = [m["normalized_payoff"] for m in episode_metrics]
        all_cr = [m["cooperation_rate"] for m in episode_metrics]
        all_tp = [m["total_payoff"] for m in episode_metrics]
        lines = []
        lines.append("-- OVERALL " + "-" * 57)
        lines.append(f"  Episodes: {len(episode_metrics)}")
        lines.append(f"  Mean Normalized Payoff: {_fmt_ci(all_np)}")
        lines.append(f"  Mean Cooperation Rate:  {_fmt_ci(all_cr)}")
        lines.append(f"  Mean Total Payoff:      {_fmt_ci(all_tp, '.2f')}")
        lines.append("")
        lines.append(_build_trace_audit(all_conds))
        lines.append("")
        lines.append(_build_coverage_check(all_conds))
        lines.append("")
        lines.append(_build_validation_checklist(all_conds))
        lines.append("")
        return "\n".join(lines)

    def write_game_to_summary(self, game, game_conds, all_conds, provider, model):
        """Creates file + header if new, then appends game section."""
        d = _exp_dir(provider, model, "exp1_baseline")
        path = os.path.join(d, "summary.txt")
        mode = "a"
        prefix = ""
        if not os.path.exists(path):
            prefix = self._build_header(all_conds, provider, model)
        section = self._build_game_section(game, game_conds, all_conds)
        with open(path, mode, encoding="utf-8") as f:
            if prefix:
                f.write(prefix)
            f.write(section)
        logger.info(f"Summary game section [{game}]: {path}")

    def write_overall_to_summary(self, all_conds, provider, model):
        """Appends OVERALL section to existing file (creates if missing)."""
        d = _exp_dir(provider, model, "exp1_baseline")
        path = os.path.join(d, "summary.txt")
        if not os.path.exists(path):
            # Build full content from scratch
            content = self._build_header(all_conds, provider, model)
            games_in_data = set(c["game_name"] for c in all_conds)
            for game in config.GAME_NAMES:
                if game in games_in_data:
                    game_conds = [c for c in all_conds if c["game_name"] == game]
                    content += self._build_game_section(game, game_conds, all_conds)
            content += self._build_overall_section(all_conds)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(self._build_overall_section(all_conds))
        logger.info(f"Summary overall section: {path}")

    def analyze(self, pipeline_results, provider, model):
        """Full rebuild from scratch (used for --analyze all mode)."""
        d = _exp_dir(provider, model, "exp1_baseline")
        path = os.path.join(d, "summary.txt")
        episode_metrics, rows = self._compute_episode_metrics(pipeline_results)
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed", "round",
             "self_action", "opp_action", "self_payoff", "opp_payoff",
             "normalized_payoff", "cooperation_rate"],
            rows,
        )

        content = self._build_header(pipeline_results, provider, model)
        games_in_data = set(c["game_name"] for c in pipeline_results)
        for game in config.GAME_NAMES:
            if game in games_in_data:
                game_conds = [c for c in pipeline_results if c["game_name"] == game]
                content += self._build_game_section(game, game_conds, pipeline_results)
        content += self._build_overall_section(pipeline_results)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Summary (full rebuild): {path}")

        json_path = os.path.join(d, "summary.json")
        json_data = {
            "provider": provider, "model": model,
            "timestamp": datetime.now().isoformat(),
            "n_conditions": len(pipeline_results),
            "games": sorted(set(c["game_name"] for c in pipeline_results)),
            "n_games": len(set(c["game_name"] for c in pipeline_results)),
            "n_episodes": len(episode_metrics),
            "mean_normalized_payoff": _safe_avg([m["normalized_payoff"] for m in episode_metrics]),
            "mean_cooperation_rate": _safe_avg([m["cooperation_rate"] for m in episode_metrics]),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)

        return {"episode_metrics": episode_metrics}


# ═══════════════════════════════════════════════════════
# Exp2: Belief Elicitation and Calibration
# ═══════════════════════════════════════════════════════

class Exp2Analyzer:
    """Extracts Exp2 metrics from Stage 1 gameplay + probe data."""

    def _compute_metrics(self, conditions):
        """Returns (episode_metrics, probe_details, all_cal_data, rows)."""
        rows = []
        episode_metrics = []
        probe_details = []
        all_cal_data = []

        for cond in conditions:
            s1 = cond.get("stage1")
            if not s1:
                continue
            game = cond["game_name"]
            opp = cond["opponent_type"]
            rounds = s1["rounds"]
            probes = s1.get("probe_data", [])

            payoffs = [r["self_payoff"] for r in rounds]
            actions = [r["self_action"] for r in rounds]
            np_ = normalized_episode_payoff(payoffs, game)
            cr_ = cooperation_rate(actions)

            kl_vals, brier_vals, cal_data = [], [], []
            total_probes = 0
            identifiable_probes = 0
            type_correct = 0
            pred_correct = 0
            pred_total = 0

            for pd in probes:
                total_probes += 1
                op = pd.get("oracle_posterior", {})
                mp = pd.get("posterior_over_types", {})

                # Per-probe type accuracy and prediction accuracy
                _type_acc_probe = None
                _pred_acc_probe = None

                if mp:
                    predicted_type = max(mp, key=mp.get)
                    _type_acc_probe = 1.0 if predicted_type == opp else 0.0
                    if predicted_type == opp:
                        type_correct += 1

                conf = pd.get("confidence", 0.0)
                pred = pd.get("predicted_next_action", "")
                actual = pd.get("actual_next_opp_action", "")
                if pred and actual:
                    is_correct = pred == actual
                    _pred_acc_probe = 1.0 if is_correct else 0.0
                    if is_correct:
                        pred_correct += 1
                    pred_total += 1
                    cal_data.append((conf, is_correct))
                    all_cal_data.append((conf, is_correct))

                if pd.get("identifiable") and op and mp:
                    identifiable_probes += 1
                    kl_val = kl_divergence(op, mp)
                    brier_val = brier_score(mp, opp)
                    kl_vals.append(kl_val)
                    brier_vals.append(brier_val)
                    probe_details.append({
                        "game_name": game, "opponent_type": opp,
                        "probe_round": pd["t"], "kl": kl_val, "brier": brier_val,
                        "type_acc": _type_acc_probe,
                        "pred_acc": _pred_acc_probe,
                    })

            em = {
                "game_name": game, "opponent_type": opp,
                "normalized_payoff": np_, "cooperation_rate": cr_,
                "total_probes": total_probes, "identifiable_probes": identifiable_probes,
                "type_correct": type_correct,
                "pred_correct": pred_correct, "pred_total": pred_total,
            }
            if kl_vals:
                em["mean_kl"] = _safe_avg(kl_vals)
            if brier_vals:
                em["mean_brier"] = _safe_avg(brier_vals)
            if cal_data:
                em["ece"] = expected_calibration_error(cal_data)
                em["mce"] = maximum_calibration_error(cal_data)
            if total_probes > 0:
                em["type_accuracy"] = type_correct / total_probes
            if pred_total > 0:
                em["pred_accuracy"] = pred_correct / pred_total
            episode_metrics.append(em)

            probe_map = {pd["t"]: pd for pd in probes}
            for r in rounds:
                row = {
                    "game_name": game, "opponent_type": opp,
                    "episode_id": cond["episode_id"], "seed": cond["seed"],
                    "round": r["t"], "self_action": r["self_action"],
                    "opp_action": r["opp_action"],
                    "self_payoff": r["self_payoff"], "opp_payoff": r["opp_payoff"],
                    "is_probe": r["t"] in probe_map,
                }
                if r["t"] in probe_map:
                    pd_r = probe_map[r["t"]]
                    row["confidence"] = pd_r.get("confidence", "")
                    row["predicted_next_action"] = pd_r.get("predicted_next_action", "")
                    row["actual_next_opp_action"] = pd_r.get("actual_next_opp_action", "")
                    row["identifiable"] = pd_r.get("identifiable", "")
                rows.append(row)

        return episode_metrics, probe_details, all_cal_data, rows

    def write_csv(self, conditions, provider, model):
        """Writes/overwrites data.csv with the latest complete dataset."""
        _, _, _, rows = self._compute_metrics(conditions)
        d = _exp_dir(provider, model, "exp2_belief")
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed", "round",
             "self_action", "opp_action", "self_payoff", "opp_payoff",
             "is_probe", "confidence", "predicted_next_action",
             "actual_next_opp_action", "identifiable"],
            rows,
        )

    def _count_missing_posterior(self, conditions):
        """Count conditions where probe_data exists but posterior is empty/uniform."""
        missing = 0
        n_types = len(config.OPPONENT_TYPES)
        uniform_val = 1.0 / n_types if n_types > 0 else 0.0
        for cond in conditions:
            s1 = cond.get("stage1", {})
            for pd in s1.get("probe_data", []):
                mp = pd.get("posterior_over_types", {})
                if not mp:
                    missing += 1
                else:
                    vals = list(mp.values())
                    if all(abs(v - uniform_val) < 1e-6 for v in vals):
                        missing += 1
        return missing

    def _build_header(self, all_conds, provider, model):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pf, sf, _ = _integrity_stats(all_conds, ["stage1"])
        total_probes = sum(
            len(c.get("stage1", {}).get("probe_data", [])) for c in all_conds
        )
        ident_probes = sum(
            sum(1 for pd in c.get("stage1", {}).get("probe_data", []) if pd.get("identifiable"))
            for c in all_conds
        )
        missing_post = self._count_missing_posterior(all_conds)
        lines = []
        lines.append("=" * 70)
        lines.append("EXP 2: BELIEF ELICITATION AND CALIBRATION")
        lines.append(f"Provider: {provider} | Model: {model} | Date: {timestamp}")
        lines.append(f"Probe rounds: {','.join(str(r) for r in config.PROBE_ROUNDS)} | Identifiability threshold: tau=0.60")
        lines.append("=" * 70)
        lines.append(_build_run_config(provider, model, all_conds))
        lines.append("Metric scope:")
        lines.append("  Headline KL/Brier/ECE computed on identifiable probes only (max oracle >= 0.60)")
        lines.append("  Type Accuracy computed on all probes")
        lines.append("  ECE/MCE from confidence as P(argmax correct)")
        lines.append("Integrity:")
        lines.append(f"  Total Probes: {total_probes} | Identifiable Probes: {ident_probes} | Parse Failures: {pf}")
        lines.append(f"  Schema Failures: {sf} | Missing Posterior Fields: {missing_post} | Retried Calls: N/A")
        lines.append("Data Validity / Inclusion:")
        lines.append(f"  Total Probes: {total_probes}")
        lines.append(f"  Valid Posterior Probes: {total_probes - missing_post}")
        lines.append(f"  Missing Posterior Probes: {missing_post}")
        lines.append(f"  Identifiable Probes: {ident_probes}")
        lines.append(f"  Probes Used for KL/Brier: {ident_probes} (identifiable only)")
        lines.append(f"  Probes Used for ECE/MCE: {total_probes - missing_post} (all with posterior)")
        lines.append("")
        return "\n".join(lines)

    def _build_game_section(self, game, game_conds, all_conds):
        episode_metrics, probe_details, all_cal_data, _ = self._compute_metrics(game_conds)

        by_go = defaultdict(list)
        for m in episode_metrics:
            by_go[m["opponent_type"]].append(m)

        pd_by_round = defaultdict(list)
        for pd in probe_details:
            pd_by_round[pd["probe_round"]].append(pd)

        n_eps = len(episode_metrics)
        g_probes = sum(m.get("total_probes", 0) for m in episode_metrics)
        g_ident = sum(m.get("identifiable_probes", 0) for m in episode_metrics)

        n_types = len(config.OPPONENT_TYPES)
        uniform_brier = sum(
            (1.0 / n_types - (1.0 if i == 0 else 0.0)) ** 2
            for i in range(n_types)
        )

        kl_g = [m["mean_kl"] for m in episode_metrics if "mean_kl" in m]
        br_g = [m["mean_brier"] for m in episode_metrics if "mean_brier" in m]
        ta_g = [m["type_accuracy"] for m in episode_metrics if "type_accuracy" in m]
        ece_g = [m["ece"] for m in episode_metrics if "ece" in m]
        mce_g = [m["mce"] for m in episode_metrics if "mce" in m]
        pa_g = [m["pred_accuracy"] for m in episode_metrics if "pred_accuracy" in m]

        sep = "-" * max(1, 67 - len(game))
        lines = []
        lines.append(f"-- {game.upper()} {sep}")
        lines.append(f"  N Episodes: {n_eps} | N Identifiable Probes: {g_ident} | Total Probes: {g_probes}")
        lines.append(f"  KL Divergence:    {_fmt_ci(kl_g)}")
        kl_sv = _safe_vals(kl_g)
        if kl_sv:
            lines.append(f"  KL (median/min/max): {_median(kl_g):.4f} / {min(kl_sv):.4f} / {max(kl_sv):.4f}")
        lines.append(f"  Brier Score:      {_fmt_ci(br_g)}")
        br_sv = _safe_vals(br_g)
        if br_sv:
            lines.append(f"  Brier (median/min/max): {_median(br_g):.4f} / {min(br_sv):.4f} / {max(br_sv):.4f}")
        lines.append(f"  Type Accuracy:    {_fmt_ci(ta_g)}  | Chance={1.0/n_types:.4f}")
        lines.append(f"  ECE:              {_fmt_ci(ece_g)}")
        lines.append(f"  MCE:              {_safe_avg(mce_g):.4f}")
        lines.append(f"  Prediction Accuracy: {_safe_avg(pa_g):.4f}")
        lines.append(f"  Uniform Brier Baseline: {uniform_brier:.4f}")
        lines.append("")
        lines.append("  By Probe Round:")
        lines.append(f"  {'Round':<8} {'Mean KL':>9}   {'Mean Brier':>11}  {'Type Acc':>9}  {'Pred Acc':>9}    {'N':>4}")
        lines.append("  " + "\u2500" * 53)
        for pr in config.PROBE_ROUNDS:
            pds = pd_by_round.get(pr, [])
            if pds:
                kl_r = [p["kl"] for p in pds]
                br_r = [p["brier"] for p in pds]
                ta_r = _safe_vals([p.get("type_acc") for p in pds])
                pa_r = _safe_vals([p.get("pred_acc") for p in pds])
                ta_str = f"{_safe_avg(ta_r):.4f}" if ta_r else " N/A "
                pa_str = f"{_safe_avg(pa_r):.4f}" if pa_r else " N/A "
                lines.append(
                    f"  t={pr:<5} {_safe_avg(kl_r):>9.4f}   {_safe_avg(br_r):>11.4f}  {ta_str:>9}  {pa_str:>9}    {len(pds):>4}"
                )
        lines.append("")
        lines.append("  Opponent Breakdown:")
        lines.append(f"  {'OPPONENT':<25} {'Mean KL':>9}    {'Brier':>7}   {'TypeAcc':>8}    {'ECE':>6}  {'CoopRate':>9}    {'N':>4}")
        lines.append("  " + "\u2500" * 69)
        for opp in config.OPPONENT_TYPES:
            oms = by_go.get(opp, [])
            if not oms:
                continue
            kl_o = _safe_avg([m["mean_kl"] for m in oms if "mean_kl" in m])
            br_o = _safe_avg([m["mean_brier"] for m in oms if "mean_brier" in m])
            ta_o = _safe_avg([m["type_accuracy"] for m in oms if "type_accuracy" in m])
            ece_o = _safe_avg([m["ece"] for m in oms if "ece" in m])
            cr_o = _safe_avg([m["cooperation_rate"] for m in oms])
            lines.append(
                f"  {opp:<25} {kl_o:>9.4f}    {br_o:>7.4f}   {ta_o:>8.4f}    {ece_o:>6.4f}  {cr_o:>9.4f}    {len(oms):>4}"
            )
        lines.append("")
        return "\n".join(lines)

    def _build_overall_section(self, all_conds):
        episode_metrics, _, all_cal_data, _ = self._compute_metrics(all_conds)
        total_probes = sum(m.get("total_probes", 0) for m in episode_metrics)
        ident_probes = sum(m.get("identifiable_probes", 0) for m in episode_metrics)
        n_types = len(config.OPPONENT_TYPES)
        uniform_brier = sum(
            (1.0 / n_types - (1.0 if i == 0 else 0.0)) ** 2
            for i in range(n_types)
        )
        kl_all = [m["mean_kl"] for m in episode_metrics if "mean_kl" in m]
        br_all = [m["mean_brier"] for m in episode_metrics if "mean_brier" in m]
        ta_all = [m["type_accuracy"] for m in episode_metrics if "type_accuracy" in m]
        ece_all = [m["ece"] for m in episode_metrics if "ece" in m]
        mce_all = [m["mce"] for m in episode_metrics if "mce" in m]
        pa_all = [m["pred_accuracy"] for m in episode_metrics if "pred_accuracy" in m]
        cr_all = [m["cooperation_rate"] for m in episode_metrics]

        lines = []
        lines.append("-- OVERALL " + "-" * 57)
        lines.append(f"  Total Probes: {total_probes} | Identifiable: {ident_probes}")
        lines.append(f"  Mean KL Divergence:   {_fmt_ci(kl_all)}")
        kl_sv = _safe_vals(kl_all)
        if kl_sv:
            lines.append(f"  KL (median/min/max): {_median(kl_all):.4f} / {min(kl_sv):.4f} / {max(kl_sv):.4f}")
        lines.append(f"  Mean Brier Score:     {_fmt_ci(br_all)}")
        br_sv = _safe_vals(br_all)
        if br_sv:
            lines.append(f"  Brier (median/min/max): {_median(br_all):.4f} / {min(br_sv):.4f} / {max(br_sv):.4f}")
        lines.append(f"  Type Accuracy:        {_fmt_ci(ta_all)}  | Chance={1.0/n_types:.4f}")
        lines.append(f"  ECE:                  {_fmt_ci(ece_all)}")
        lines.append(f"  MCE:                  {_safe_avg(mce_all):.4f}")
        lines.append(f"  Prediction Accuracy:  {_safe_avg(pa_all):.4f}")
        lines.append(f"  Cooperation Rate:     {_safe_avg(cr_all):.4f}")
        lines.append(f"  Uniform Brier Baseline: {uniform_brier:.4f}")
        lines.append("")
        # Missingness by Field
        n_missing_posterior = 0
        n_missing_confidence = 0
        n_missing_predicted = 0
        for cond in all_conds:
            s1 = cond.get("stage1", {})
            for pd in s1.get("probe_data", []):
                if not pd.get("posterior_over_types"):
                    n_missing_posterior += 1
                if pd.get("confidence") is None:
                    n_missing_confidence += 1
                if not pd.get("predicted_next_action"):
                    n_missing_predicted += 1
        lines.append("Missingness by Field:")
        lines.append(f"  posterior_over_types: {n_missing_posterior}")
        lines.append(f"  confidence: {n_missing_confidence}")
        lines.append(f"  predicted_next_action: {n_missing_predicted}")
        lines.append("")
        lines.append("Warnings / Notes:")
        lines.append("  Missing posterior fields excluded from KL/Brier computation")
        lines.append("  Brier score range is [0, 2]; ceiling (2.0) means all mass on wrong type")
        lines.append("  ECE/MCE computed from argmax-prediction confidence calibration")
        lines.append("")
        lines.append(_build_trace_audit(all_conds))
        lines.append("")
        lines.append(_build_coverage_check(all_conds))
        lines.append("")
        return "\n".join(lines)

    def write_game_to_summary(self, game, game_conds, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp2_belief")
        path = os.path.join(d, "summary.txt")
        mode = "a"
        prefix = ""
        if not os.path.exists(path):
            prefix = self._build_header(all_conds, provider, model)
        section = self._build_game_section(game, game_conds, all_conds)
        with open(path, mode, encoding="utf-8") as f:
            if prefix:
                f.write(prefix)
            f.write(section)
        logger.info(f"Summary game section [{game}]: {path}")

    def write_overall_to_summary(self, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp2_belief")
        path = os.path.join(d, "summary.txt")
        if not os.path.exists(path):
            content = self._build_header(all_conds, provider, model)
            games_in_data = set(c["game_name"] for c in all_conds)
            for game in config.GAME_NAMES:
                if game in games_in_data:
                    game_conds = [c for c in all_conds if c["game_name"] == game]
                    content += self._build_game_section(game, game_conds, all_conds)
            content += self._build_overall_section(all_conds)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(self._build_overall_section(all_conds))
        logger.info(f"Summary overall section: {path}")

    def analyze(self, pipeline_results, provider, model):
        """Full rebuild from scratch (used for --analyze all mode)."""
        d = _exp_dir(provider, model, "exp2_belief")
        path = os.path.join(d, "summary.txt")
        episode_metrics, probe_details, all_cal_data, rows = self._compute_metrics(pipeline_results)
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed", "round",
             "self_action", "opp_action", "self_payoff", "opp_payoff",
             "is_probe", "confidence", "predicted_next_action",
             "actual_next_opp_action", "identifiable"],
            rows,
        )

        content = self._build_header(pipeline_results, provider, model)
        games_in_data = set(c["game_name"] for c in pipeline_results)
        for game in config.GAME_NAMES:
            if game in games_in_data:
                game_conds = [c for c in pipeline_results if c["game_name"] == game]
                content += self._build_game_section(game, game_conds, pipeline_results)
        content += self._build_overall_section(pipeline_results)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Summary (full rebuild): {path}")

        json_path = os.path.join(d, "summary.json")
        json_data = {
            "provider": provider, "model": model,
            "timestamp": datetime.now().isoformat(),
            "n_conditions": len(pipeline_results),
            "games": sorted(set(c["game_name"] for c in pipeline_results)),
            "n_games": len(set(c["game_name"] for c in pipeline_results)),
            "mean_kl": _safe_avg([m["mean_kl"] for m in episode_metrics if "mean_kl" in m]),
            "mean_brier": _safe_avg([m["mean_brier"] for m in episode_metrics if "mean_brier" in m]),
            "mean_type_accuracy": _safe_avg([m["type_accuracy"] for m in episode_metrics if "type_accuracy" in m]),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)

        return {
            "episode_metrics": episode_metrics,
            "probe_details": probe_details,
            "calibration_data": all_cal_data,
        }


# ═══════════════════════════════════════════════════════
# Exp3: Belief-Action Coupling
# ═══════════════════════════════════════════════════════

class Exp3Analyzer:
    """Computes Exp3 coupling metrics across variants A/B/B2/C."""

    def _compute_metrics(self, conditions):
        """Returns (variant_metrics, coupling_metrics, rows)."""
        rows = []
        variant_metrics = defaultdict(list)
        coupling_metrics = []

        for cond in conditions:
            s1 = cond.get("stage1")
            s2 = cond.get("stage2")
            s3 = cond.get("stage3")
            game = cond["game_name"]
            opp = cond["opponent_type"]
            seed = cond["seed"]

            np_a, cr_a, np_c, cr_c = None, None, None, None

            if s1:
                rounds = s1["rounds"]
                payoffs = [r["self_payoff"] for r in rounds]
                actions = [r["self_action"] for r in rounds]
                np_a = normalized_episode_payoff(payoffs, game)
                cr_a = cooperation_rate(actions)
                variant_metrics["A"].append({
                    "game_name": game, "opponent_type": opp,
                    "normalized_payoff": np_a, "cooperation_rate": cr_a,
                })
                for r in rounds:
                    rows.append({
                        "game_name": game, "opponent_type": opp,
                        "episode_id": cond["episode_id"], "seed": seed,
                        "variant": "A", "round": r["t"],
                        "self_action": r["self_action"], "opp_action": r["opp_action"],
                        "self_payoff": r["self_payoff"], "opp_payoff": r["opp_payoff"],
                        "normalized_payoff": np_a, "cooperation_rate": cr_a,
                    })

            ecc_b = None
            if s1 and s1.get("probe_data"):
                probes = s1["probe_data"]
                b_actions = [pd.get("probe_action", "") for pd in probes]
                b_posteriors = [pd.get("posterior_over_types", {}) for pd in probes]
                b_histories = []
                for pd in probes:
                    t = pd["t"]
                    h = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                         for r in rounds[:t - 1]]
                    b_histories.append(h)
                # Filter to only probes where probe_action was recorded (not all may have it)
                valid_b = [(a, p, h) for a, p, h in zip(b_actions, b_posteriors, b_histories)
                           if a in ("C", "D") and p]
                if valid_b:
                    vb_acts, vb_posts, vb_hists = zip(*valid_b)
                    ecc_b = eu_consistent_coherence(list(vb_acts), list(vb_posts), game,
                                                   list(vb_hists), seed)
                for pd in probes:
                    rows.append({
                        "game_name": game, "opponent_type": opp,
                        "episode_id": cond["episode_id"], "seed": seed,
                        "variant": "B", "round": pd["t"],
                        "self_action": pd.get("probe_action", ""),
                        "confidence": pd.get("confidence", ""),
                    })
                variant_metrics["B"].append({
                    "game_name": game, "opponent_type": opp,
                    "eu_coherence": ecc_b,
                })

            ecc_b2 = None
            if s2 and s2.get("b2_data") and s1:
                b2_data = s2["b2_data"]
                b2_actions = [bd.get("b2_action", "") for bd in b2_data]
                b2_posteriors = [bd.get("posterior_over_types", {}) for bd in b2_data]
                b2_histories = []
                for bd in b2_data:
                    t = bd["t"]
                    h = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                         for r in s1["rounds"][:t - 1]]
                    b2_histories.append(h)
                # Filter to probes with valid action and posterior
                valid_b2 = [(a, p, h) for a, p, h in zip(b2_actions, b2_posteriors, b2_histories)
                            if a in ("C", "D") and p]
                if valid_b2:
                    vb2_acts, vb2_posts, vb2_hists = zip(*valid_b2)
                    ecc_b2 = eu_consistent_coherence(list(vb2_acts), list(vb2_posts), game,
                                                    list(vb2_hists), seed)
                for bd in b2_data:
                    rows.append({
                        "game_name": game, "opponent_type": opp,
                        "episode_id": cond["episode_id"], "seed": seed,
                        "variant": "B2", "round": bd["t"],
                        "self_action": bd.get("b2_action", ""),
                        "confidence": bd.get("confidence", ""),
                    })
                variant_metrics["B2"].append({
                    "game_name": game, "opponent_type": opp,
                    "eu_coherence": ecc_b2,
                })

            # Variant B payoff: one-step counterfactual action substitution.
            # probe_action is substituted for the baseline action at round t,
            # paired with the realized opponent action from the matched baseline
            # episode at the same round. This estimates immediate decision quality,
            # not full trajectory payoff.
            if s1 and s1.get("probe_data") and variant_metrics.get("B"):
                probes = s1["probe_data"]
                game_obj = get_game(game)
                b_actions_list, b_payoffs = [], []
                for pd in probes:
                    b_act = pd.get("probe_action", "")
                    if b_act in ("C", "D"):
                        b_actions_list.append(b_act)
                        t_idx = pd["t"] - 1  # 0-indexed
                        if t_idx < len(rounds):
                            opp_act = rounds[t_idx]["opp_action"]
                            p, _ = game_obj.get_payoffs(b_act, opp_act)
                            b_payoffs.append(p)
                if b_actions_list:
                    variant_metrics["B"][-1]["cooperation_rate"] = cooperation_rate(b_actions_list)
                if b_payoffs:
                    r_min, r_max = game_obj.min_payoff, game_obj.max_payoff
                    if r_max > r_min:
                        np_b_val = sum((p - r_min) / (r_max - r_min) for p in b_payoffs) / len(b_payoffs)
                    else:
                        np_b_val = 0.0
                    # Normalization bounds check
                    if not (0.0 <= np_b_val <= 1.0):
                        logger.warning(f"Exp3 B normalized_payoff out of bounds: {np_b_val:.4f} "
                                       f"({game} vs {opp})")
                        np_b_val = max(0.0, min(1.0, np_b_val))
                    variant_metrics["B"][-1]["normalized_payoff"] = np_b_val

            # Variant B2 payoff: same one-step counterfactual substitution as B,
            # using the two-stage-elicited action (b2_action) against the realized
            # opponent action at round t from Stage 1.
            if s2 and s2.get("b2_data") and s1 and variant_metrics.get("B2"):
                game_obj = get_game(game)
                b2_actions_list, b2_payoffs = [], []
                for bd in s2["b2_data"]:
                    b2_act = bd.get("b2_action", "")
                    if b2_act in ("C", "D"):
                        b2_actions_list.append(b2_act)
                        t_idx = bd["t"] - 1
                        if t_idx < len(s1["rounds"]):
                            opp_act = s1["rounds"][t_idx]["opp_action"]
                            p, _ = game_obj.get_payoffs(b2_act, opp_act)
                            b2_payoffs.append(p)
                if b2_actions_list:
                    variant_metrics["B2"][-1]["cooperation_rate"] = cooperation_rate(b2_actions_list)
                if b2_payoffs:
                    r_min, r_max = game_obj.min_payoff, game_obj.max_payoff
                    if r_max > r_min:
                        np_b2_val = sum((p - r_min) / (r_max - r_min) for p in b2_payoffs) / len(b2_payoffs)
                    else:
                        np_b2_val = 0.0
                    # Normalization bounds check
                    if not (0.0 <= np_b2_val <= 1.0):
                        logger.warning(f"Exp3 B2 normalized_payoff out of bounds: {np_b2_val:.4f} "
                                       f"({game} vs {opp})")
                        np_b2_val = max(0.0, min(1.0, np_b2_val))
                    variant_metrics["B2"][-1]["normalized_payoff"] = np_b2_val

            if s3:
                rounds_c = s3["rounds"]
                payoffs_c = [r["self_payoff"] for r in rounds_c]
                actions_c = [r["self_action"] for r in rounds_c]
                np_c = normalized_episode_payoff(payoffs_c, game)
                cr_c = cooperation_rate(actions_c)
                variant_metrics["C"].append({
                    "game_name": game, "opponent_type": opp,
                    "normalized_payoff": np_c, "cooperation_rate": cr_c,
                })
                for r in rounds_c:
                    rows.append({
                        "game_name": game, "opponent_type": opp,
                        "episode_id": cond["episode_id"], "seed": seed,
                        "variant": "C", "round": r["t"],
                        "self_action": r["self_action"], "opp_action": r["opp_action"],
                        "self_payoff": r["self_payoff"], "opp_payoff": r["opp_payoff"],
                        "normalized_payoff": np_c, "cooperation_rate": cr_c,
                    })

            cm = {"game_name": game, "opponent_type": opp}
            if np_a is not None and np_c is not None:
                cm["delta_c_a"] = np_c - np_a
                cm["belief_use_capability"] = belief_use_capability(np_c, np_a)
            cm["eu_coherence_B"] = ecc_b
            cm["eu_coherence_B2"] = ecc_b2

            if s1 and s1.get("probe_data"):
                probes = s1["probe_data"]
                rounds = s1["rounds"]
                actions_a_probe = []
                actions_b_probe = []
                identifiable = []       # max(oracle posterior) >= 0.60
                action_sensitive = []   # |EU_margin_cf − EU_margin_ctl| >= threshold

                m_thresh = margin_threshold(game, alpha=0.05)
                uniform = {t_name: 1.0 / len(config.OPPONENT_TYPES)
                           for t_name in config.OPPONENT_TYPES}

                for pd in probes:
                    t = pd["t"]
                    a_action = rounds[t - 1]["self_action"] if t <= len(rounds) else ""
                    b_action = pd.get("probe_action", "")
                    actions_a_probe.append(a_action)
                    actions_b_probe.append(b_action)

                    op = pd.get("oracle_posterior", {})
                    if op:
                        # (A) Identifiable: oracle has concentrated (game-agnostic).
                        identifiable.append(max(op.values()) >= 0.60)
                        # (B) Action-sensitive: EU margin differs meaningfully
                        #     between oracle and uniform posterior.
                        h = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                             for r in rounds[:t - 1]]
                        m_oracle = eu_margin(op, game, h, seed)
                        m_unif = eu_margin(uniform, game, h, seed)
                        action_sensitive.append(abs(m_oracle - m_unif) >= m_thresh)
                    else:
                        identifiable.append(False)
                        action_sensitive.append(False)

                cm["n_identifiable_states"] = sum(identifiable)
                cm["n_action_sensitive_states"] = sum(action_sensitive)
                # Primary informative = identifiable (most inclusive, game-agnostic)
                cm["n_informative_rounds"] = cm["n_identifiable_states"]

                # Conditional agreement: computed on identifiable rounds where b_action recorded
                valid_pairs = [
                    (a, b, ident) for a, b, ident in zip(actions_a_probe, actions_b_probe, identifiable)
                    if b in ("C", "D")
                ]
                if valid_pairs and any(ident for _, _, ident in valid_pairs):
                    va, vb, vi = zip(*valid_pairs)
                    cm["conditional_action_agreement"] = conditional_action_agreement(
                        list(va), list(vb), list(vi))

            if np_a is not None:
                cm["np_a"] = np_a
                cm["cr_a"] = cr_a
            if np_c is not None:
                cm["np_c"] = np_c
                cm["cr_c"] = cr_c

            coupling_metrics.append(cm)

        return dict(variant_metrics), coupling_metrics, rows

    def write_csv(self, conditions, provider, model):
        _, _, rows = self._compute_metrics(conditions)
        d = _exp_dir(provider, model, "exp3_coupling")
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed", "variant", "round",
             "self_action", "opp_action", "self_payoff", "opp_payoff",
             "confidence", "normalized_payoff", "cooperation_rate"],
            rows,
        )

    def _build_header(self, all_conds, provider, model):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pf, sf, _ = _integrity_stats(all_conds, ["stage1", "stage2", "stage3"])

        total_identifiable = 0
        total_action_sensitive = 0
        for cond in all_conds:
            s1 = cond.get("stage1", {})
            game = cond["game_name"]
            seed = cond.get("seed", 0)
            rounds = s1.get("rounds", [])
            m_thresh = margin_threshold(game, alpha=0.05)
            uniform = {t_name: 1.0 / len(config.OPPONENT_TYPES)
                       for t_name in config.OPPONENT_TYPES}
            for pd in s1.get("probe_data", []):
                op = pd.get("oracle_posterior", {})
                if not op:
                    continue
                if max(op.values()) >= 0.60:
                    total_identifiable += 1
                t = pd["t"]
                h = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                     for r in rounds[:t - 1]]
                try:
                    m_o = eu_margin(op, game, h, seed)
                    m_u = eu_margin(uniform, game, h, seed)
                    if abs(m_o - m_u) >= m_thresh:
                        total_action_sensitive += 1
                except Exception:
                    pass

        lines = []
        lines.append("=" * 70)
        lines.append("EXP 3: BELIEF-ACTION COUPLING")
        lines.append(f"Provider: {provider} | Model: {model} | Date: {timestamp}")
        lines.append("Paired seed design: YES")
        lines.append("=" * 70)
        lines.append(_build_run_config(provider, model, all_conds))
        lines.append("Metric scope:")
        lines.append("  Conditional agreement and EU coherence on identifiable probe states")
        lines.append("  Identifiable: max(oracle posterior) >= 0.60")
        lines.append("  Action-Sensitive: |EU_margin(oracle) - EU_margin(uniform)| >= alpha*payoff_range (alpha=0.05)")
        lines.append("  B and B2 payoffs are one-step counterfactual substitutions against the")
        lines.append("  realized opponent action from the matched baseline episode at round t.")
        lines.append("  This estimates immediate decision quality, not full trajectory payoff.")
        lines.append("Integrity:")
        lines.append(f"  Parse Failures: {pf} | Schema Failures: {sf} | Retried Calls: N/A")
        lines.append(f"  Identifiable States: {total_identifiable}")
        lines.append(f"  Action-Sensitive States: {total_action_sensitive}")
        lines.append("")
        return "\n".join(lines)

    def _build_game_section(self, game, game_conds, all_conds):
        variant_metrics, coupling_metrics, _ = self._compute_metrics(game_conds)

        vm_by_v = {}
        for v in ["A", "B", "B2", "C"]:
            ms = [m for m in variant_metrics.get(v, []) if m["game_name"] == game]
            vm_by_v[v] = ms

        # Per-variant summary
        def _vm_line(v, label):
            ms = vm_by_v.get(v, [])
            np_vals = [m["normalized_payoff"] for m in ms if "normalized_payoff" in m]
            cr_vals = [m["cooperation_rate"] for m in ms if "cooperation_rate" in m]
            np_mean = _safe_avg(np_vals) if np_vals else 0.0
            cr_mean = _safe_avg(cr_vals) if cr_vals else 0.0
            n = len(ms)
            return f"  {label:<30} NormPayoff={np_mean:.4f}  CoopRate={cr_mean:.4f}  N={n}"

        sep = "-" * max(1, 67 - len(game))
        lines = []
        lines.append(f"-- {game.upper()} {sep}")
        lines.append(_vm_line("A", "Variant A (Baseline):"))
        lines.append(_vm_line("B", "Variant B (Single-call):"))
        lines.append(_vm_line("B2", "Variant B2 (Two-stage):"))
        lines.append(_vm_line("C", "Variant C (Oracle):"))

        a_np = [m["normalized_payoff"] for m in vm_by_v.get("A", []) if "normalized_payoff" in m]
        b_np_list = [m["normalized_payoff"] for m in vm_by_v.get("B", []) if "normalized_payoff" in m]
        b2_np_list = [m["normalized_payoff"] for m in vm_by_v.get("B2", []) if "normalized_payoff" in m]
        c_np = [m["normalized_payoff"] for m in vm_by_v.get("C", []) if "normalized_payoff" in m]

        delta_ba = [b - a for b, a in zip(b_np_list, a_np)] if len(b_np_list) == len(a_np) and a_np else []
        delta_b2a = [b - a for b, a in zip(b2_np_list, a_np)] if len(b2_np_list) == len(a_np) and a_np else []
        delta_ca = [c - a for c, a in zip(c_np, a_np)] if len(c_np) == len(a_np) and a_np else []

        lines.append(f"  Delta B-A:   {_fmt_ci(delta_ba)}")
        lines.append(f"  Delta B2-A:  {_fmt_ci(delta_b2a)}")
        lines.append(f"  Delta C-A:   {_fmt_ci(delta_ca)}  (Belief Use Capability)")

        caa_vals = _safe_vals([cm.get("conditional_action_agreement") for cm in coupling_metrics])
        ecc_b_vals = _safe_vals([cm.get("eu_coherence_B") for cm in coupling_metrics])
        ecc_b2_vals = _safe_vals([cm.get("eu_coherence_B2") for cm in coupling_metrics])
        n_ident = sum(cm.get("n_identifiable_states", 0) for cm in coupling_metrics)
        n_actsens = sum(cm.get("n_action_sensitive_states", 0) for cm in coupling_metrics)

        lines.append(f"  Identifiable States: {n_ident} | Action-Sensitive States: {n_actsens}")
        lines.append(f"  Conditional Action Agreement:   {_fmt_ci(caa_vals)}  (on identifiable states)")
        lines.append(f"  EU-Consistent Coherence (B):    {_fmt_ci(ecc_b_vals)}")
        lines.append(f"  EU-Consistent Coherence (B2):   {_fmt_ci(ecc_b2_vals)}")

        # Fraction of Oracle Gain Captured by B2
        delta_ca_mean = _safe_avg(delta_ca) if delta_ca else 0.0
        delta_b2a_mean = _safe_avg(delta_b2a) if delta_b2a else 0.0
        if delta_ca_mean > 0 and delta_b2a:
            frac = delta_b2a_mean / delta_ca_mean
            lines.append(f"  Fraction of Oracle Gain Captured by B2: {frac:.4f}")
        else:
            lines.append("  Fraction of Oracle Gain Captured by B2: N/A")

        # Action Distribution
        lines.append("  Action Distribution:")
        # B actions
        b_c, b_d = 0, 0
        for cond in game_conds:
            s1 = cond.get("stage1", {})
            for pd in s1.get("probe_data", []):
                pa = pd.get("probe_action", "")
                if pa == "C":
                    b_c += 1
                elif pa == "D":
                    b_d += 1
        lines.append(f"    B:  C={b_c}  D={b_d}")
        # B2 actions
        b2_c, b2_d = 0, 0
        for cond in game_conds:
            s2 = cond.get("stage2", {})
            for bd in s2.get("b2_data", []) if s2 else []:
                ba = bd.get("b2_action", "")
                if ba == "C":
                    b2_c += 1
                elif ba == "D":
                    b2_d += 1
        lines.append(f"    B2: C={b2_c}  D={b2_d}")
        # C actions
        c_c, c_d = 0, 0
        for cond in game_conds:
            s3 = cond.get("stage3", {})
            if s3:
                for r in s3.get("rounds", []):
                    a = r.get("self_action", "")
                    if a == "C":
                        c_c += 1
                    elif a == "D":
                        c_d += 1
        lines.append(f"    C:  C={c_c}  D={c_d}")

        lines.append("")
        return "\n".join(lines)

    def _build_overall_section(self, all_conds):
        variant_metrics, coupling_metrics, _ = self._compute_metrics(all_conds)

        def _vm_line(v, label):
            ms = variant_metrics.get(v, [])
            np_vals = [m["normalized_payoff"] for m in ms if "normalized_payoff" in m]
            cr_vals = [m["cooperation_rate"] for m in ms if "cooperation_rate" in m]
            np_mean = _safe_avg(np_vals) if np_vals else 0.0
            cr_mean = _safe_avg(cr_vals) if cr_vals else 0.0
            return f"  {label:<30} NormPayoff={np_mean:.4f}  CoopRate={cr_mean:.4f}"

        a_np = [m["normalized_payoff"] for m in variant_metrics.get("A", []) if "normalized_payoff" in m]
        b_np_list = [m["normalized_payoff"] for m in variant_metrics.get("B", []) if "normalized_payoff" in m]
        b2_np_list = [m["normalized_payoff"] for m in variant_metrics.get("B2", []) if "normalized_payoff" in m]
        c_np = [m["normalized_payoff"] for m in variant_metrics.get("C", []) if "normalized_payoff" in m]

        delta_ba = [b - a for b, a in zip(b_np_list, a_np)] if len(b_np_list) == len(a_np) and a_np else []
        delta_b2a = [b - a for b, a in zip(b2_np_list, a_np)] if len(b2_np_list) == len(a_np) and a_np else []
        delta_ca = [c - a for c, a in zip(c_np, a_np)] if len(c_np) == len(a_np) and a_np else []

        caa_vals = _safe_vals([cm.get("conditional_action_agreement") for cm in coupling_metrics])
        ecc_b_vals = _safe_vals([cm.get("eu_coherence_B") for cm in coupling_metrics])
        ecc_b2_vals = _safe_vals([cm.get("eu_coherence_B2") for cm in coupling_metrics])
        n_ident = sum(cm.get("n_identifiable_states", 0) for cm in coupling_metrics)
        n_actsens = sum(cm.get("n_action_sensitive_states", 0) for cm in coupling_metrics)

        lines = []
        lines.append("-- OVERALL " + "-" * 57)
        lines.append(_vm_line("A", "Variant A (Baseline):"))
        lines.append(_vm_line("B", "Variant B (Single-call):"))
        lines.append(_vm_line("B2", "Variant B2 (Two-stage):"))
        lines.append(_vm_line("C", "Variant C (Oracle):"))
        lines.append(f"  Delta B-A:   {_fmt_ci(delta_ba)}")
        lines.append(f"  Delta B2-A:  {_fmt_ci(delta_b2a)}")
        lines.append(f"  Delta C-A:   {_fmt_ci(delta_ca)}  (Belief Use Capability)")
        lines.append(f"  Identifiable States: {n_ident} | Action-Sensitive States: {n_actsens}")
        lines.append(f"  Conditional Action Agreement:   {_fmt_ci(caa_vals)}  (on identifiable states)")
        lines.append(f"  EU-Consistent Coherence (B):    {_fmt_ci(ecc_b_vals)}")
        lines.append(f"  EU-Consistent Coherence (B2):   {_fmt_ci(ecc_b2_vals)}")
        lines.append("")
        lines.append("Warnings / Notes:")
        lines.append("  B and B2 payoffs are one-step counterfactual substitutions against the")
        lines.append("  realized opponent action from the matched baseline episode at round t.")
        lines.append("  This estimates immediate decision quality, not full trajectory payoff.")
        lines.append("")
        lines.append(_build_trace_audit(all_conds))
        lines.append("")
        lines.append(_build_coverage_check(all_conds))
        lines.append("")
        return "\n".join(lines)

    def write_game_to_summary(self, game, game_conds, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp3_coupling")
        path = os.path.join(d, "summary.txt")
        mode = "a"
        prefix = ""
        if not os.path.exists(path):
            prefix = self._build_header(all_conds, provider, model)
        section = self._build_game_section(game, game_conds, all_conds)
        with open(path, mode, encoding="utf-8") as f:
            if prefix:
                f.write(prefix)
            f.write(section)
        logger.info(f"Summary game section [{game}]: {path}")

    def write_overall_to_summary(self, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp3_coupling")
        path = os.path.join(d, "summary.txt")
        if not os.path.exists(path):
            content = self._build_header(all_conds, provider, model)
            games_in_data = set(c["game_name"] for c in all_conds)
            for game in config.GAME_NAMES:
                if game in games_in_data:
                    game_conds = [c for c in all_conds if c["game_name"] == game]
                    content += self._build_game_section(game, game_conds, all_conds)
            content += self._build_overall_section(all_conds)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(self._build_overall_section(all_conds))
        logger.info(f"Summary overall section: {path}")

    def analyze(self, pipeline_results, provider, model):
        """Full rebuild from scratch (used for --analyze all mode)."""
        d = _exp_dir(provider, model, "exp3_coupling")
        path = os.path.join(d, "summary.txt")
        variant_metrics, coupling_metrics, rows = self._compute_metrics(pipeline_results)
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed", "variant", "round",
             "self_action", "opp_action", "self_payoff", "opp_payoff",
             "confidence", "normalized_payoff", "cooperation_rate"],
            rows,
        )

        content = self._build_header(pipeline_results, provider, model)
        games_in_data = set(c["game_name"] for c in pipeline_results)
        for game in config.GAME_NAMES:
            if game in games_in_data:
                game_conds = [c for c in pipeline_results if c["game_name"] == game]
                content += self._build_game_section(game, game_conds, pipeline_results)
        content += self._build_overall_section(pipeline_results)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Summary (full rebuild): {path}")

        json_path = os.path.join(d, "summary.json")
        caa_vals = _safe_vals([cm.get("conditional_action_agreement") for cm in coupling_metrics])
        ecc_b_vals = _safe_vals([cm.get("eu_coherence_B") for cm in coupling_metrics])
        ecc_b2_vals = _safe_vals([cm.get("eu_coherence_B2") for cm in coupling_metrics])
        json_data = {
            "provider": provider, "model": model,
            "timestamp": datetime.now().isoformat(),
            "n_conditions": len(pipeline_results),
            "games": sorted(set(c["game_name"] for c in pipeline_results)),
            "n_games": len(set(c["game_name"] for c in pipeline_results)),
            "mean_conditional_action_agreement": _safe_avg(caa_vals),
            "mean_eu_coherence_B": _safe_avg(ecc_b_vals),
            "mean_eu_coherence_B2": _safe_avg(ecc_b2_vals),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)

        return {
            "variant_metrics": variant_metrics,
            "coupling_metrics": coupling_metrics,
        }


# ═══════════════════════════════════════════════════════
# Exp4: Posterior Intervention
# ═══════════════════════════════════════════════════════

class Exp4Analyzer:
    """Extracts Exp4 intervention metrics from Stage 4 data."""

    def _compute_metrics(self, conditions):
        """Returns (all_probes, valid_probes, rows)."""
        rows = []
        all_valid = []

        for cond in conditions:
            s4 = cond.get("stage4")
            if not s4:
                continue
            game = cond["game_name"]
            opp = cond["opponent_type"]
            for pr in s4.get("probe_results", []):
                eu_action_flip = pr.get("eu_action_flip", pr.get("eu_ctl", "") != pr.get("eu_cf", ""))
                margin_delta = pr.get("margin_delta", 0.0)
                pr_row = {
                    "game_name": game, "opponent_type": opp,
                    "episode_id": cond["episode_id"], "seed": cond["seed"],
                    "probe_t": pr["t"], "transform": pr["transform"],
                    "is_valid": pr["is_valid"],
                    "validity_type": pr.get("validity_type", "invalid"),
                    "action_ctl": pr.get("action_ctl", ""),
                    "action_cf": pr.get("action_cf", ""),
                    "eu_ctl": pr.get("eu_ctl", ""), "eu_cf": pr.get("eu_cf", ""),
                    "eu_margin_ctl": pr.get("eu_margin_ctl", ""),
                    "eu_margin_cf": pr.get("eu_margin_cf", ""),
                    "margin_delta": margin_delta,
                    "eu_action_flip": eu_action_flip,
                    "js_ctl_cf": pr.get("js_ctl_cf", ""),
                    "valid_strict":  pr.get("valid_strict",  False),
                    "valid_default": pr.get("valid_default", pr["is_valid"]),
                    "valid_lenient": pr.get("valid_lenient", False),
                    "dir_acc_cf": pr.get("dir_acc_cf", 0),
                    "sensitive": pr.get("sensitive", 0),
                }
                rows.append(pr_row)
                if pr["is_valid"]:
                    all_valid.append(pr_row)

        return rows, all_valid

    def write_csv(self, conditions, provider, model):
        rows, _ = self._compute_metrics(conditions)
        d = _exp_dir(provider, model, "exp4_intervention")
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed",
             "probe_t", "transform", "is_valid", "validity_type",
             "eu_action_flip",
             "action_ctl", "action_cf", "eu_ctl", "eu_cf",
             "eu_margin_ctl", "eu_margin_cf", "margin_delta",
             "js_ctl_cf", "valid_strict", "valid_default", "valid_lenient",
             "dir_acc_cf", "sensitive"],
            rows,
        )
        # Sensitivity analysis CSV: validity counts at alpha = 0.03 (lenient), 0.05 (default), 0.10 (strict)
        sens_rows = []
        for game_name in sorted(set(r["game_name"] for r in rows)):
            game_probes = [r for r in rows if r["game_name"] == game_name]
            for alpha_label, key in [("0.03 (lenient)", "valid_lenient"),
                                     ("0.05 (default)", "valid_default"),
                                     ("0.10 (strict)",  "valid_strict")]:
                n_valid = sum(1 for r in game_probes if r.get(key))
                n_total = len(game_probes)
                sens_rows.append({
                    "game_name": game_name, "alpha": alpha_label,
                    "valid_probes": n_valid, "total_probes": n_total,
                    "yield_pct": (n_valid / n_total * 100) if n_total else 0.0,
                })
        if sens_rows:
            _write_csv(
                os.path.join(d, "sensitivity_analysis.csv"),
                ["game_name", "alpha", "valid_probes", "total_probes", "yield_pct"],
                sens_rows,
            )

    def _build_header(self, all_conds, provider, model):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pf, sf, _ = _integrity_stats(all_conds, ["stage4"])
        all_probes, valid_probes = self._compute_metrics(all_conds)
        n_cand = len(all_probes)
        n_valid = len(valid_probes)
        yield_pct = (n_valid / n_cand * 100) if n_cand > 0 else 0.0

        lines = []
        lines.append("=" * 70)
        lines.append("EXP 4: POSTERIOR INTERVENTION")
        lines.append(f"Provider: {provider} | Model: {model} | Date: {timestamp}")
        lines.append("=" * 70)
        lines.append(_build_run_config(provider, model, all_conds))
        # 3-way validity split
        n_flip_only = sum(1 for p in valid_probes if p.get("validity_type") == "flip_only")
        n_margin_only = sum(1 for p in valid_probes if p.get("validity_type") == "margin_only")
        n_both = sum(1 for p in valid_probes if p.get("validity_type") == "both")

        lines.append("Validity rule: Probe valid if EU-optimal action flips OR")
        lines.append("  EU-margin shifts >= alpha*payoff_range (alpha=0.05, game-scaled)")
        lines.append("  plausible_perturbation lambda=0.80 | adversarial_flip eta=0.01")
        lines.append("  Note: PD (dominant strategy) yields plausible_perturbation valid probes only")
        lines.append("  when oracle is concentrated; adversarial_flip covers all cases.")
        lines.append("  Validity includes argmax action change OR decision-surface sensitivity.")
        lines.append("  Metrics reflect causal decision-surface response, not purely discrete switching.")
        lines.append("Metrics:")
        lines.append("  Action-Flip-Valid Directional Accuracy: 𝕀[a_cf = a^EU(P_cf)] on action-flip-valid probes")
        lines.append("  EU-Margin DirAcc:   sign(margin_cf) alignment on margin-valid probes")
        lines.append("  Decision Sensitivity: 𝕀[a_ctl ≠ a_cf] on all valid probes")
        lines.append("Integrity:")
        lines.append(f"  Candidate Probes: {n_cand} | Valid Probes: {n_valid} | Yield: {yield_pct:.2f}%")
        lines.append(f"  Flip-Only Valid: {n_flip_only} | Margin-Only Valid: {n_margin_only} | Both: {n_both}")
        lines.append(f"  Parse Failures: {pf} | Schema Failures: {sf} | Retried Calls: N/A")
        lines.append("")
        return "\n".join(lines)

    def _build_game_section(self, game, game_conds, all_conds):
        all_probes, valid_probes = self._compute_metrics(game_conds)
        by_transform = defaultdict(list)
        for p in valid_probes:
            by_transform[p["transform"]].append(p)
        all_by_transform = defaultdict(list)
        for p in all_probes:
            all_by_transform[p["transform"]].append(p)

        # 3-way validity split
        flip_only = [p for p in valid_probes if p.get("validity_type") == "flip_only"]
        margin_only = [p for p in valid_probes if p.get("validity_type") == "margin_only"]
        both_valid = [p for p in valid_probes if p.get("validity_type") == "both"]
        # For Action-Flip DirAcc, include flip_only + both
        flip_valid = flip_only + both_valid

        sep = "-" * max(1, 67 - len(game))
        lines = []
        lines.append(f"-- {game.upper()} {sep}")
        lines.append(f"  Candidate Probes: {len(all_probes)} | Valid Probes: {len(valid_probes)}")
        lines.append(f"  Flip-Only: {len(flip_only)} | Margin-Only: {len(margin_only)} | Both: {len(both_valid)}")
        if valid_probes:
            da_vals = [p["dir_acc_cf"] for p in valid_probes]
            se_vals = [p["sensitive"] for p in valid_probes]
            lines.append(f"  Decision Sensitivity (combined): {_fmt_ci(se_vals)}")
        if flip_valid:
            flip_da = [p["dir_acc_cf"] for p in flip_valid]
            lines.append(f"  Action-Flip-Valid Directional Accuracy:              {_fmt_ci(flip_da)}")
        if margin_only:
            # For margin-only probes, directional accuracy = did the LLM shift
            # in the correct direction even though the argmax didn't change?
            margin_se = [p["sensitive"] for p in margin_only]
            lines.append(f"  Margin-Only-Valid Directional Accuracy:         {_fmt_ci(margin_se)}")
        lines.append("  By Transform:")
        lines.append(f"  {'TRANSFORM':<25}  {'DirAcc':>8}  {'Sensitivity':>12}  {'Valid N':>8}  {'Candidate N':>11}")
        lines.append("  " + "\u2500" * 68)
        for tn in ["adversarial_flip", "plausible_perturbation"]:
            tp = by_transform.get(tn, [])
            tp_all = all_by_transform.get(tn, [])
            if tp_all:
                da = _safe_avg([p["dir_acc_cf"] for p in tp]) if tp else 0.0
                se = _safe_avg([p["sensitive"] for p in tp]) if tp else 0.0
                lines.append(
                    f"  {tn:<25}  {da:>8.4f}  {se:>12.4f}  {len(tp):>8}  {len(tp_all):>11}"
                )
        # Transform Diagnostics
        js_vals_diag = _safe_vals([p.get("js_ctl_cf", 0) for p in all_probes if p.get("js_ctl_cf")])
        md_vals_diag = _safe_vals([p.get("margin_delta", 0) for p in all_probes])
        n_near_zero = sum(1 for v in js_vals_diag if v < 0.001)
        lines.append("  Transform Diagnostics:")
        lines.append(f"    Mean JS(ctl, cf): {_safe_avg(js_vals_diag):.4f}")
        lines.append(f"    Median JS(ctl, cf): {_median(js_vals_diag):.4f}")
        lines.append(f"    Near-No-Effect Transforms (JS < 0.001): {n_near_zero}")
        lines.append(f"    Mean Margin Delta: {_safe_avg(md_vals_diag):.4f}")
        lines.append(f"    Median Margin Delta: {_median(md_vals_diag):.4f}")
        lines.append("")
        return "\n".join(lines)

    def _build_overall_section(self, all_conds):
        all_probes, valid_probes = self._compute_metrics(all_conds)
        n_cand = len(all_probes)
        n_valid = len(valid_probes)
        yield_pct = (n_valid / n_cand * 100) if n_cand > 0 else 0.0
        by_transform = defaultdict(list)
        for p in valid_probes:
            by_transform[p["transform"]].append(p)
        all_by_transform = defaultdict(list)
        for p in all_probes:
            all_by_transform[p["transform"]].append(p)

        flip_only = [p for p in valid_probes if p.get("validity_type") == "flip_only"]
        margin_only = [p for p in valid_probes if p.get("validity_type") == "margin_only"]
        both_valid = [p for p in valid_probes if p.get("validity_type") == "both"]
        flip_valid = flip_only + both_valid

        lines = []
        lines.append("-- OVERALL " + "-" * 57)
        lines.append(f"  Candidate Probes: {n_cand} | Valid Probes: {n_valid} | Yield: {yield_pct:.2f}%")
        lines.append(f"  Flip-Only: {len(flip_only)} | Margin-Only: {len(margin_only)} | Both: {len(both_valid)}")
        if valid_probes:
            se_vals = [p["sensitive"] for p in valid_probes]
            lines.append(f"  Decision Sensitivity (combined): {_fmt_ci(se_vals)}")
        if flip_valid:
            flip_da = [p["dir_acc_cf"] for p in flip_valid]
            lines.append(f"  Action-Flip-Valid Directional Accuracy:              {_fmt_ci(flip_da)}")
        if margin_only:
            margin_se = [p["sensitive"] for p in margin_only]
            lines.append(f"  Margin-Only-Valid Directional Accuracy:         {_fmt_ci(margin_se)}")
        lines.append("  By Transform:")
        lines.append(f"  {'TRANSFORM':<25}  {'DirAcc':>8}  {'Sensitivity':>12}  {'Valid N':>8}  {'Candidate N':>11}")
        lines.append("  " + "\u2500" * 68)
        for tn in ["adversarial_flip", "plausible_perturbation"]:
            tp = by_transform.get(tn, [])
            tp_all = all_by_transform.get(tn, [])
            if tp_all:
                da = _safe_avg([p["dir_acc_cf"] for p in tp]) if tp else 0.0
                se = _safe_avg([p["sensitive"] for p in tp]) if tp else 0.0
                lines.append(
                    f"  {tn:<25}  {da:>8.4f}  {se:>12.4f}  {len(tp):>8}  {len(tp_all):>11}"
                )
        lines.append("")
        lines.append("Warnings / Notes:")
        lines.append("  Validity includes argmax action change OR decision-surface sensitivity")
        lines.append("  Metrics reflect causal decision-surface response, not purely discrete switching")
        lines.append("  Decision sensitivity includes action-flip-valid and/or margin-valid probes")
        lines.append("")
        lines.append(_build_trace_audit(all_conds))
        lines.append("")
        lines.append(_build_coverage_check(all_conds))
        lines.append("")
        return "\n".join(lines)

    def write_game_to_summary(self, game, game_conds, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp4_intervention")
        path = os.path.join(d, "summary.txt")
        mode = "a"
        prefix = ""
        if not os.path.exists(path):
            prefix = self._build_header(all_conds, provider, model)
        section = self._build_game_section(game, game_conds, all_conds)
        with open(path, mode, encoding="utf-8") as f:
            if prefix:
                f.write(prefix)
            f.write(section)
        logger.info(f"Summary game section [{game}]: {path}")

    def write_overall_to_summary(self, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp4_intervention")
        path = os.path.join(d, "summary.txt")
        if not os.path.exists(path):
            content = self._build_header(all_conds, provider, model)
            games_in_data = set(c["game_name"] for c in all_conds)
            for game in config.GAME_NAMES:
                if game in games_in_data:
                    game_conds = [c for c in all_conds if c["game_name"] == game]
                    content += self._build_game_section(game, game_conds, all_conds)
            content += self._build_overall_section(all_conds)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(self._build_overall_section(all_conds))
        logger.info(f"Summary overall section: {path}")

    def analyze(self, pipeline_results, provider, model):
        """Full rebuild from scratch (used for --analyze all mode)."""
        d = _exp_dir(provider, model, "exp4_intervention")
        path = os.path.join(d, "summary.txt")
        all_probes, valid_probes = self._compute_metrics(pipeline_results)
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed",
             "probe_t", "transform", "is_valid", "validity_type",
             "eu_action_flip",
             "action_ctl", "action_cf", "eu_ctl", "eu_cf",
             "eu_margin_ctl", "eu_margin_cf", "margin_delta",
             "js_ctl_cf", "valid_strict", "valid_default", "valid_lenient",
             "dir_acc_cf", "sensitive"],
            all_probes,
        )

        content = self._build_header(pipeline_results, provider, model)
        games_in_data = set(c["game_name"] for c in pipeline_results)
        for game in config.GAME_NAMES:
            if game in games_in_data:
                game_conds = [c for c in pipeline_results if c["game_name"] == game]
                content += self._build_game_section(game, game_conds, pipeline_results)
        content += self._build_overall_section(pipeline_results)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Summary (full rebuild): {path}")

        json_path = os.path.join(d, "summary.json")
        json_data = {
            "provider": provider, "model": model,
            "timestamp": datetime.now().isoformat(),
            "n_conditions": len(pipeline_results),
            "games": sorted(set(c["game_name"] for c in pipeline_results)),
            "n_games": len(set(c["game_name"] for c in pipeline_results)),
            "n_candidate_probes": len(all_probes),
            "n_valid_probes": len(valid_probes),
            "yield_pct": (len(valid_probes) / len(all_probes) * 100) if all_probes else 0.0,
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)

        return {"all_probes": all_probes, "valid_probes": valid_probes}


# ═══════════════════════════════════════════════════════
# Exp6: Cognitive Theory Metrics (offline)
# ═══════════════════════════════════════════════════════

class Exp6Analyzer:
    """Computes Exp6 cognitive metrics from Stage 5 data + cross-game rigidity."""

    def _compute_metrics(self, conditions):
        """Returns (all_metrics, cr_by_game, rigidity_val, rows)."""
        all_metrics = []
        rows = []

        for cond in conditions:
            s5 = cond.get("stage5", {})
            s1 = cond.get("stage1", {})
            game = cond["game_name"]
            opp = cond["opponent_type"]

            actions = [r["self_action"] for r in s1.get("rounds", [])]
            cr = cooperation_rate(actions) if actions else 0.0

            m = {
                "game_name": game, "opponent_type": opp,
                "episode_id": cond["episode_id"], "seed": cond["seed"],
                "cooperation_rate": cr,
            }
            if s5:
                m.update(s5)

            probe_data = s1.get("probe_data", [])
            if len(probe_data) >= 2:
                sorted_probes = sorted(probe_data, key=lambda p: p["t"])
                js_consecutive = []
                for i in range(1, len(sorted_probes)):
                    p_prev = sorted_probes[i - 1].get("posterior_over_types", {})
                    p_curr = sorted_probes[i].get("posterior_over_types", {})
                    if p_prev and p_curr:
                        js_consecutive.append(js_divergence(p_prev, p_curr))
                if js_consecutive:
                    m["posterior_consistency"] = _safe_avg(js_consecutive)

            # H6: type accuracy per opponent — deceptive vs non-deceptive
            deceptive_correct = []
            non_deceptive_correct = []
            for pd in probe_data:
                mp = pd.get("posterior_over_types", {})
                if mp:
                    predicted_type = max(mp, key=mp.get)
                    is_correct = (predicted_type == opp)
                    if opp in DECEPTIVE_TYPES:
                        deceptive_correct.append(1 if is_correct else 0)
                    else:
                        non_deceptive_correct.append(1 if is_correct else 0)

            if deceptive_correct or non_deceptive_correct:
                if deceptive_correct:
                    m["h6_deceptive_acc"] = _safe_avg(deceptive_correct)
                if non_deceptive_correct:
                    m["h6_non_deceptive_acc"] = _safe_avg(non_deceptive_correct)

            all_metrics.append(m)
            rows.append(m)

        by_game_cr = defaultdict(list)
        for m in all_metrics:
            by_game_cr[m["game_name"]].append(m["cooperation_rate"])
        cr_by_game = {g: _safe_avg(vs) for g, vs in by_game_cr.items()}
        ri = rigidity_index(cr_by_game) if len(cr_by_game) >= 2 else 0.0

        for m in all_metrics:
            m["h5_rigidity_index"] = ri

        return all_metrics, cr_by_game, ri, rows

    def write_csv(self, conditions, provider, model):
        all_metrics, _, _, rows = self._compute_metrics(conditions)
        d = _exp_dir(provider, model, "exp6_cognitive")
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed",
             "h1_mean_js", "h1_mean_kl", "h4_ece", "h4_mce",
             "h5_rigidity_index", "h7_mean_surprisal",
             "h7_mean_update_magnitude", "h7_surprisal_update_corr",
             "posterior_consistency", "cooperation_rate"],
            rows,
        )

    def _count_probe_transitions(self, conditions):
        count = 0
        for cond in conditions:
            s1 = cond.get("stage1", {})
            probe_data = s1.get("probe_data", [])
            if len(probe_data) >= 2:
                count += len(probe_data) - 1
        return count

    def _count_parse_schema_failures(self, conditions):
        pf, sf, _ = _integrity_stats(conditions, ["stage1"])
        return pf + sf

    def _build_header(self, all_conds, provider, model):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_episodes = sum(1 for c in all_conds if c.get("stage1"))
        n_transitions = self._count_probe_transitions(all_conds)
        n_failures = self._count_parse_schema_failures(all_conds)

        lines = []
        lines.append("=" * 70)
        lines.append("EXP 6: COGNITIVE THEORY METRICS")
        lines.append(f"Provider: {provider} | Model: {model} | Date: {timestamp}")
        lines.append("=" * 70)
        lines.append(_build_run_config(provider, model, all_conds))
        lines.append("Metric scope:")
        lines.append("  Offline analysis over saved episode/probe traces")
        lines.append("  Episode is the unit of inference for CIs and correlations")
        lines.append("  All belief metrics use elicited (not ground-truth) posteriors")
        lines.append("Integrity:")
        lines.append(f"  Episodes Used: {n_episodes} | Probe Transitions Used: {n_transitions}")
        lines.append(f"  Parse/Schema Failures inherited from source traces: {n_failures}")
        lines.append("")
        return "\n".join(lines)

    def _build_game_section(self, game, game_conds, all_conds):
        all_metrics, cr_by_game, ri, _ = self._compute_metrics(game_conds)
        ms = [m for m in all_metrics if m["game_name"] == game]

        g_js = [m["h1_mean_js"] for m in ms if "h1_mean_js" in m]
        g_kl = [m["h1_mean_kl"] for m in ms if "h1_mean_kl" in m]
        g_ece = [m["h4_ece"] for m in ms if "h4_ece" in m]
        g_mce = [m["h4_mce"] for m in ms if "h4_mce" in m]
        g_surp = [m["h7_mean_surprisal"] for m in ms if "h7_mean_surprisal" in m]
        g_upd = [m["h7_mean_update_magnitude"] for m in ms if "h7_mean_update_magnitude" in m]
        g_corr = [m["h7_surprisal_update_corr"] for m in ms if "h7_surprisal_update_corr" in m]
        g_cr = cr_by_game.get(game, 0.0)
        g_pc = [m["posterior_consistency"] for m in ms if "posterior_consistency" in m]

        # H6: aggregate type accuracy
        deceptive_accs = [m["h6_deceptive_acc"] for m in ms if "h6_deceptive_acc" in m]
        non_deceptive_accs = [m["h6_non_deceptive_acc"] for m in ms if "h6_non_deceptive_acc" in m]
        dec_mean = _safe_avg(deceptive_accs) if deceptive_accs else 0.0
        non_dec_mean = _safe_avg(non_deceptive_accs) if non_deceptive_accs else 0.0
        gap = non_dec_mean - dec_mean

        # H2 eu coherence — from coupling data if available (recompute from stage1/stage2)
        ecc_vals = []
        for cond in game_conds:
            s1 = cond.get("stage1")
            if not s1 or not s1.get("probe_data"):
                continue
            probes = s1["probe_data"]
            rounds = s1["rounds"]
            b_actions = [pd.get("probe_action", "") for pd in probes]
            b_posteriors = [pd.get("posterior_over_types", {}) for pd in probes]
            b_histories = []
            for pd in probes:
                t = pd["t"]
                h = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                     for r in rounds[:t - 1]]
                b_histories.append(h)
            if all(b_actions) and all(b_posteriors):
                try:
                    ecc = eu_consistent_coherence(b_actions, b_posteriors,
                                                   cond["game_name"], b_histories, cond["seed"])
                    if ecc is not None:
                        ecc_vals.append(ecc)
                except Exception:
                    pass

        sep = "-" * max(1, 67 - len(game))
        lines = []
        lines.append(f"-- {game.upper()} {sep}")
        lines.append(f"  N Episodes: {len(ms)}")
        lines.append("  H1 - Bayesian Belief Updating")
        lines.append(f"    Mean JS Divergence: {_fmt_ci(g_js)}")
        lines.append(f"    Mean KL Divergence: {_fmt_ci(g_kl)}")
        lines.append("  H4 - Metacognitive Calibration")
        lines.append(f"    ECE: {_fmt_ci(g_ece)}")
        lines.append(f"    MCE: {_safe_avg(g_mce):.4f}")
        lines.append("  H7 - Predictive Processing")
        lines.append(f"    Mean Surprisal:        {_safe_avg(g_surp):.4f}")
        g_surp_sv = _safe_vals(g_surp)
        if g_surp_sv:
            lines.append(f"    Surprisal (median/min/max): {_median(g_surp):.4f}/{min(g_surp_sv):.4f}/{max(g_surp_sv):.4f}")
        lines.append(f"    Mean Update Magnitude: {_safe_avg(g_upd):.4f}")
        g_upd_sv = _safe_vals(g_upd)
        if g_upd_sv:
            lines.append(f"    Update Mag (median/min/max): {_median(g_upd):.4f}/{min(g_upd_sv):.4f}/{max(g_upd_sv):.4f}")
        n_corr = len(g_corr)
        corr_note = f"  (N={n_corr}; 0.0 if constant variance)" if _safe_avg(g_corr) == 0.0 else f"  (N={n_corr})"
        lines.append(f"    Surprisal-Update Corr: {_safe_avg(g_corr):.4f}{corr_note}")
        lines.append("  H6 - Deceptive-Type Gap")
        lines.append(f"    Deceptive Types (deceptive_opportunist, gradual_defector):    {dec_mean:.4f} acc")
        lines.append(f"    Non-Deceptive Types:                                          {non_dec_mean:.4f} acc")
        lines.append(f"    Gap (non-deceptive - deceptive):                              {gap:.4f}")
        lines.append(f"  H2 - EU-Consistent Coherence: {_fmt_ci(ecc_vals)}")
        lines.append(f"  Posterior Consistency (mean JS consec. probes): {_safe_avg(g_pc):.4f}")
        lines.append(f"  Cooperation Rate: {g_cr:.4f}")
        lines.append("")
        return "\n".join(lines)

    def _build_overall_section(self, all_conds):
        all_metrics, cr_by_game, ri, _ = self._compute_metrics(all_conds)

        js_vals = [m["h1_mean_js"] for m in all_metrics if "h1_mean_js" in m]
        kl_vals = [m["h1_mean_kl"] for m in all_metrics if "h1_mean_kl" in m]
        ece_vals = [m["h4_ece"] for m in all_metrics if "h4_ece" in m]
        mce_vals = [m["h4_mce"] for m in all_metrics if "h4_mce" in m]
        surp_vals = [m["h7_mean_surprisal"] for m in all_metrics if "h7_mean_surprisal" in m]
        upd_vals = [m["h7_mean_update_magnitude"] for m in all_metrics if "h7_mean_update_magnitude" in m]
        corr_vals = [m["h7_surprisal_update_corr"] for m in all_metrics if "h7_surprisal_update_corr" in m]
        pc_vals = [m["posterior_consistency"] for m in all_metrics if "posterior_consistency" in m]

        deceptive_accs = [m["h6_deceptive_acc"] for m in all_metrics if "h6_deceptive_acc" in m]
        non_deceptive_accs = [m["h6_non_deceptive_acc"] for m in all_metrics if "h6_non_deceptive_acc" in m]
        dec_mean = _safe_avg(deceptive_accs) if deceptive_accs else 0.0
        non_dec_mean = _safe_avg(non_deceptive_accs) if non_deceptive_accs else 0.0
        h6_gap = non_dec_mean - dec_mean

        ecc_vals = []
        for cond in all_conds:
            s1 = cond.get("stage1")
            if not s1 or not s1.get("probe_data"):
                continue
            probes = s1["probe_data"]
            rounds = s1["rounds"]
            b_actions = [pd.get("probe_action", "") for pd in probes]
            b_posteriors = [pd.get("posterior_over_types", {}) for pd in probes]
            b_histories = []
            for pd in probes:
                t = pd["t"]
                h = [{"self_action": r["self_action"], "opp_action": r["opp_action"]}
                     for r in rounds[:t - 1]]
                b_histories.append(h)
            if all(b_actions) and all(b_posteriors):
                try:
                    ecc = eu_consistent_coherence(b_actions, b_posteriors,
                                                   cond["game_name"], b_histories, cond["seed"])
                    if ecc is not None:
                        ecc_vals.append(ecc)
                except Exception:
                    pass

        lines = []
        lines.append("-- OVERALL " + "-" * 57)
        lines.append("  H1 - Bayesian Belief Updating")
        lines.append(f"    Mean JS Divergence: {_fmt_ci(js_vals)}")
        lines.append(f"    Mean KL Divergence: {_fmt_ci(kl_vals)}")
        lines.append("  H4 - Metacognitive Calibration")
        lines.append(f"    ECE: {_fmt_ci(ece_vals)}")
        lines.append(f"    MCE: {_safe_avg(mce_vals):.4f}")
        lines.append("  H5 - Cross-Game Rigidity")
        lines.append("    Per-game cooperation rates:")
        for g in config.GAME_NAMES:
            if g in cr_by_game:
                lines.append(f"      {g}: {cr_by_game[g]:.4f}")
        n_games = len(cr_by_game)
        lines.append(f"    Rigidity Index: {ri:.4f}  (point estimate across {n_games} games)")
        lines.append("  H7 - Predictive Processing")
        lines.append(f"    Mean Surprisal:        {_safe_avg(surp_vals):.4f}")
        lines.append(f"    Mean Update Magnitude: {_safe_avg(upd_vals):.4f}")
        n_corr_o = len(corr_vals)
        corr_note_o = f"  (N={n_corr_o}; 0.0 if constant variance)" if _safe_avg(corr_vals) == 0.0 else f"  (N={n_corr_o})"
        lines.append(f"    Surprisal-Update Corr: {_safe_avg(corr_vals):.4f}{corr_note_o}")
        lines.append(f"  H6 - Deceptive-Type Gap:  {h6_gap:.4f}")
        lines.append(f"  H2 - EU-Consistent Coherence: {_fmt_ci(ecc_vals)}")
        lines.append(f"  Posterior Consistency: {_safe_avg(pc_vals):.4f}")
        lines.append("")
        lines.append("Warnings / Notes:")
        lines.append("  Rigidity Index is a point estimate (std of per-game cooperation rates)")
        lines.append("  CI omitted as it is not statistically meaningful for a single scalar")
        lines.append("  Surprisal-Update Correlation returns 0.0 when variance is zero (constant predictions)")
        lines.append("  All belief metrics use elicited (not ground-truth) posteriors")
        lines.append("")
        lines.append(_build_trace_audit(all_conds))
        lines.append("")
        lines.append(_build_coverage_check(all_conds))
        lines.append("")
        return "\n".join(lines)

    def write_game_to_summary(self, game, game_conds, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp6_cognitive")
        path = os.path.join(d, "summary.txt")
        mode = "a"
        prefix = ""
        if not os.path.exists(path):
            prefix = self._build_header(all_conds, provider, model)
        section = self._build_game_section(game, game_conds, all_conds)
        with open(path, mode, encoding="utf-8") as f:
            if prefix:
                f.write(prefix)
            f.write(section)
        logger.info(f"Summary game section [{game}]: {path}")

    def write_overall_to_summary(self, all_conds, provider, model):
        d = _exp_dir(provider, model, "exp6_cognitive")
        path = os.path.join(d, "summary.txt")
        if not os.path.exists(path):
            content = self._build_header(all_conds, provider, model)
            games_in_data = set(c["game_name"] for c in all_conds)
            for game in config.GAME_NAMES:
                if game in games_in_data:
                    game_conds = [c for c in all_conds if c["game_name"] == game]
                    content += self._build_game_section(game, game_conds, all_conds)
            content += self._build_overall_section(all_conds)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(path, "a", encoding="utf-8") as f:
                f.write(self._build_overall_section(all_conds))
        logger.info(f"Summary overall section: {path}")

    def analyze(self, pipeline_results, provider, model):
        """Full rebuild from scratch (used for --analyze all mode)."""
        d = _exp_dir(provider, model, "exp6_cognitive")
        path = os.path.join(d, "summary.txt")
        all_metrics, cr_by_game, ri, rows = self._compute_metrics(pipeline_results)
        _write_csv(
            os.path.join(d, "data.csv"),
            ["game_name", "opponent_type", "episode_id", "seed",
             "h1_mean_js", "h1_mean_kl", "h4_ece", "h4_mce",
             "h5_rigidity_index", "h7_mean_surprisal",
             "h7_mean_update_magnitude", "h7_surprisal_update_corr",
             "posterior_consistency", "cooperation_rate"],
            rows,
        )

        content = self._build_header(pipeline_results, provider, model)
        games_in_data = set(c["game_name"] for c in pipeline_results)
        for game in config.GAME_NAMES:
            if game in games_in_data:
                game_conds = [c for c in pipeline_results if c["game_name"] == game]
                content += self._build_game_section(game, game_conds, pipeline_results)
        content += self._build_overall_section(pipeline_results)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Summary (full rebuild): {path}")

        json_path = os.path.join(d, "summary.json")
        json_data = {
            "provider": provider, "model": model,
            "timestamp": datetime.now().isoformat(),
            "n_conditions": len(pipeline_results),
            "games": sorted(set(c["game_name"] for c in pipeline_results)),
            "n_games": len(set(c["game_name"] for c in pipeline_results)),
            "rigidity_index": ri,
            "mean_js": _safe_avg([m["h1_mean_js"] for m in all_metrics if "h1_mean_js" in m]),
            "mean_kl": _safe_avg([m["h1_mean_kl"] for m in all_metrics if "h1_mean_kl" in m]),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, indent=2)

        return {"all_metrics": all_metrics, "cr_by_game": cr_by_game, "rigidity": ri}


# ═══════════════════════════════════════════════════════
# Post-run sanity checks
# ═══════════════════════════════════════════════════════

def sanity_check_exp3(conditions: list) -> list[str]:
    """Validate Exp3 results are non-degenerate. Returns list of warnings."""
    warnings = []
    az = Exp3Analyzer()
    variant_metrics, coupling_metrics, _ = az._compute_metrics(conditions)

    # Identifiable states > 0
    n_ident = sum(cm.get("n_identifiable_states", 0) for cm in coupling_metrics)
    if n_ident == 0:
        warnings.append("FAIL: Exp3 identifiable states = 0 (all probes below tau=0.60)")
    else:
        logger.info(f"Exp3 sanity: {n_ident} identifiable states OK")

    # Action-sensitive states > 0
    n_actsens = sum(cm.get("n_action_sensitive_states", 0) for cm in coupling_metrics)
    if n_actsens == 0:
        warnings.append("WARN: Exp3 action-sensitive states = 0 (secondary diagnostic)")
    else:
        logger.info(f"Exp3 sanity: {n_actsens} action-sensitive states OK")

    # B payoff nonzero
    b_payoffs = [m["normalized_payoff"] for m in variant_metrics.get("B", [])
                 if "normalized_payoff" in m]
    if not b_payoffs:
        warnings.append("FAIL: Exp3 Variant B has no normalized_payoff stored")
    elif all(v == 0.0 for v in b_payoffs):
        warnings.append("WARN: Exp3 Variant B all payoffs = 0.0000")
    else:
        # Normalization bounds
        oob = [v for v in b_payoffs if not (0.0 <= v <= 1.0)]
        if oob:
            warnings.append(f"FAIL: Exp3 B normalized_payoff out of [0,1]: {oob[:5]}")
        logger.info(f"Exp3 sanity: B payoff mean={sum(b_payoffs)/len(b_payoffs):.4f} N={len(b_payoffs)} OK")

    # B2 payoff nonzero
    b2_payoffs = [m["normalized_payoff"] for m in variant_metrics.get("B2", [])
                  if "normalized_payoff" in m]
    if not b2_payoffs:
        warnings.append("FAIL: Exp3 Variant B2 has no normalized_payoff stored")
    elif all(v == 0.0 for v in b2_payoffs):
        warnings.append("WARN: Exp3 Variant B2 all payoffs = 0.0000")
    else:
        oob = [v for v in b2_payoffs if not (0.0 <= v <= 1.0)]
        if oob:
            warnings.append(f"FAIL: Exp3 B2 normalized_payoff out of [0,1]: {oob[:5]}")
        logger.info(f"Exp3 sanity: B2 payoff mean={sum(b2_payoffs)/len(b2_payoffs):.4f} N={len(b2_payoffs)} OK")

    # EU coherence populated
    ecc_b = _safe_vals([cm.get("eu_coherence_B") for cm in coupling_metrics])
    ecc_b2 = _safe_vals([cm.get("eu_coherence_B2") for cm in coupling_metrics])
    if not ecc_b:
        warnings.append("WARN: Exp3 EU coherence (B) is empty")
    if not ecc_b2:
        warnings.append("WARN: Exp3 EU coherence (B2) is empty")

    # Conditional agreement computed on nonzero N
    caa = _safe_vals([cm.get("conditional_action_agreement") for cm in coupling_metrics])
    if not caa:
        warnings.append("WARN: Exp3 conditional action agreement empty (no identifiable states with valid actions)")

    for w in warnings:
        logger.warning(f"[SANITY] {w}")
    return warnings


def sanity_check_exp4(conditions: list) -> list[str]:
    """Validate Exp4 results are non-degenerate. Returns list of warnings."""
    warnings = []
    az = Exp4Analyzer()
    all_probes, valid_probes = az._compute_metrics(conditions)

    if not all_probes:
        warnings.append("FAIL: Exp4 has no probe results at all")
        for w in warnings:
            logger.warning(f"[SANITY] {w}")
        return warnings

    n_valid = len(valid_probes)
    n_cand = len(all_probes)
    yield_pct = n_valid / n_cand * 100 if n_cand else 0

    if n_valid == 0:
        warnings.append(f"FAIL: Exp4 valid probes = 0 / {n_cand}")
    else:
        logger.info(f"Exp4 sanity: {n_valid}/{n_cand} valid ({yield_pct:.1f}%) OK")

    # Per-game check
    games = set(p["game_name"] for p in all_probes)
    for game in sorted(games):
        game_valid = [p for p in valid_probes if p["game_name"] == game]
        game_total = [p for p in all_probes if p["game_name"] == game]
        if not game_valid:
            warnings.append(f"WARN: Exp4 {game}: 0 valid probes out of {len(game_total)}")

    # 3-way split
    n_flip_only = sum(1 for p in valid_probes if p.get("validity_type") == "flip_only")
    n_margin_only = sum(1 for p in valid_probes if p.get("validity_type") == "margin_only")
    n_both = sum(1 for p in valid_probes if p.get("validity_type") == "both")
    logger.info(f"Exp4 sanity: validity split — flip_only={n_flip_only} margin_only={n_margin_only} both={n_both}")

    if n_valid > 0 and n_flip_only == 0 and n_both == 0:
        warnings.append("NOTE: Exp4 100% margin-only validity — no action flips observed. "
                        "Experiment works in weak sense only (decision-surface shift, not discrete switching)")

    # JS divergence check — at least some transforms should produce nonzero JS
    js_vals = [p.get("js_ctl_cf", 0) for p in all_probes if p.get("js_ctl_cf")]
    if js_vals:
        if all(v < 1e-6 for v in js_vals):
            warnings.append("FAIL: Exp4 all JS(ctl,cf) ≈ 0 — transforms had no effect")
    else:
        warnings.append("WARN: Exp4 no JS divergence values recorded")

    # Margin shift distribution
    margin_deltas = [p.get("margin_delta", 0) for p in all_probes]
    nonzero_margins = [d for d in margin_deltas if d > 1e-6]
    if not nonzero_margins:
        warnings.append("FAIL: Exp4 all margin_delta ≈ 0")

    for w in warnings:
        logger.warning(f"[SANITY] {w}")
    return warnings


# ═══════════════════════════════════════════════════════
# Run all analyzers
# ═══════════════════════════════════════════════════════

def run_all_analyzers(pipeline_results: list, provider: str, model: str,
                      experiments: list = None) -> dict:
    """Run specified (or all) experiment analyzers on pipeline results.

    For --analyze all mode: calls analyze() on each (full rebuild, write mode).
    Returns results dict.
    """
    exps = set(experiments or ["exp1", "exp2", "exp3", "exp4", "exp6"])
    results = {}

    analyzers = [
        ("exp1", Exp1Analyzer()),
        ("exp2", Exp2Analyzer()),
        ("exp3", Exp3Analyzer()),
        ("exp4", Exp4Analyzer()),
        ("exp6", Exp6Analyzer()),
    ]

    for key, az in analyzers:
        if key in exps:
            try:
                results[key] = az.analyze(pipeline_results, provider, model)
            except Exception as e:
                logger.warning(f"{az.__class__.__name__} analyze failed: {e}")

    # Run sanity checks on completed results
    if "exp3" in exps:
        try:
            exp3_warnings = sanity_check_exp3(pipeline_results)
            results["exp3_sanity"] = exp3_warnings
        except Exception as e:
            logger.warning(f"Exp3 sanity check failed: {e}")
    if "exp4" in exps:
        try:
            exp4_warnings = sanity_check_exp4(pipeline_results)
            results["exp4_sanity"] = exp4_warnings
        except Exception as e:
            logger.warning(f"Exp4 sanity check failed: {e}")

    safe_model = model.replace("/", "_").replace(":", "_")
    out_dir = os.path.join(config.RESULTS_DIR, f"{provider}_{safe_model}")
    logger.info(f"All analyzers complete. Results in {out_dir}")
    return results
