"""
Plotting module for adversarial game experiment results.

Generates per-experiment plots and cross-model comparison plots.
All plots use matplotlib with Agg backend (non-interactive) and save at 300 DPI.
"""
import os
import math
import logging
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Styling constants ──
DPI = 300
FIG_SIZE = (8, 5)
FIG_SIZE_WIDE = (10, 6)
COLORS_9 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]
COLORS_4 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
FONT_SIZE = 11
TITLE_SIZE = 13

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "figure.dpi": 100,
})


def _plot_dir(provider: str, model: str) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    d = os.path.join(config.RESULTS_DIR, f"{provider}_{safe_model}", "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved: {path}")


def _safe_avg(vals):
    v = [x for x in vals if x is not None and x == x]
    return sum(v) / len(v) if v else 0.0


def _safe_se(vals):
    v = [x for x in vals if x is not None and x == x]
    if len(v) < 2:
        return 0.0
    return float(np.std(v, ddof=1)) / math.sqrt(len(v))


# ═══════════════════════════════════════════════════════
# Exp1: Baseline Strategic Play
# ═══════════════════════════════════════════════════════

def plot_exp1(analyzer_result: dict, provider: str, model: str):
    """Exp1 plots: payoff by game/opponent, cooperation heatmap."""
    metrics = analyzer_result.get("episode_metrics", [])
    if not metrics:
        return
    d = _plot_dir(provider, model)

    by_go = defaultdict(list)
    for m in metrics:
        by_go[(m["game_name"], m["opponent_type"])].append(m)

    # ── Plot 1: Normalized payoff by game (faceted, one subplot per game) ──
    games = config.GAME_NAMES
    opps = config.OPPONENT_TYPES
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(opps))
    width = 0.7

    for idx, game in enumerate(games):
        ax = axes[idx]
        means = [_safe_avg([m["normalized_payoff"] for m in by_go.get((game, o), [])]) for o in opps]
        errs = [_safe_se([m["normalized_payoff"] for m in by_go.get((game, o), [])]) for o in opps]
        bars = ax.bar(x, means, width, yerr=errs, capsize=3,
                      color=COLORS_9[:len(opps)], edgecolor="white", linewidth=0.5)
        ax.set_title(game.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("_", "\n") for o in opps], fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Normalized Payoff" if idx % 2 == 0 else "")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    fig.suptitle("Exp1: Normalized Payoff by Game and Opponent", fontsize=TITLE_SIZE, y=1.02)
    _save(fig, os.path.join(d, "exp1_payoff_by_game.png"))

    # ── Plot 2: Cooperation rate heatmap ──
    mat = np.zeros((len(games), len(opps)))
    for i, game in enumerate(games):
        for j, opp in enumerate(opps):
            mat[i, j] = _safe_avg([m["cooperation_rate"] for m in by_go.get((game, opp), [])])

    fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(opps)))
    ax.set_xticklabels([o.replace("_", "\n") for o in opps], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(games)))
    ax.set_yticklabels([g.replace("_", " ").title() for g in games])
    for i in range(len(games)):
        for j in range(len(opps)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < mat[i, j] < 0.7 else "white")
    fig.colorbar(im, ax=ax, label="Cooperation Rate")
    ax.set_title("Exp1: Cooperation Rate Heatmap")
    _save(fig, os.path.join(d, "exp1_cooperation_heatmap.png"))


# ═══════════════════════════════════════════════════════
# Exp2: Belief Elicitation
# ═══════════════════════════════════════════════════════

def plot_exp2(analyzer_result: dict, provider: str, model: str):
    """Exp2 plots: KL by probe round, calibration diagram, Brier by game."""
    episode_metrics = analyzer_result.get("episode_metrics", [])
    probe_details = analyzer_result.get("probe_details", [])
    cal_data = analyzer_result.get("calibration_data", [])
    if not episode_metrics:
        return
    d = _plot_dir(provider, model)

    # ── Plot 1: KL by probe round (one line per game) ──
    if probe_details:
        pd_by_gr = defaultdict(list)
        for pd in probe_details:
            pd_by_gr[(pd["game_name"], pd["probe_round"])].append(pd["kl"])

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for idx, game in enumerate(config.GAME_NAMES):
            rounds_x = []
            means = []
            errs = []
            for pr in config.PROBE_ROUNDS:
                vals = pd_by_gr.get((game, pr), [])
                if vals:
                    rounds_x.append(pr)
                    means.append(_safe_avg(vals))
                    errs.append(_safe_se(vals))
            if rounds_x:
                ax.errorbar(rounds_x, means, yerr=errs, marker="o", capsize=4,
                            label=game.replace("_", " ").title(),
                            color=COLORS_4[idx], linewidth=2)
        ax.set_xlabel("Probe Round")
        ax.set_ylabel("KL Divergence")
        ax.set_title("Exp2: KL Divergence by Probe Round")
        ax.legend()
        ax.set_xticks(config.PROBE_ROUNDS)
        _save(fig, os.path.join(d, "exp2_kl_by_probe_round.png"))

    # ── Plot 2: Calibration diagram ──
    if cal_data:
        n_bins = 10
        bins = [[] for _ in range(n_bins)]
        for conf, correct in cal_data:
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, correct))

        bin_confs, bin_accs, bin_sizes = [], [], []
        for b in bins:
            if b:
                bin_confs.append(sum(c for c, _ in b) / len(b))
                bin_accs.append(sum(1 for _, c in b if c) / len(b))
                bin_sizes.append(len(b))

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        if bin_confs:
            sizes = np.array(bin_sizes)
            sizes_norm = sizes / max(sizes) * 300 + 30
            ax.scatter(bin_confs, bin_accs, s=sizes_norm, alpha=0.7, color=COLORS_4[0],
                       edgecolors="white", linewidth=0.5)
            ax.plot(bin_confs, bin_accs, "-", color=COLORS_4[0], alpha=0.5)
            for x, y, n in zip(bin_confs, bin_accs, bin_sizes):
                ax.annotate(f"n={n}", (x, y), fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Observed Accuracy")
        ax.set_title("Exp2: Calibration Diagram")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _save(fig, os.path.join(d, "exp2_calibration_diagram.png"))

    # ── Plot 3: Brier score by game ──
    by_game = defaultdict(list)
    for m in episode_metrics:
        if "mean_brier" in m:
            by_game[m["game_name"]].append(m["mean_brier"])

    if by_game:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        games = [g for g in config.GAME_NAMES if g in by_game]
        means = [_safe_avg(by_game[g]) for g in games]
        errs = [_safe_se(by_game[g]) for g in games]
        x = np.arange(len(games))
        ax.bar(x, means, yerr=errs, capsize=4, color=COLORS_4[:len(games)],
               edgecolor="white", linewidth=0.5)
        # Uniform baseline
        uniform_brier = sum((1.0 / len(config.OPPONENT_TYPES) - (1.0 if i == 0 else 0.0)) ** 2
                            for i in range(len(config.OPPONENT_TYPES)))
        ax.axhline(uniform_brier, color="red", linestyle="--", alpha=0.6,
                    label=f"Uniform baseline ({uniform_brier:.3f})")
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_", " ").title() for g in games])
        ax.set_ylabel("Mean Brier Score")
        ax.set_title("Exp2: Brier Score by Game")
        ax.legend()
        _save(fig, os.path.join(d, "exp2_brier_by_game.png"))


# ═══════════════════════════════════════════════════════
# Exp3: Belief-Action Coupling
# ═══════════════════════════════════════════════════════

def plot_exp3(analyzer_result: dict, provider: str, model: str):
    """Exp3 plots: variant comparison, belief use capability."""
    vm = analyzer_result.get("variant_metrics", {})
    cm = analyzer_result.get("coupling_metrics", [])
    if not vm:
        return
    d = _plot_dir(provider, model)

    # ── Plot 1: Variant comparison (A vs C payoffs by game) ──
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    games = config.GAME_NAMES
    x = np.arange(len(games))
    width = 0.35

    for vi, (v, label, color) in enumerate([("A", "Baseline", COLORS_4[0]),
                                             ("C", "Oracle", COLORS_4[1])]):
        ms_by_game = defaultdict(list)
        for m in vm.get(v, []):
            if "normalized_payoff" in m:
                ms_by_game[m["game_name"]].append(m["normalized_payoff"])
        means = [_safe_avg(ms_by_game.get(g, [])) for g in games]
        errs = [_safe_se(ms_by_game.get(g, [])) for g in games]
        ax.bar(x + vi * width, means, width, yerr=errs, capsize=4,
               label=f"Variant {v} ({label})", color=color, edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([g.replace("_", " ").title() for g in games])
    ax.set_ylabel("Normalized Payoff")
    ax.set_title("Exp3: Variant A (Baseline) vs C (Oracle-Injected)")
    ax.legend()
    ax.set_ylim(0, 1)
    _save(fig, os.path.join(d, "exp3_variant_comparison.png"))

    # ── Plot 2: Belief use capability by game ──
    if cm:
        buc_by_game = defaultdict(list)
        for c in cm:
            if "belief_use_capability" in c:
                buc_by_game[c["game_name"]].append(c["belief_use_capability"])

        if buc_by_game:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            games_with = [g for g in games if g in buc_by_game]
            means = [_safe_avg(buc_by_game[g]) for g in games_with]
            errs = [_safe_se(buc_by_game[g]) for g in games_with]
            x = np.arange(len(games_with))
            colors = [COLORS_4[2] if m >= 0 else COLORS_4[3] for m in means]
            ax.bar(x, means, yerr=errs, capsize=4, color=colors, edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([g.replace("_", " ").title() for g in games_with])
            ax.set_ylabel("Belief Use Capability (C - A)")
            ax.set_title("Exp3: Belief Use Capability (Oracle-Posterior Condition)")
            _save(fig, os.path.join(d, "exp3_belief_use.png"))


# ═══════════════════════════════════════════════════════
# Exp4: Posterior Intervention
# ═══════════════════════════════════════════════════════

def plot_exp4(analyzer_result: dict, provider: str, model: str):
    """Exp4 plots: by transform, by game."""
    valid = analyzer_result.get("valid_probes", [])
    if not valid:
        return
    d = _plot_dir(provider, model)

    # ── Plot 1: By transform ──
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    transforms = ["adversarial_flip", "plausible_perturbation"]
    x = np.arange(len(transforms))
    width = 0.35

    for mi, (metric, label, color) in enumerate([
        ("dir_acc_cf", "EU-Consistent DirAcc", COLORS_4[0]),
        ("sensitive", "Sensitivity", COLORS_4[1]),
    ]):
        means = []
        errs = []
        for tn in transforms:
            vals = [p[metric] for p in valid if p["transform"] == tn]
            means.append(_safe_avg(vals))
            errs.append(_safe_se(vals))
        ax.bar(x + mi * width, means, width, yerr=errs, capsize=4,
               label=label, color=color, edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([t.replace("_", " ").title() for t in transforms])
    ax.set_ylabel("Rate")
    ax.set_title("Exp4: Intervention Metrics by Transform Type")
    ax.legend()
    ax.set_ylim(0, 1)
    _save(fig, os.path.join(d, "exp4_by_transform.png"))

    # ── Plot 2: By game ──
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    games = [g for g in config.GAME_NAMES if any(p["game_name"] == g for p in valid)]
    x = np.arange(len(games))
    width = 0.35

    for mi, (metric, label, color) in enumerate([
        ("dir_acc_cf", "EU-Consistent DirAcc", COLORS_4[0]),
        ("sensitive", "Sensitivity", COLORS_4[1]),
    ]):
        means = [_safe_avg([p[metric] for p in valid if p["game_name"] == g]) for g in games]
        errs = [_safe_se([p[metric] for p in valid if p["game_name"] == g]) for g in games]
        ax.bar(x + mi * width, means, width, yerr=errs, capsize=4,
               label=label, color=color, edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([g.replace("_", " ").title() for g in games])
    ax.set_ylabel("Rate")
    ax.set_title("Exp4: Intervention Metrics by Game")
    ax.legend()
    ax.set_ylim(0, 1)
    _save(fig, os.path.join(d, "exp4_by_game.png"))


# ═══════════════════════════════════════════════════════
# Exp6: Cognitive Theory Metrics
# ═══════════════════════════════════════════════════════

def plot_exp6(analyzer_result: dict, provider: str, model: str):
    """Exp6 plots: cognitive radar, JS by game."""
    all_metrics = analyzer_result.get("all_metrics", [])
    if not all_metrics:
        return
    d = _plot_dir(provider, model)

    by_game = defaultdict(list)
    for m in all_metrics:
        by_game[m["game_name"]].append(m)

    # ── Plot 1: Cognitive radar chart ──
    # Axes: JS, ECE, Rigidity, Surprisal, Surprisal-Update Corr
    metric_keys = ["h1_mean_js", "h4_ece", "h5_rigidity_index",
                   "h7_mean_surprisal", "h7_surprisal_update_corr"]
    metric_labels = ["JS Divergence", "ECE", "Rigidity",
                     "Mean Surprisal", "Surp-Update r"]
    # Theoretical max for normalization
    max_vals = [math.log(2), 1.0, 1.0, 5.0, 1.0]

    games_with_data = [g for g in config.GAME_NAMES if by_game.get(g)]
    if games_with_data and len(metric_keys) >= 3:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        for idx, game in enumerate(games_with_data):
            ms = by_game[game]
            values = []
            for mk, mx in zip(metric_keys, max_vals):
                raw = _safe_avg([m.get(mk, 0) for m in ms if mk in m])
                values.append(min(abs(raw) / mx, 1.0) if mx > 0 else 0.0)
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=game.replace("_", " ").title(),
                    color=COLORS_4[idx % len(COLORS_4)])
            ax.fill(angles, values, alpha=0.1, color=COLORS_4[idx % len(COLORS_4)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Exp6: Cognitive Metrics Radar", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        _save(fig, os.path.join(d, "exp6_cognitive_radar.png"))

    # ── Plot 2: JS divergence by game ──
    js_by_game = {g: [m["h1_mean_js"] for m in ms if "h1_mean_js" in m]
                  for g, ms in by_game.items()}
    games_js = [g for g in config.GAME_NAMES if js_by_game.get(g)]
    if games_js:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        means = [_safe_avg(js_by_game[g]) for g in games_js]
        errs = [_safe_se(js_by_game[g]) for g in games_js]
        x = np.arange(len(games_js))
        ax.bar(x, means, yerr=errs, capsize=4, color=COLORS_4[:len(games_js)],
               edgecolor="white")
        ax.axhline(math.log(2), color="red", linestyle="--", alpha=0.5,
                    label=f"Max JS = ln(2) = {math.log(2):.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_", " ").title() for g in games_js])
        ax.set_ylabel("Mean JS Divergence")
        ax.set_title("Exp6: JS Divergence by Game")
        ax.legend()
        _save(fig, os.path.join(d, "exp6_js_by_game.png"))


# ═══════════════════════════════════════════════════════
# Plot all experiments
# ═══════════════════════════════════════════════════════

def plot_all(analyzer_results: dict, provider: str, model: str):
    """Generate all per-experiment plots from analyzer results."""
    if "exp1" in analyzer_results:
        plot_exp1(analyzer_results["exp1"], provider, model)
    if "exp2" in analyzer_results:
        plot_exp2(analyzer_results["exp2"], provider, model)
    if "exp3" in analyzer_results:
        plot_exp3(analyzer_results["exp3"], provider, model)
    if "exp4" in analyzer_results:
        plot_exp4(analyzer_results["exp4"], provider, model)
    if "exp6" in analyzer_results:
        plot_exp6(analyzer_results["exp6"], provider, model)
    logger.info(f"All plots generated in {_plot_dir(provider, model)}")


# ═══════════════════════════════════════════════════════
# Model Comparison Plots
# ═══════════════════════════════════════════════════════

def plot_model_comparison(results_dir: str):
    """Generate cross-model comparison plots from existing CSV data."""
    # Scan for model directories
    model_dirs = []
    if not os.path.exists(results_dir):
        return
    for entry in os.listdir(results_dir):
        full = os.path.join(results_dir, entry)
        if os.path.isdir(full) and entry != "model_comparison":
            model_dirs.append((entry, full))

    if len(model_dirs) < 2:
        logger.info("Need >= 2 model results for comparison plots. Skipping.")
        return

    out_dir = os.path.join(results_dir, "model_comparison")
    os.makedirs(out_dir, exist_ok=True)

    model_names = []
    exp1_payoffs = {}
    exp2_kls = {}
    exp4_da = {}
    exp4_sens = {}

    for name, path in model_dirs:
        model_names.append(name)

        # Exp1 data
        exp1_csv = os.path.join(path, "exp1_baseline", "data.csv")
        if os.path.exists(exp1_csv):
            df = pd.read_csv(exp1_csv)
            if "normalized_payoff" in df.columns:
                exp1_payoffs[name] = df["normalized_payoff"].dropna().tolist()

        # Exp2 data (need per-episode KL; approximate from round-level won't work,
        # so we just use the CSV and compute means per episode)
        exp2_csv = os.path.join(path, "exp2_belief", "data.csv")
        if os.path.exists(exp2_csv):
            df = pd.read_csv(exp2_csv)
            if "confidence" in df.columns:
                # We don't have KL in the round-level CSV. Use episode-level from summary.
                pass  # KL comparison requires analyzer return data, not raw CSV

        # Exp4 data
        exp4_csv = os.path.join(path, "exp4_intervention", "data.csv")
        if os.path.exists(exp4_csv):
            df = pd.read_csv(exp4_csv)
            valid = df[df["is_valid"] == True] if "is_valid" in df.columns else df
            if "dir_acc_cf" in valid.columns:
                exp4_da[name] = valid["dir_acc_cf"].dropna().tolist()
            if "sensitive" in valid.columns:
                exp4_sens[name] = valid["sensitive"].dropna().tolist()

    # ── Payoff comparison ──
    if exp1_payoffs:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        names = [n for n in model_names if n in exp1_payoffs]
        means = [_safe_avg(exp1_payoffs[n]) for n in names]
        errs = [_safe_se(exp1_payoffs[n]) for n in names]
        x = np.arange(len(names))
        ax.bar(x, means, yerr=errs, capsize=4, color=COLORS_4[:len(names)],
               edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Mean Normalized Payoff")
        ax.set_title("Model Comparison: Baseline Payoff")
        ax.set_ylim(0, 1)
        _save(fig, os.path.join(out_dir, "payoff_comparison.png"))

    # ── Intervention sensitivity comparison ──
    if exp4_da or exp4_sens:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        names = sorted(set(list(exp4_da.keys()) + list(exp4_sens.keys())))
        x = np.arange(len(names))
        width = 0.35

        if exp4_da:
            means_da = [_safe_avg(exp4_da.get(n, [])) for n in names]
            errs_da = [_safe_se(exp4_da.get(n, [])) for n in names]
            ax.bar(x, means_da, width, yerr=errs_da, capsize=4,
                   label="Directional Accuracy", color=COLORS_4[0], edgecolor="white")
        if exp4_sens:
            means_s = [_safe_avg(exp4_sens.get(n, [])) for n in names]
            errs_s = [_safe_se(exp4_sens.get(n, [])) for n in names]
            ax.bar(x + width, means_s, width, yerr=errs_s, capsize=4,
                   label="Sensitivity", color=COLORS_4[1], edgecolor="white")

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Rate")
        ax.set_title("Model Comparison: Intervention Metrics")
        ax.legend()
        ax.set_ylim(0, 1)
        _save(fig, os.path.join(out_dir, "intervention_sensitivity.png"))

    logger.info(f"Model comparison plots saved to {out_dir}")
