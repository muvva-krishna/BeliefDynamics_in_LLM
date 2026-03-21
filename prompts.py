"""
Prompt templates for the unified pipeline.
Design: large system prompt (cached by providers) + minimal user prompts (dynamic only).
"""
import json
from games import get_game, GAME_DESCRIPTIONS
import config


# ═══════════════════════════════════════════════════════
# JSON Schemas (3 distinct schemas)
# ═══════════════════════════════════════════════════════

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["C", "D"]}
    },
    "required": ["action"],
    "additionalProperties": False,
}

PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["C", "D"]},
        "predicted_next_action": {"type": "string", "enum": ["C", "D"]},
        "confidence": {"type": "number"},
        "posterior_over_types": {
            "type": "object",
            "properties": {t: {"type": "number"} for t in config.OPPONENT_TYPES},
            "required": list(config.OPPONENT_TYPES),
            "additionalProperties": False,
        },
    },
    "required": ["action", "predicted_next_action", "confidence", "posterior_over_types"],
    "additionalProperties": False,
}

BELIEF_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "predicted_next_action": {"type": "string", "enum": ["C", "D"]},
        "confidence": {"type": "number"},
        "posterior_over_types": {
            "type": "object",
            "properties": {t: {"type": "number"} for t in config.OPPONENT_TYPES},
            "required": list(config.OPPONENT_TYPES),
            "additionalProperties": False,
        },
    },
    "required": ["predicted_next_action", "confidence", "posterior_over_types"],
    "additionalProperties": False,
}


# ═══════════════════════════════════════════════════════
# System Prompts (large, cached per episode)
# ═══════════════════════════════════════════════════════

def build_system_prompt(game_name: str) -> str:
    """
    Build the large system prompt containing all static content for a game.
    Cached by OpenAI/Anthropic/Gemini — sent once per episode, reused across rounds.
    """
    game = get_game(game_name)
    desc = GAME_DESCRIPTIONS[game_name]
    opp_types = ", ".join(config.OPPONENT_TYPES)

    return f"""You are a strategic JSON-only reasoning agent playing a repeated 2x2 game.
Your entire response must be a single JSON object. No explanation text.

GAME: {game_name}
{desc}

PAYOFF MATRIX (Your action, Opponent action -> your payoff, opponent payoff):
{game.format_matrix_verbose()}

KNOWN OPPONENT TYPES: {opp_types}

Your goal: infer the opponent's type from interaction history and maximize your cumulative payoff.

RESPONSE FORMATS:
- Action only: {{"action": "C"}} or {{"action": "D"}}
- Belief probe: {{"action": "C"|"D", "predicted_next_action": "C"|"D", "confidence": 0.0-1.0, "posterior_over_types": {{type: probability, ...}} (9 types, sum=1.0)}}
- Belief only: same as probe but without "action" key

HISTORY FORMAT: t=round self_action opp_action self_payoff opp_payoff"""


def build_oracle_system_prompt(game_name: str) -> str:
    """System prompt for oracle-injected pipeline (Exp3-C). No history — uses posterior."""
    game = get_game(game_name)
    desc = GAME_DESCRIPTIONS[game_name]

    return f"""You are a strategic JSON-only agent playing a repeated 2x2 game.
Your entire response must be a single JSON object: {{"action": "C"}} or {{"action": "D"}}

GAME: {game_name}
{desc}

PAYOFF MATRIX:
{game.format_matrix_verbose()}

You will be given a probability distribution over opponent types.
Each opponent type deterministically selects its next action given the history.
Choose the action that maximizes expected immediate payoff under the provided posterior.
No explanation text."""


def build_intervention_system_prompt(game_name: str) -> str:
    """System prompt for Exp4 posterior intervention queries."""
    game = get_game(game_name)
    desc = GAME_DESCRIPTIONS[game_name]

    return f"""You are a strategic JSON-only agent.
Your entire response must be: {{"action": "C"}} or {{"action": "D"}}

GAME: {game_name}
{desc}

PAYOFF MATRIX:
{game.format_matrix_verbose()}

You will receive a probability distribution over opponent types.
Each type deterministically selects its next action.
Choose the action that maximizes expected immediate payoff under the provided posterior.
No explanation text."""


# ═══════════════════════════════════════════════════════
# User Prompts (minimal, dynamic only)
# ═══════════════════════════════════════════════════════

def _format_history(history: list[dict]) -> str:
    """Compact history: one line per round, ~14 tokens each."""
    if not history:
        return ""
    lines = []
    for r in history:
        lines.append(
            f"t={r['t']:02d} {r['self_action']} {r['opp_action']} "
            f"{r['self_payoff']:.1f} {r['opp_payoff']:.1f}"
        )
    return "\n".join(lines)


def action_user_prompt(round_num: int, total_rounds: int,
                       history: list[dict]) -> str:
    """Minimal action-only prompt. Used for Stage 1 non-probe rounds."""
    h = _format_history(history)
    parts = [f"Round {round_num}/{total_rounds}"]
    if h:
        parts.append(h)
    parts.append('{"action":"C"} or {"action":"D"}')
    return "\n".join(parts)


def probe_user_prompt(round_num: int, total_rounds: int,
                      history: list[dict]) -> str:
    """Probe prompt: action + beliefs. Used for Stage 1 probe rounds."""
    h = _format_history(history)
    parts = [f"Round {round_num}/{total_rounds}"]
    if h:
        parts.append(h)
    parts.append(
        '[BELIEF PROBE] Reply with: "action", "predicted_next_action", '
        '"confidence", "posterior_over_types" (9 types, sum=1.0)'
    )
    return "\n".join(parts)


def b2_belief_prompt(round_num: int, total_rounds: int,
                     history: list[dict]) -> str:
    """B2 Call 1: belief only, no action. Used for Stage 2."""
    h = _format_history(history)
    parts = [f"Round {round_num}/{total_rounds}"]
    if h:
        parts.append(h)
    parts.append(
        '[BELIEF ONLY] Reply with: "predicted_next_action", '
        '"confidence", "posterior_over_types" (9 types, sum=1.0). No action key.'
    )
    return "\n".join(parts)


def b2_action_prompt(round_num: int, total_rounds: int,
                     history: list[dict],
                     posterior: dict[str, float]) -> str:
    """B2 Call 2: action from posterior. Used for Stage 2."""
    h = _format_history(history)
    # Only show top types to keep prompt short
    top = sorted(posterior.items(), key=lambda x: -x[1])[:5]
    post_str = ", ".join(f'"{k}":{v:.3f}' for k, v in top)
    parts = [f"Round {round_num}/{total_rounds}"]
    if h:
        parts.append(h)
    parts.append(f"Your beliefs: {{{post_str}}}")
    parts.append('Choose action maximizing expected payoff. {{"action":"C"}} or {{"action":"D"}}')
    return "\n".join(parts)


def oracle_user_prompt(round_num: int, total_rounds: int,
                       history: list[dict],
                       oracle_posterior: dict[str, float]) -> str:
    """Oracle-injected prompt. Used for Stage 3 (Exp3-C)."""
    h = _format_history(history)
    top = sorted(oracle_posterior.items(), key=lambda x: -x[1])[:5]
    post_str = ", ".join(f'"{k}":{v:.4f}' for k, v in top)
    parts = [f"Round {round_num}/{total_rounds}"]
    if h:
        parts.append(h)
    parts.append(f"Posterior over opponent types: {{{post_str}}}")
    parts.append('{"action":"C"} or {"action":"D"}')
    return "\n".join(parts)


def intervention_user_prompt(round_num: int, total_rounds: int,
                             history: list[dict],
                             posterior: dict[str, float]) -> str:
    """Exp4 posterior intervention prompt. Posterior only, no history context."""
    post_str = json.dumps({k: round(v, 6) for k, v in posterior.items()})
    parts = [
        f"Round {round_num}/{total_rounds}",
        f"Posterior: {post_str}",
        '{"action":"C"} or {"action":"D"}',
    ]
    return "\n".join(parts)
