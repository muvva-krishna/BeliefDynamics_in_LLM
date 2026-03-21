"""
Configuration for the adversarial game experiment framework.
API keys and model settings are loaded from .env file.
"""
import os
from pathlib import Path

# ── Load .env file ──
_ENV_PATH = Path(__file__).parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#"):
                continue
            if "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip()
                # Only set if not already in environment (env vars take precedence)
                if _key and not os.environ.get(_key):
                    os.environ[_key] = _val

# ── API Keys ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── Google Cloud / Vertex AI ──
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
VERTEX_LLAMA_LOCATION   = os.environ.get("VERTEX_LLAMA_LOCATION",   "us-east5")
VERTEX_ANTHROPIC_LOCATION = os.environ.get("VERTEX_ANTHROPIC_LOCATION", "us-east5")
VERTEX_DEEPSEEK_LOCATION  = os.environ.get("VERTEX_DEEPSEEK_LOCATION",  "global")

# ── Active provider (can be overridden with -p flag) ──
ACTIVE_PROVIDER = os.environ.get("ACTIVE_PROVIDER", "openai")

# ── Default models per provider (loaded from .env or fallback) ──
DEFAULT_MODELS = {
    "openai":            os.environ.get("OPENAI_MODEL",           "gpt-4o"),
    "anthropic":         os.environ.get("ANTHROPIC_MODEL",        "claude-sonnet-4-20250514"),
    "groq":              os.environ.get("GROQ_MODEL",             "llama-3.3-70b-versatile"),
    "gemini":            os.environ.get("GEMINI_MODEL",           "gemini-2.0-flash"),
    # Vertex AI / Model Garden providers
    "vertex_llama":      os.environ.get("VERTEX_LLAMA_MODEL",     "meta/llama-4-maverick-17b-128e-instruct-maas"),
    "vertex_anthropic":  os.environ.get("VERTEX_ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    "vertex_deepseek":   os.environ.get("VERTEX_DEEPSEEK_MODEL",  "deepseek-ai/deepseek-v3.2-maas"),
}

# ── Rate-limit settings per provider (requests per minute, tokens per minute) ──
# These are conservative defaults; adjust in .env based on your subscription tier.
RATE_LIMITS = {
    "openai": {
        "rpm": int(os.environ.get("OPENAI_RPM", 30)),
        "tpm": int(os.environ.get("OPENAI_TPM", 80000)),
        "max_retries": 8, "base_delay": 2.0,
    },
    "anthropic": {
        "rpm": int(os.environ.get("ANTHROPIC_RPM", 30)),
        "tpm": int(os.environ.get("ANTHROPIC_TPM", 80000)),
        "max_retries": 8, "base_delay": 2.0,
    },
    "groq": {
        "rpm": int(os.environ.get("GROQ_RPM", 25)),
        "tpm": int(os.environ.get("GROQ_TPM", 50000)),
        "max_retries": 8, "base_delay": 2.0,
    },
    "gemini": {
        "rpm": int(os.environ.get("GEMINI_RPM", 10)),
        "tpm": int(os.environ.get("GEMINI_TPM", 60000)),
        "max_retries": 8, "base_delay": 3.0,
    },
    # Vertex AI providers — backed by GCP $300 credits, no hard throttle
    "vertex_llama": {
        "rpm": int(os.environ.get("VERTEX_LLAMA_RPM", 1000)),
        "tpm": int(os.environ.get("VERTEX_LLAMA_TPM", 10000000)),
        "max_retries": 8, "base_delay": 1.0,
    },
    "vertex_anthropic": {
        "rpm": int(os.environ.get("VERTEX_ANTHROPIC_RPM", 1000)),
        "tpm": int(os.environ.get("VERTEX_ANTHROPIC_TPM", 10000000)),
        "max_retries": 8, "base_delay": 1.0,
    },
    "vertex_deepseek": {
        "rpm": int(os.environ.get("VERTEX_DEEPSEEK_RPM", 1000)),
        "tpm": int(os.environ.get("VERTEX_DEEPSEEK_TPM", 10000000)),
        "max_retries": 8, "base_delay": 1.0,
    },
}

# ── Experiment defaults ──
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512          # For simple action-only calls
MAX_TOKENS_PROBE = 1024   # For probe/belief calls (9 opponent posteriors need more space)
ROUNDS_PER_EPISODE = 10
PROBE_ROUNDS = [3, 6, 10]  # 1-indexed rounds where belief probes happen

# ── Speed / output defaults (from .env) ──
DEFAULT_EPISODES = int(os.environ.get("DEFAULT_EPISODES", 10))
AUTO_PLOT = os.environ.get("AUTO_PLOT", "false").lower() == "true"

# ── Opponent types ──
OPPONENT_TYPES = [
    "always_cooperate",
    "always_defect",
    "tit_for_tat",
    "grim_trigger",
    "win_stay_lose_shift",
    "noisy_tit_for_tat",
    "deceptive_opportunist",
    "gradual_defector",
    "adaptive_mirror",
]

# ── Game names ──
GAME_NAMES = [
    "prisoners_dilemma",
    "stag_hunt",
    "chicken",
    "battle_of_sexes",
]

# ── Output paths ──
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
