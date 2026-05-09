[README.md](https://github.com/user-attachments/files/27545719/README.md)
# BeliefDynamics in LLM

**Evaluating Belief Formation and Belief–Action Coupling in Large Language Models**

> *Do LLM actions depend on LLM beliefs — or are stated beliefs and actions just two parallel outputs of the same heuristic?*

This repository is the codebase for the paper *"Evaluating Belief Formation and Belief–Action Coupling in Large Language Models"* (Muvva Krishna Harshith, BITS Pilani). It introduces a benchmark that instruments the full **inference → belief representation → action** pipeline of LLMs through repeated strategic interaction with hidden opponent types, and tests the causal link between stated beliefs and realized actions.

---

## Overview

Existing LLM benchmarks measure behavioral outcomes — payoff, win rate, task success. These metrics cannot distinguish an agent that *reasons* from one that *approximates reasoning* through heuristics. This benchmark addresses that gap directly.

The core question is **causal**: does an LLM’s stated posterior over opponent types actually cause its action selection? To answer this, we place LLMs in partially-observable repeated 2×2 games where the opponent’s hidden policy creates a tractable Bayesian inference problem. We elicit beliefs explicitly, compare them to an oracle posterior, and — crucially — **inject counterfactual beliefs** while holding the game state fixed to test the belief–action link causally.

**Central finding:** Stated beliefs and realized actions are largely decoupled processes. Models frequently report posteriors prescribing one EU-optimal action while selecting another. Strong behavioral performance does not imply coherent belief-driven reasoning.

---

## Benchmark Design

### Games

Four canonical 2×2 games with qualitatively distinct incentive structures:

| Game | (C,C) | (C,D) | (D,C) | (D,D) |
|---|---|---|---|---|
| Prisoner’s Dilemma | (3,3) | (0,5) | (5,0) | (1,1) |
| Stag Hunt | (4,4) | (0,3) | (3,0) | (2,2) |
| Chicken | (3,3) | (1,5) | (5,1) | (0,0) |
| Battle of the Sexes | (3,2) | (0,0) | (0,0) | (2,3) |

The game selection is not cosmetic — different payoff structures invert the EU-optimal action under different beliefs, isolating whether action selection is belief-driven or a game-structure heuristic.

### Opponent Types (9)

| Type | Behavior | Stochastic? |
|---|---|---|
| `always_cooperate` | Always C | No |
| `always_defect` | Always D | No |
| `tit_for_tat` | Copies agent’s last action | No |
| `grim_trigger` | C until first agent D, then D forever | No |
| `win_stay_lose_shift` | Repeat if payoff ≥ midpoint, else switch | No |
| `noisy_tit_for_tat` | Tit-for-tat with 10% flip noise | Yes |
| `deceptive_opportunist` | C for rounds 1–3, then D w.p. 0.80 | Yes |
| `gradual_defector` | Retaliates with *k* Ds per *k*-th agent defection | No |
| `adaptive_mirror` | C iff agent cooperation rate in last 5 rounds > 0.60 | No |

The deceptive subset `{deceptive_opportunist, gradual_defector}` requires second-order inference — the agent must attribute early cooperation to *instrumental* strategy, not genuine cooperative intent.

### Pipeline Stages

All five stages run over a single shared condition `(game, opponent_type, episode, seed)`, so opponent behavioral sequences are identical across stages:

| Stage | Description | API Calls/Condition |
|---|---|---|
| Stage 1 | Baseline gameplay + belief probes at t ∈ {3,6,10} | 13 |
| Stage 2 | Two-stage B2 protocol (belief-first, then action) | 6 |
| Stage 3 | Oracle-injected gameplay (ground-truth posterior provided) | 10 |
| Stage 4 | Posterior intervention (counterfactual belief injection) | 0–12 |
| Stage 5 | Cognitive theory metrics (offline, 0 API calls) | 0 |

**Full factorial design:** 4 games × 9 opponent types × 10 episodes = **360 conditions per model**, yielding 1,080 belief-quality datapoints and up to 4,320 intervention queries before validity filtering.

---

## Metric Suite

Metrics form a **strict hierarchy of causal claim strength**:

| Level | Metric | What it tests |
|---|---|---|
| L1 | `Sens` — Sensitivity | Does the model respond to any posterior change at all? |
| L2 | `CAA` — Conditional Action Agreement | Do expressed beliefs align with action choices? |
| L3 | `Coh(B)` — EU-Consistent Coherence | Are actions EU-optimal under the model’s own stated beliefs? |
| L4 | `DirAcc(flip)` — Directional Accuracy | Does causally injected belief produce EU-optimal action? |

A model can pass lower levels while failing higher ones. The pattern of failures precisely locates the breakdown in the pipeline.

Additional metrics: `KL` / `JS` divergence from oracle posterior, `Brier score`, `TypeAcc`, `ECE` / `MCE` calibration, `BUC` (Belief Use Capability = oracle payoff − baseline payoff), `RI` (Rigidity Index), `GapToM` (Theory-of-Mind deceptive-type gap), `ρH7` (surprisal–update correlation).

Full metric definitions are in `metrics.py`; per-experiment computation logic is in `analyzers.py`.

---

## Models Evaluated

| Provider Key | Model String | Backend |
|---|---|---|
| `openai` | `gpt-5.4-nano` | OpenAI Chat Completions |
| `gemini` | `gemini-2.0-flash` | Google GenAI |
| `vertex_gemini` | `gemini-2.5-pro` | Vertex AI (GCP) |
| `vertex_llama` | `meta/llama-4-maverick-17b-128e-instruct-maas` | Vertex AI Model Garden |
| `groq` | `llama-3.1-8b-instant` | Groq |
| `vertex_deepseek` | `deepseek-ai/deepseek-v3.2-maas` | Vertex AI Model Garden |

Model strings are hardcoded as defaults in `config.py` and can be overridden via `--model` or environment variables.

---

## Key Results

- **EU-consistent coherence averages near chance** across all games — stated beliefs and action selection are largely decoupled.
- **4 of 6 models perform *worse* with oracle-accurate beliefs** than with their own inferred beliefs (negative BUC), revealing behavioral lock-in from training.
- **Chicken is the universal diagnostic game** — every model achieves Coh(B) > 0.83 in Stag Hunt but collapses in Chicken, the only game where no constant action policy approximates EU-optimality.
- **ρH7 = 0.000 for every model** — prediction error magnitude and belief update magnitude are completely orthogonal. LLMs re-infer beliefs from full context at each step; they do not update them.
- **Two-axis characterization:** Inference quality (how well beliefs track the oracle) and coupling fidelity (whether beliefs drive action) are empirically independent. No model occupies the high-inference, high-coupling ideal quadrant.

---

## Installation

```bash
git clone https://github.com/muvva-krishna/BeliefDynamics_in_LLM.git
cd BeliefDynamics_in_LLM
pip install -r requirements.txt
```

**Requirements:** `openai>=1.30.0`, `anthropic>=0.39.0`, `groq>=0.9.0`, `google-genai>=1.0.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.10.0`, `matplotlib>=3.7.0`

---

## Configuration

Edit `.env` in the project root:

```bash
# Provider to use by default
ACTIVE_PROVIDER=openai

# API keys — only the provider(s) you intend to run need to be set
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...

# Vertex AI providers (vertex_llama, vertex_deepseek, vertex_gemini)
# authenticate via Google Application Default Credentials — no API key needed
GCP_PROJECT_ID=your-gcp-project-id
# Run once: gcloud auth application-default login

# Run settings
DEFAULT_EPISODES=10
AUTO_PLOT=false
```

---

## Usage

### Full pipeline

```bash
# Run all 5 stages for the default provider
python run.py -p openai

# Override model and episode count
python run.py -p anthropic -m claude-sonnet-4-20250514 --episodes 30

# Restrict to specific games or opponents
python run.py -p openai --games prisoners_dilemma,chicken
python run.py -p openai --opponents tit_for_tat,always_defect,deceptive_opportunist

# Pipeline + generate all plots
python run.py -p openai --plot
```

### Run specific stages only

```bash
python run.py -p openai --stages stage1,stage2
python run.py -p openai --stages stage4          # Intervention only (requires Stage 1 data)
```

### Analyze existing checkpoint data without re-running

```bash
python run.py -p openai --analyze exp1,exp2
python run.py -p openai --analyze all --plot
```

### Parallel execution across games (recommended for large runs)

Run one terminal per game with a `--tag`, then merge when all finish:

```bash
# Four terminals in parallel
python run.py -p openai --games prisoners_dilemma --tag pd
python run.py -p openai --games stag_hunt --tag sh
python run.py -p openai --games chicken --tag ch
python run.py -p openai --games battle_of_sexes --tag bos

# Merge and rebuild all analyzers from the combined data
python run.py -p openai --merge-tags pd,sh,ch,bos --plot
```

### Cross-model comparison

```bash
# After running multiple providers/models, generate comparison figures
python run.py --compare-models
```

### Checkpoint management

```bash
python run.py --list-checkpoints        # Show progress for all saved runs
python run.py -p openai --no-resume     # Ignore checkpoint, start fresh
```

---

## Project Structure

```
BeliefDynamics_in_LLM/
├── run.py              # CLI entry point — start here
├── pipeline.py         # Core 5-stage evaluation pipeline (UnifiedPipeline)
├── config.py           # Provider defaults, model strings, directory paths
├── games.py            # Payoff matrices for all 4 games
├── opponents.py        # 9 opponent policy implementations
├── oracle.py           # Bayesian oracle posterior (log-space, ε-smoothed)
├── prompts.py          # All prompt templates (system, action, probe, B2, oracle, Stage 4)
├── api_clients.py      # Multi-provider API wrappers with exponential-backoff retry
├── metrics.py          # Full metric suite (KL, Brier, Coh, DirAcc, RI, GapToM, ρH7 ...)
├── analyzers.py        # Exp1–Exp4, Exp6 analyzers + sanity checks
├── plots.py            # All figures from the paper
├── checkpoint.py       # Atomic condition-level checkpointing (os.replace)
├── requirements.txt
├── .env                # API keys and run config (not committed)
├── checkpoints/        # Auto-saved run state — supports mid-run resumption
└── results/
    └── {provider}_{model}/
        ├── exp1_baseline/      data.csv  summary.txt
        ├── exp2_belief/        data.csv  summary.txt
        ├── exp3_coupling/      data.csv  summary.txt
        ├── exp4_intervention/  data.csv  summary.txt
        ├── exp6_cognitive/     data.csv  summary.txt
        └── plots/              PNG figures
```

---

## Reproducibility

Each condition `(game, opponent_type, episode_index)` is assigned a deterministic seed:

```
s = 42 + hash(game, opponent_type, episode_index) mod 2^31
```

This guarantees identical opponent behavioral sequences across all five pipeline stages and across provider re-runs. Every API call is assigned a 16-character SHA-256 prompt hash stored in the log alongside response metadata, enabling post-hoc verification of prompt integrity.

All experiments use `temperature=0.0` and `top_p=1.0`. JSON output is enforced via provider-specific mechanisms: strict schema mode for OpenAI, `response_schema` + MIME type for Gemini, `json_object` mode for Groq and Vertex Model Garden backends. Parse failure rate: 0 across all models and all experiments.

Checkpoints are written atomically via `os.replace()` after each completed condition, preventing corrupt state from partial writes. Runs can be interrupted and resumed at any point without re-running completed conditions.

---

## Citation

```bibtex
@article{harshith2025beliefdynamics,
  title   = {Evaluating Belief Formation and Belief--Action Coupling in Large Language Models},
  author  = {Muvva Krishna Harshith},
  school  = {Birla Institute of Technology and Science, Pilani},
  year    = {2025}
}
```

---

## License

MIT
