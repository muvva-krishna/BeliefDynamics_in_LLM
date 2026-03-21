"""
Checkpoint system for saving and resuming experiment progress.
Saves after each completed episode so that server errors don't lose progress.
"""
import json
import os
import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)


def _checkpoint_path(experiment_name: str, provider: str, model: str) -> str:
    """Generate a checkpoint file path."""
    safe_model = model.replace("/", "_").replace(":", "_")
    filename = f"ckpt_{experiment_name}_{provider}_{safe_model}.json"
    return os.path.join(config.CHECKPOINTS_DIR, filename)


def save_checkpoint(
    experiment_name: str,
    provider: str,
    model: str,
    completed_episodes: list[dict],
    pending_conditions: list[dict],
    metadata: dict | None = None,
) -> str:
    """
    Save experiment progress to a checkpoint file.

    Args:
        experiment_name: e.g. "exp1_baseline"
        provider: API provider name
        model: model string
        completed_episodes: list of completed episode result dicts
        pending_conditions: list of (game, opponent_type, seed, ...) still to run
        metadata: any additional metadata

    Returns:
        Path to the saved checkpoint file.
    """
    path = _checkpoint_path(experiment_name, provider, model)

    checkpoint = {
        "experiment_name": experiment_name,
        "provider": provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "n_completed": len(completed_episodes),
        "n_pending": len(pending_conditions),
        "completed_episodes": completed_episodes,
        "pending_conditions": pending_conditions,
        "metadata": metadata or {},
    }

    # Write atomically: write to temp file then rename
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, default=str)
    os.replace(temp_path, path)

    logger.info(f"Checkpoint saved: {len(completed_episodes)} episodes done, "
                f"{len(pending_conditions)} pending → {path}")
    return path


def load_checkpoint(
    experiment_name: str,
    provider: str,
    model: str,
) -> dict | None:
    """
    Load a checkpoint if it exists.

    Returns:
        The checkpoint dict, or None if no checkpoint exists.
    """
    path = _checkpoint_path(experiment_name, provider, model)
    if not os.path.exists(path):
        logger.info(f"No checkpoint found at {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint: {checkpoint['n_completed']} episodes completed, "
                     f"{checkpoint['n_pending']} pending")
        return checkpoint
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Corrupted checkpoint at {path}: {e}")
        return None


def clear_checkpoint(experiment_name: str, provider: str, model: str) -> None:
    """Delete a checkpoint file after experiment completes successfully."""
    path = _checkpoint_path(experiment_name, provider, model)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Checkpoint cleared: {path}")


# ═══════════════════════════════════════════════════════
# Pipeline-level checkpointing
# ═══════════════════════════════════════════════════════

def _pipeline_checkpoint_path(provider: str, model: str, tag: str = "") -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    suffix = f"_{tag}" if tag else ""
    return os.path.join(config.CHECKPOINTS_DIR, f"ckpt_pipeline_{provider}_{safe_model}{suffix}.json")


def _strip_api_logs(cond: dict) -> dict:
    """Remove api_log entries from all stages before saving to checkpoint."""
    import copy
    c = copy.deepcopy(cond)
    for stage_key in ("stage1", "stage2", "stage3", "stage4"):
        stage = c.get(stage_key)
        if isinstance(stage, dict):
            stage.pop("api_log", None)
    return c


def save_pipeline_checkpoint(
    provider: str, model: str,
    completed: list[dict], pending: list[dict],
    tag: str = "",
) -> str:
    path = _pipeline_checkpoint_path(provider, model, tag)
    data = {
        "type": "pipeline",
        "provider": provider,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "n_completed": len(completed),
        "n_pending": len(pending),
        "completed": [_strip_api_logs(c) for c in completed],
        "pending": pending,
    }
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(temp_path, path)
    logger.info(f"Pipeline checkpoint: {len(completed)} done, {len(pending)} pending")
    return path


def load_pipeline_checkpoint(provider: str, model: str, tag: str = "") -> dict | None:
    path = _pipeline_checkpoint_path(provider, model, tag)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Corrupted pipeline checkpoint: {e}")
        return None


def clear_pipeline_checkpoint(provider: str, model: str, tag: str = "") -> None:
    path = _pipeline_checkpoint_path(provider, model, tag)
    if os.path.exists(path):
        os.remove(path)
        logger.info(f"Pipeline checkpoint cleared: {path}")


def list_checkpoints() -> list[dict]:
    """List all existing checkpoint files with summary info."""
    checkpoints = []
    ckpt_dir = config.CHECKPOINTS_DIR
    if not os.path.exists(ckpt_dir):
        return []
    for fname in os.listdir(ckpt_dir):
        if fname.startswith("ckpt_") and fname.endswith(".json"):
            path = os.path.join(ckpt_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints.append({
                    "file": fname,
                    "experiment": data.get("experiment_name"),
                    "provider": data.get("provider"),
                    "model": data.get("model"),
                    "completed": data.get("n_completed", 0),
                    "pending": data.get("n_pending", 0),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                checkpoints.append({"file": fname, "error": "corrupted"})
    return checkpoints
