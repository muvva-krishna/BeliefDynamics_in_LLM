"""
CLI entry point for the unified adversarial game experiment pipeline.

Usage:
  python run.py -p openai                              # Full pipeline + all analyzers
  python run.py -p anthropic -m claude-sonnet-4-20250514 --episodes 30
  python run.py -p openai --stages stage1,stage2       # Run specific stages only
  python run.py -p openai --analyze exp1,exp2          # Just compute metrics from existing data
  python run.py -p openai --plot                       # Pipeline + analyzers + plots
  python run.py --compare-models                       # Cross-model comparison plots
  python run.py --list-checkpoints
"""
import argparse
import logging
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import config
from pipeline import UnifiedPipeline
from analyzers import Exp1Analyzer, Exp2Analyzer, Exp3Analyzer, Exp4Analyzer, Exp6Analyzer, run_all_analyzers
from checkpoint import list_checkpoints, load_pipeline_checkpoint

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, provider: str = "", model: str = ""):
    """Setup logging to stdout + per-model log file."""
    level = logging.DEBUG if verbose else logging.INFO

    # Log into per-model directory if provider/model given
    if provider and model:
        safe_model = model.replace("/", "_").replace(":", "_")
        log_dir = os.path.join(config.RESULTS_DIR, f"{provider}_{safe_model}")
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = config.RESULTS_DIR

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir,
                             f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                encoding="utf-8",
            ),
        ],
    )

    # Silence noisy third-party loggers — only show warnings+
    for noisy in ("httpx", "httpcore", "google_genai", "google_genai.models",
                  "urllib3", "google.auth", "google.auth.transport",
                  "openai", "openai._base_client", "openai.http_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_VERTEX_PROVIDERS = {"vertex_llama", "vertex_anthropic", "vertex_deepseek"}

def validate_api_key(provider: str):
    # Vertex AI providers authenticate via Application Default Credentials — no API key needed
    if provider in _VERTEX_PROVIDERS:
        if not config.GCP_PROJECT_ID:
            print("ERROR: GCP_PROJECT_ID is not set. Add it to your .env file.")
            sys.exit(1)
        return  # auth via gcloud ADC

    keys = {
        "openai":    config.OPENAI_API_KEY,
        "anthropic": config.ANTHROPIC_API_KEY,
        "groq":      config.GROQ_API_KEY,
        "gemini":    config.GEMINI_API_KEY,
    }
    if not keys.get(provider, ""):
        print(f"ERROR: No API key for '{provider}'. Set {provider.upper()}_API_KEY env var.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Unified Adversarial Game Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py -p openai                                  # Full pipeline
  python run.py -p anthropic --episodes 30                 # More episodes
  python run.py -p openai --stages stage1,stage2           # Specific stages
  python run.py -p openai --analyze exp1,exp2              # Metrics only
  python run.py -p openai --plot                           # Pipeline + plots
  python run.py -p groq --games prisoners_dilemma          # Single game
  python run.py --compare-models                           # Cross-model plots
  python run.py --list-checkpoints                         # Show progress
        """,
    )

    parser.add_argument("--provider", "-p", type=str,
                        choices=["openai", "anthropic", "groq", "gemini",
                                 "vertex_llama", "vertex_anthropic", "vertex_deepseek"],
                        default=config.ACTIVE_PROVIDER,
                        help=f"API provider (default: {config.ACTIVE_PROVIDER} from .env)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name (defaults to provider's default)")
    parser.add_argument("--episodes", type=int, default=config.DEFAULT_EPISODES,
                        help=f"Episodes per (game x opponent) condition (default: {config.DEFAULT_EPISODES} from .env)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Rounds per episode (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated game names (default: all 4)")
    parser.add_argument("--opponents", type=str, default=None,
                        help="Comma-separated opponent types (default: all 9)")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stages: stage1,stage2,stage3,stage4,stage5 (default: all)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Run analyzers only on existing data: exp1,exp2,exp3,exp4,exp6 or 'all'")
    parser.add_argument("--plot", action="store_true", default=config.AUTO_PLOT,
                        help=f"Generate plots after run (default: {'on' if config.AUTO_PLOT else 'off'} from .env AUTO_PLOT)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plot generation even if AUTO_PLOT=true in .env")
    parser.add_argument("--compare-models", action="store_true",
                        help="Generate cross-model comparison plots from all existing results")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoints, start fresh")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List checkpoints and exit")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag suffix for checkpoint file — use different tag per parallel terminal (e.g. --tag pd)")
    parser.add_argument("--merge-tags", type=str, default=None,
                        help="Merge multiple tagged checkpoints into final output. e.g. --merge-tags pd,sh,bos,ch")

    args = parser.parse_args()
    # --no-plot overrides AUTO_PLOT
    if args.no_plot:
        args.plot = False

    # List checkpoints mode
    if args.list_checkpoints:
        ckpts = list_checkpoints()
        if not ckpts:
            print("No checkpoints found.")
        else:
            for c in ckpts:
                if "error" in c:
                    print(f"  {c['file']}: CORRUPTED")
                else:
                    print(f"  {c.get('experiment') or 'pipeline'} | {c.get('provider','?')} | "
                          f"{c.get('model','?')} | {c.get('completed',0)} done | "
                          f"{c.get('pending',0)} pending | {c.get('timestamp','?')}")
        return

    # Merge tagged checkpoints into one final output
    if args.merge_tags:
        tags = [t.strip() for t in args.merge_tags.split(",")]
        model = args.model or config.DEFAULT_MODELS.get(args.provider, "")
        setup_logging(args.verbose, args.provider, model)
        all_completed = []
        for t in tags:
            ckpt = load_pipeline_checkpoint(args.provider, model, tag=t)
            if ckpt and ckpt.get("completed"):
                print(f"  Tag '{t}': {len(ckpt['completed'])} conditions loaded")
                all_completed.extend(ckpt["completed"])
            else:
                print(f"  Tag '{t}': no checkpoint found, skipping")
        if not all_completed:
            print("No data found across tags. Run the per-game terminals first.")
            return

        # Report what we found
        games_found = set(c["game_name"] for c in all_completed)
        for g in sorted(games_found):
            n = sum(1 for c in all_completed if c["game_name"] == g)
            print(f"    {g}: {n} conditions")
        print(f"\nMerged {len(all_completed)} total conditions across {len(games_found)} games.")
        print("Running FULL REBUILD of all analyzers...")

        # Use analyze() for a clean from-scratch rebuild (mode="w", not append)
        # This avoids all append-vs-overwrite bugs
        _analyzer_final = [Exp1Analyzer(), Exp2Analyzer(), Exp3Analyzer(), Exp4Analyzer(), Exp6Analyzer()]
        for az in _analyzer_final:
            try:
                az.analyze(all_completed, args.provider, model)
            except Exception as e:
                logger.warning(f"Merge analyzer failed for {az.__class__.__name__}: {e}")

        safe_model = model.replace("/", "_").replace(":", "_")
        out_dir = os.path.join(config.RESULTS_DIR, f"{args.provider}_{safe_model}")
        print(f"Merged results saved to: {out_dir}/")

        # Sanity checks on merged data
        try:
            from analyzers import sanity_check_exp3, sanity_check_exp4
            exp3_warns = sanity_check_exp3(all_completed)
            exp4_warns = sanity_check_exp4(all_completed)
            if exp3_warns or exp4_warns:
                print(f"  Exp3 warnings: {len(exp3_warns)} | Exp4 warnings: {len(exp4_warns)}")
                for w in exp3_warns + exp4_warns:
                    print(f"    {w}")
            else:
                print("  All sanity checks passed.")
        except Exception as e:
            logger.warning(f"Sanity check failed: {e}")

        if args.plot:
            from plots import plot_all
            analyzer_results = run_all_analyzers(all_completed, args.provider, model)
            plot_all(analyzer_results, args.provider, model)
            print("Plots generated.")
        return

    # Cross-model comparison only (no provider required)
    if args.compare_models and not args.provider:
        from plots import plot_model_comparison
        plot_model_comparison(config.RESULTS_DIR)
        print(f"Model comparison plots saved to {os.path.join(config.RESULTS_DIR, 'model_comparison')}")
        return

    if not args.provider:
        print(f"No provider set. Configure ACTIVE_PROVIDER in .env or use -p flag.")
        parser.print_help()
        sys.exit(1)

    model = args.model or config.DEFAULT_MODELS.get(args.provider, "")
    setup_logging(args.verbose, args.provider, model)
    validate_api_key(args.provider)

    games = args.games.split(",") if args.games else None
    opponents = args.opponents.split(",") if args.opponents else None
    stages = args.stages.split(",") if args.stages else None

    # Analyze-only mode: load existing pipeline data and run analyzers
    if args.analyze:
        exps = args.analyze.split(",") if args.analyze != "all" else None
        ckpt = load_pipeline_checkpoint(args.provider, model)
        if ckpt and ckpt.get("completed"):
            print(f"Loading {len(ckpt['completed'])} conditions from checkpoint...")
            analyzer_results = run_all_analyzers(ckpt["completed"], args.provider, model, exps)
            if args.plot:
                from plots import plot_all
                plot_all(analyzer_results, args.provider, model)
                print("Plots generated.")
        else:
            print("No pipeline data found. Run the pipeline first.")
        if args.compare_models:
            from plots import plot_model_comparison
            plot_model_comparison(config.RESULTS_DIR)
        return

    # Full pipeline mode
    safe_model = model.replace("/", "_").replace(":", "_")
    out_dir = os.path.join(config.RESULTS_DIR, f"{args.provider}_{safe_model}")

    # Live callback: write game sections when each game completes
    _written_games = set()

    pipeline = UnifiedPipeline(
        provider=args.provider,
        model=model,
        games=games,
        opponent_types=opponents,
        episodes_per_condition=args.episodes,
        rounds_per_episode=args.rounds,
        base_seed=args.seed,
        resume=not args.no_resume,
        stages=stages,
        on_condition_complete=None,
        tag=args.tag,
    )

    _expected_per_game = args.episodes * len(pipeline.opponent_types)

    def _live_update(completed, provider, mdl):
        _analyzer_instances = [Exp1Analyzer(), Exp2Analyzer(), Exp3Analyzer(), Exp4Analyzer(), Exp6Analyzer()]
        for game in pipeline.games:
            n_done = sum(1 for c in completed if c["game_name"] == game)
            if n_done >= _expected_per_game and game not in _written_games:
                _written_games.add(game)
                game_conds = [c for c in completed if c["game_name"] == game]
                for az in _analyzer_instances:
                    try:
                        az.write_csv(completed, provider, mdl)
                        # Only write summaries when NOT using --tag (parallel mode)
                        # When using --tag, summaries are built cleanly by --merge-tags
                        if not args.tag:
                            az.write_game_to_summary(game, game_conds, completed, provider, mdl)
                    except Exception as e:
                        logger.warning(f"{az.__class__.__name__} game-section failed: {e}")
                print(f"\n  [{game}] complete → results/{provider}_{mdl.replace('/','_')}/")

    pipeline.on_condition_complete = _live_update

    print("=" * 70)
    print("UNIFIED EXPERIMENT PIPELINE")
    print(f"Provider: {args.provider} | Model: {model}")
    print(f"Games: {pipeline.games}")
    print(f"Opponents: {len(pipeline.opponent_types)} types")
    print(f"Episodes/condition: {args.episodes} | Rounds/episode: {args.rounds}")
    print(f"Stages: {pipeline.stages}")
    print(f"Output: {out_dir}")
    print(f"Live CSV/TXT: written after every condition ✓")
    print("=" * 70)

    results = pipeline.run()

    # Final pass: write CSV + summaries
    print("\nFinalizing all experiments...")
    _analyzer_final = [Exp1Analyzer(), Exp2Analyzer(), Exp3Analyzer(), Exp4Analyzer(), Exp6Analyzer()]
    for az in _analyzer_final:
        try:
            az.write_csv(results, args.provider, model)
            if args.tag:
                # Tagged run: skip summary writing entirely — merge-tags does the full rebuild
                pass
            else:
                # Non-tagged run: use analyze() for a clean full rebuild (mode="w")
                az.analyze(results, args.provider, model)
        except Exception as e:
            logger.warning(f"Final pass failed for {az.__class__.__name__}: {e}")

    # Sanity checks for Exp3 and Exp4
    from analyzers import sanity_check_exp3, sanity_check_exp4
    print("\nRunning post-run sanity checks...")
    exp3_warns = sanity_check_exp3(results)
    exp4_warns = sanity_check_exp4(results)
    if exp3_warns or exp4_warns:
        print(f"  Exp3 warnings: {len(exp3_warns)} | Exp4 warnings: {len(exp4_warns)}")
        for w in exp3_warns + exp4_warns:
            print(f"    {w}")
    else:
        print("  All sanity checks passed.")

    # Build analyzer_results for optional plotting (uses analyze() which does full rebuild)
    analyzer_results = {}
    print(f"Results saved to: {out_dir}/")
    print(f"  exp1_baseline/data.csv + summary.txt")
    print(f"  exp2_belief/data.csv + summary.txt")
    print(f"  exp3_coupling/data.csv + summary.txt")
    print(f"  exp4_intervention/data.csv + summary.txt")
    print(f"  exp6_cognitive/data.csv + summary.txt")

    # Generate plots if requested (requires full analyze pass)
    if args.plot:
        from plots import plot_all
        analyzer_results = run_all_analyzers(results, args.provider, model)
        plot_all(analyzer_results, args.provider, model)
        print(f"  plots/ (PNG files)")

    if args.compare_models:
        from plots import plot_model_comparison
        plot_model_comparison(config.RESULTS_DIR)

    print(f"\nDone! Open: {out_dir}")


if __name__ == "__main__":
    main()
