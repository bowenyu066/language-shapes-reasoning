#!/usr/bin/env python3
"""
Evaluate frontier API models on MMATH (multilingual hard math).

Models: ChatGPT-5.1, Gemini-3, DeepSeek-R1/R2

Usage:
    python scripts/11_eval_mmath_api.py
    python scripts/11_eval_mmath_api.py --model chatgpt-5.1 --language en
    python scripts/11_eval_mmath_api.py --limit 10  # For testing
"""

import argparse
import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_interface import OpenAIChatModel, GeminiModel, DeepSeekAPIModel
from src.eval_runner import run_experiment, generate_output_path
from src.data_utils import ensure_dir, load_jsonl, save_jsonl


# Model configurations
API_MODELS = {
    "chatgpt-5.1": {
        "type": "openai",
        "model_name": "gpt-5.1",
        "env_key": "OPENAI_API_KEY"
    },
    "gemini-3": {
        "type": "gemini",
        "model_name": "gemini-3-pro-preview",
        "env_key": "GOOGLE_API_KEY"
    },
    "deepseek-v3.2": {
        "type": "deepseek",
        "model_name": "deepseek-reasoner",
        "env_key": "DEEPSEEK_API_KEY"
    },
}

# Experiment configurations
EXPERIMENT_CONFIGS = [
    # (language, mode, max_tokens)
    # ("en", "direct", 128),
    # ("en", "direct", 256),
    # ("en", "direct", 512),
    # ("en", "direct", 1024),
    # ("zh", "direct", 128),
    # ("zh", "direct", 256),
    # ("zh", "direct", 512),
    ("zh", "direct", 4096),
]


def create_model(model_name: str):
    """Create an API model instance."""
    if model_name not in API_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(API_MODELS.keys())}")
    
    config = API_MODELS[model_name]
    model_type = config["type"]
    
    # Check for API key
    env_key = config.get("env_key", "")
    if env_key and not os.environ.get(env_key):
        print(f"Warning: {env_key} not set. API calls may fail.")
    
    if model_type == "openai":
        return OpenAIChatModel(
            name=model_name,
            model_name=config["model_name"]
        )
    elif model_type == "gemini":
        return GeminiModel(
            name=model_name,
            model_name=config["model_name"]
        )
    elif model_type == "deepseek":
        return DeepSeekAPIModel(
            name=model_name,
            model_name=config["model_name"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def filter_dataset_by_language(
    input_path: str,
    language: str,
    output_path: str
) -> str:
    """
    Filter MMATH dataset by language and save to temp file.
    
    Args:
        input_path: Path to full multilingual dataset.
        language: Language to filter by.
        output_path: Path to save filtered dataset.
        
    Returns:
        Path to filtered dataset.
    """
    records = load_jsonl(input_path)
    filtered = [r for r in records if r.get("language") == language]
    ensure_dir(os.path.dirname(output_path))
    save_jsonl(output_path, filtered)
    return output_path


def run_all_experiments(
    models: list[str],
    configs: list[tuple],
    output_dir: str,
    limit: int | None = None
):
    """
    Run all experiment configurations.
    
    Args:
        models: List of model names to evaluate.
        configs: List of (language, mode, max_tokens) tuples.
        output_dir: Directory to save results.
        limit: Optional limit on examples per experiment.
    """
    ensure_dir(output_dir)
    
    mmath_path = "data/processed/mmath"
    
    if not os.path.exists(mmath_path):
        print(f"Error: Dataset not found: {mmath_path}")
        print("Please run: python scripts/02_process_mmath.py first")
        sys.exit(1)
    
    summaries = []
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        try:
            model = create_model(model_name)
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            continue
        
        for language, mode, max_tokens in configs:
            # Create filtered dataset for this language
            filtered_path = f"data/processed/mmath/mmath_{language}.jsonl"
            
            output_path = generate_output_path(
                output_dir=output_dir,
                dataset_name="mmath",
                model_name=model_name,
                language=language,
                mode=mode,
                max_tokens=max_tokens
            )
            
            try:
                summary = run_experiment(
                    model=model,
                    dataset_path=filtered_path,
                    dataset_name="mmath",
                    language=language,
                    mode=mode,
                    max_tokens=max_tokens,
                    output_csv_path=output_path,
                    limit=limit
                )
                summaries.append(summary)
            except Exception as e:
                print(f"Error in experiment: {e}")
                import traceback
                traceback.print_exc()
    
    return summaries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate API models on MMATH"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(API_MODELS.keys()),
        help="Specific model to evaluate (default: all)"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "zh"],
        help="Specific language to evaluate (default: all)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct"],
        help="Specific mode to evaluate (default: all)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens setting"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mmath",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Determine models to run
    if args.model:
        models = [args.model]
    else:
        models = list(API_MODELS.keys())
    
    # Determine configs to run
    if args.language or args.max_tokens:
        configs = []
        for lang, mode, max_tok in EXPERIMENT_CONFIGS:
            if args.language and lang != args.language:
                continue
            if args.max_tokens:
                max_tok = args.max_tokens
            configs.append((lang, mode, max_tok))
    else:
        configs = EXPERIMENT_CONFIGS
    
    print("=" * 60)
    print("MMATH API Model Evaluation")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Configs: {configs}")
    print(f"Output: {args.output_dir}")
    if args.limit:
        print(f"Limit: {args.limit} examples per experiment")
    
    # Check API keys
    print("\nAPI Key Status:")
    for model_name in models:
        config = API_MODELS.get(model_name, {})
        env_key = config.get("env_key", "")
        status = "✓ Set" if os.environ.get(env_key) else "✗ Not set"
        print(f"  {model_name}: {env_key} {status}")
    
    # Run experiments
    summaries = run_all_experiments(
        models=models,
        configs=configs,
        output_dir=args.output_dir,
        limit=args.limit
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for s in summaries:
        print(f"{s['model']} | {s['language']} | {s['mode']} | "
              f"acc={s['accuracy']:.4f} ({s['correct']}/{s['n']})")
        with open(os.path.join(
            args.output_dir,
            f"summary_{s['model']}_{s['language']}_{s['mode']}_{s['max_tokens']}.json"),
            "w",
        ) as f:
            json.dump(s, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
