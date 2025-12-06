#!/usr/bin/env python3
"""
Evaluate open-source models on GSM8K (English + Chinese).

Models: Qwen3-8B, Llama-3.1-8B, DeepSeekMath-7B

Usage:
    python scripts/10_eval_gsm8k_local.py
    python scripts/10_eval_gsm8k_local.py --model qwen3-8b --language en
    python scripts/10_eval_gsm8k_local.py --limit 10  # For testing
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_interface import LocalModel
from src.eval_runner import run_experiment, generate_output_path
from src.data_utils import ensure_dir


# Model configurations
LOCAL_MODELS = {
    "qwen3-8b": {
        "model_path": "Qwen/Qwen3-8B",
        "backend": "transformers",
        "use_temperature": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    },
    "llama-3.1-8b": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "transformers",
        "use_temperature": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    },
    "deepseekmath-7b": {
        "model_path": "deepseek-ai/deepseek-math-7b-instruct",
        "backend": "transformers",
        "use_temperature": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    }
}

# Experiment configurations
EXPERIMENT_CONFIGS = [
    # (language, mode, max_tokens)
    ("en", "direct", 128),
    ("en", "direct", 256),
    ("zh", "direct", 128),
    ("zh", "direct", 256),
    ("zh", "zh_translate_then_solve", 256),
]


def create_model(model_name: str) -> LocalModel:
    """Create a LocalModel instance."""
    if model_name not in LOCAL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(LOCAL_MODELS.keys())}")
    
    config = LOCAL_MODELS[model_name]
    return LocalModel(
        name=model_name,
        model_path=config["model_path"],
        backend=config["backend"],
    )


def get_dataset_path(language: str) -> str:
    """Get the dataset path for a language."""
    return f"data/processed/gsm8k/gsm8k_{language}_test.jsonl"


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
            # Skip translate-then-solve for English
            if mode == "zh_translate_then_solve" and language == "en":
                continue
            
            dataset_path = get_dataset_path(language)
            
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset not found: {dataset_path}")
                continue
            
            output_path = generate_output_path(
                output_dir=output_dir,
                dataset_name="gsm8k",
                model_name=model_name,
                language=language,
                mode=mode,
                max_tokens=max_tokens
            )
            
            try:
                summary = run_experiment(
                    model=model,
                    dataset_path=dataset_path,
                    dataset_name="gsm8k",
                    language=language,
                    mode=mode,
                    max_tokens=max_tokens,
                    output_csv_path=output_path,
                    limit=limit
                )
                summaries.append(summary)
            except NotImplementedError as e:
                print(f"Skipping (not implemented): {e}")
            except Exception as e:
                print(f"Error in experiment: {e}")
    
    return summaries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate local models on GSM8K"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(LOCAL_MODELS.keys()),
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
        choices=["direct", "zh_translate_then_solve"],
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
        default="results/gsm8k",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Determine models to run
    if args.model:
        models = [args.model]
    else:
        models = list(LOCAL_MODELS.keys())
    
    # Determine configs to run
    if args.language or args.mode or args.max_tokens:
        configs = []
        for lang, mode, max_tok in EXPERIMENT_CONFIGS:
            if args.language and lang != args.language:
                continue
            if args.mode and mode != args.mode:
                continue
            if args.max_tokens:
                max_tok = args.max_tokens
            configs.append((lang, mode, max_tok))
    else:
        configs = EXPERIMENT_CONFIGS
    
    print("=" * 60)
    print("GSM8K Local Model Evaluation")
    print("=" * 60)
    print(f"Models: {models}")
    print(f"Configs: {configs}")
    print(f"Output: {args.output_dir}")
    if args.limit:
        print(f"Limit: {args.limit} examples per experiment")
    
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
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
