#!/usr/bin/env python3
"""
Evaluate fine-tuned models on GSM8K (English + Chinese).

This script evaluates models that have been fine-tuned using scripts/30_post_training.py.
The fine-tuned models are stored as LoRA adapters in outputs/sft/{model}/{language}/{timestamp}/final.

Usage:
    python scripts/31_eval_finetuned_gsm8k.py
    python scripts/31_eval_finetuned_gsm8k.py --model llama-3.1-8b --language zh
    python scripts/31_eval_finetuned_gsm8k.py --limit 10  # For testing
    python scripts/31_eval_finetuned_gsm8k.py --adapter-path outputs/sft/llama-3.1-8b/zh/20251208_123456/final
"""

import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_interface import BaseModel
from src.eval_runner import run_experiment, generate_output_path
from src.data_utils import ensure_dir


# Model configurations (base models)
BASE_MODELS = {
    "llama-3.1-8b": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20
    },
    "qwen3-8b": {
        "model_path": "Qwen/Qwen3-8B",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20
    },
}

# Experiment configurations
EXPERIMENT_CONFIGS = [
    # (language, mode, max_tokens)
    ("en", "direct", 4096),
    ("zh", "direct", 4096),
    # ("zh", "zh_translate_then_solve", 4096),
]


class FinetunedLocalModel(BaseModel):
    """
    Local model wrapper for fine-tuned models with LoRA adapters.
    """

    def __init__(
        self,
        name: str,
        base_model_path: str,
        adapter_path: str,
        device: str = "cuda",
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 20,
    ):
        """
        Initialize a fine-tuned local model.

        Args:
            name: Display name for the model.
            base_model_path: HuggingFace model ID for the base model.
            adapter_path: Path to the LoRA adapter weights.
            device: Device to run on.
            temperature: Default sampling temperature.
            top_p: Default top-p sampling parameter.
            top_k: Default top-k sampling parameter.
        """
        self.name = name
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.default_top_k = top_k
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model with LoRA adapter."""
        if self._model is not None:
            return

        print(f"Loading base model: {self.base_model_path}")
        print(f"Loading adapter from: {self.adapter_path}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            cache_dir=".cache/models",
        )
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            cache_dir=".cache/models",
            device_map=self.device
        )

        # Load LoRA adapter
        self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self._model.eval()

        print(f"Model loaded successfully with adapter")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> str:
        """Generate response using the fine-tuned model."""
        self._load_model()

        # Use defaults if not specified
        temperature = temperature if temperature is not None else self.default_temperature
        top_p = top_p if top_p is not None else self.default_top_p
        top_k = top_k if top_k is not None else self.default_top_k

        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]

        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self._tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        ).to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: list[str],
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> list[str]:
        """Generate responses for a batch of prompts."""
        return [
            self.generate(prompt, max_tokens, temperature, top_p, top_k)
            for prompt in prompts
        ]


def find_latest_adapter(model_name: str, language: str, base_dir: str = "outputs/sft") -> Optional[str]:
    """
    Find the latest adapter checkpoint for a given model and language.

    Args:
        model_name: Model name (e.g., "llama-3.1-8b")
        language: Language code (e.g., "zh")
        base_dir: Base directory for SFT outputs

    Returns:
        Path to the latest adapter, or None if not found.
    """
    model_dir = os.path.join(base_dir, model_name, language)

    if not os.path.exists(model_dir):
        return None

    # Find all timestamp directories
    timestamps = []
    for d in os.listdir(model_dir):
        final_path = os.path.join(model_dir, d, "final")
        if os.path.isdir(final_path):
            timestamps.append((d, final_path))

    if not timestamps:
        return None

    # Sort by timestamp (newest first)
    timestamps.sort(reverse=True)
    return timestamps[0][1]


def create_finetuned_model(
    model_name: str,
    adapter_path: str,
) -> FinetunedLocalModel:
    """Create a FinetunedLocalModel instance."""
    if model_name not in BASE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BASE_MODELS.keys())}")

    config = BASE_MODELS[model_name]

    # Extract training language from adapter path for the display name
    # e.g., outputs/sft/llama-3.1-8b/zh/20251208_123456/final -> zh
    path_parts = adapter_path.split(os.sep)
    try:
        # Find the language in the path
        if "sft" in path_parts:
            sft_idx = path_parts.index("sft")
            if sft_idx + 2 < len(path_parts):
                train_lang = path_parts[sft_idx + 2]
                display_name = f"{model_name}-ft-{train_lang}"
            else:
                display_name = f"{model_name}-finetuned"
        else:
            display_name = f"{model_name}-finetuned"
    except (ValueError, IndexError):
        display_name = f"{model_name}-finetuned"

    return FinetunedLocalModel(
        name=display_name,
        base_model_path=config["model_path"],
        adapter_path=adapter_path,
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
    )


def get_dataset_path(language: str) -> str:
    """Get the dataset path for a language."""
    return f"data/processed/gsm8k/gsm8k_{language}_test.jsonl"


def run_all_experiments(
    model_name: str,
    adapter_path: str,
    configs: list[tuple],
    output_dir: str,
    limit: Optional[int] = None
):
    """
    Run all experiment configurations.

    Args:
        model_name: Base model name.
        adapter_path: Path to the LoRA adapter.
        configs: List of (language, mode, max_tokens) tuples.
        output_dir: Directory to save results.
        limit: Optional limit on examples per experiment.
    """
    ensure_dir(output_dir)

    summaries = []

    print(f"\n{'='*60}")
    print(f"Model: {model_name} (fine-tuned)")
    print(f"Adapter: {adapter_path}")
    print(f"{'='*60}")

    try:
        model = create_finetuned_model(model_name, adapter_path)
    except Exception as e:
        print(f"Error creating model: {e}")
        return summaries

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
            model_name=model.name,
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
            import traceback
            traceback.print_exc()

    return summaries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned models on GSM8K"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(BASE_MODELS.keys()),
        default="llama-3.1-8b",
        help="Base model to evaluate (default: llama-3.1-8b)"
    )
    parser.add_argument(
        "--train-language",
        type=str,
        choices=["en", "zh"],
        default="zh",
        help="Language the model was fine-tuned on (default: zh)"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Explicit path to the LoRA adapter (overrides auto-detection)"
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
        default="results/gsm8k_finetuned",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Find or verify adapter path
    if args.adapter_path:
        adapter_path = args.adapter_path
        if not os.path.exists(adapter_path):
            print(f"Error: Adapter path does not exist: {adapter_path}")
            sys.exit(1)
    else:
        adapter_path = find_latest_adapter(args.model, args.train_language)
        if adapter_path is None:
            print(f"Error: No fine-tuned adapter found for {args.model} ({args.train_language})")
            print(f"Expected location: outputs/sft/{args.model}/{args.train_language}/*/final")
            print("Please run scripts/30_post_training.py first to fine-tune the model.")
            sys.exit(1)

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
    print("GSM8K Fine-tuned Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Training language: {args.train_language}")
    print(f"Adapter: {adapter_path}")
    print(f"Configs: {configs}")
    print(f"Output: {args.output_dir}")
    if args.limit:
        print(f"Limit: {args.limit} examples per experiment")

    # Run experiments
    summaries = run_all_experiments(
        model_name=args.model,
        adapter_path=adapter_path,
        configs=configs,
        output_dir=args.output_dir,
        limit=args.limit
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for s in summaries:
        print(f"{s['model']} | {s['language']} | {s['mode']} | max_tok={s['max_tokens']} | "
              f"acc={s['accuracy']:.4f} ({s['correct']}/{s['n']})")

    # Save summary to text file
    summary_path = os.path.join(args.output_dir, "summary.txt")
    ensure_dir(args.output_dir)
    with open(summary_path, "a") as f:
        f.write("=" * 60 + "\n")
        f.write("GSM8K Fine-tuned Model Evaluation Results Summary\n")
        f.write(f"Adapter: {adapter_path}\n")
        f.write("=" * 60 + "\n\n")

        for s in summaries:
            f.write(f"Max Tokens={s['max_tokens']} | {s['model']} | {s['language']} | {s['mode']} | "
                   f"acc={s['accuracy']:.4f} ({s['correct']}/{s['n']})\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Evaluation complete!\n")
        f.write(f"Results saved to: {args.output_dir}\n")
        f.write("=" * 60 + "\n\n")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
