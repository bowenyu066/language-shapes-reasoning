#!/usr/bin/env python3
"""
Compare token lengths of English vs Chinese outputs using different tokenizers.

This script loads model outputs from GSM8K evaluation results and compares
the token lengths when tokenized with different tokenizers (Qwen, DeepSeek, Seed).

The goal is to analyze whether Chinese outputs are more token-efficient than
English outputs for the same math problems.

Usage:
    python scripts/12_compare_token_lengths.py
    python scripts/12_compare_token_lengths.py --en-csv results/gsm8k/gsm8k_qwen3-8b_en_direct_maxtok4096.csv
    python scripts/12_compare_token_lengths.py --limit 100  # For testing
"""

import argparse
import csv
import os
import sys
from typing import Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import ensure_dir


# Tokenizer configurations
TOKENIZERS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "seed-coder-8b": "ByteDance-Seed/Seed-Coder-8B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt-4o": "openai/gpt-4o",  # Uses tiktoken
}


def load_csv_results(csv_path: str) -> list[dict]:
    """Load evaluation results from a CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_tokenizer(tokenizer_name: str):
    """Load a tokenizer from HuggingFace or tiktoken for OpenAI models."""
    print(f"Loading tokenizer: {tokenizer_name}...")

    # GPT-4o uses tiktoken
    if tokenizer_name == "openai/gpt-4o":
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        return tokenizer

    # Other models use HuggingFace
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        cache_dir=".cache/models",
    )
    return tokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in a text string."""
    # tiktoken tokenizers (for OpenAI models) don't have add_special_tokens parameter
    if hasattr(tokenizer, 'encode') and not hasattr(tokenizer, 'add_special_tokens'):
        # tiktoken style
        tokens = tokenizer.encode(text)
    else:
        # HuggingFace style
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def extract_answer_portion(raw_output: str) -> str:
    """
    Extract the model's answer portion from the raw output.

    The raw_output includes the prompt, so we need to extract just the
    model's response. We look for the assistant's response after the prompt.
    """
    # The raw output typically contains the full conversation
    # We want to extract just the model's generated answer

    # For Qwen models, look for the assistant marker
    if "<|assistant|>" in raw_output:
        parts = raw_output.split("<|assistant|>")
        if len(parts) > 1:
            return parts[-1].strip()

    # For other formats, try to find the answer after common markers
    markers = [
        "assistant\n",
        "Assistant:",
        "\n\n",  # Often the response starts after double newline
    ]

    for marker in markers:
        if marker in raw_output:
            idx = raw_output.rfind(marker)
            if idx != -1:
                return raw_output[idx + len(marker):].strip()

    # If no marker found, return the whole output
    return raw_output


def analyze_token_lengths(
    en_results: list[dict],
    zh_results: list[dict],
    tokenizers: dict,
    limit: Optional[int] = None,
    use_answer_only: bool = True,
    both_correct_only: bool = True,
) -> dict:
    """
    Analyze token lengths for English vs Chinese outputs.

    Args:
        en_results: English evaluation results.
        zh_results: Chinese evaluation results.
        tokenizers: Dictionary of tokenizer names to tokenizer objects.
        limit: Optional limit on number of examples to analyze.
        use_answer_only: If True, extract just the answer portion (excluding prompt).
        both_correct_only: If True, only analyze examples where both EN and ZH are correct.

    Returns:
        Dictionary with analysis results.
    """
    # Create lookup by ID for matching
    en_by_id = {r["id"]: r for r in en_results}
    zh_by_id = {r["id"]: r for r in zh_results}

    # Find common IDs (questions answered in both languages)
    common_ids = set(en_by_id.keys()) & set(zh_by_id.keys())
    print(f"Found {len(common_ids)} common questions answered in both languages")

    # Filter for both correct if requested
    if both_correct_only:
        correct_ids = set()
        for qid in common_ids:
            en_correct = str(en_by_id[qid].get("correct", "0")) == "1"
            zh_correct = str(zh_by_id[qid].get("correct", "0")) == "1"
            if en_correct and zh_correct:
                correct_ids.add(qid)
        print(f"Filtered to {len(correct_ids)} examples where both EN and ZH are correct")
        common_ids = correct_ids

    if limit:
        common_ids = sorted(list(common_ids))[:limit]
        print(f"Limiting analysis to {len(common_ids)} examples")

    # Analyze each tokenizer
    results = {}

    for tok_name, tokenizer in tokenizers.items():
        print(f"\nAnalyzing with {tok_name} tokenizer...")

        en_tokens = []
        zh_tokens = []
        ratios = []  # zh_tokens / en_tokens

        for qid in sorted(common_ids):
            en_output = en_by_id[qid]["raw_output"]
            zh_output = zh_by_id[qid]["raw_output"]

            if use_answer_only:
                en_output = extract_answer_portion(en_output)
                zh_output = extract_answer_portion(zh_output)

            en_tok_count = count_tokens(en_output, tokenizer)
            zh_tok_count = count_tokens(zh_output, tokenizer)

            en_tokens.append(en_tok_count)
            zh_tokens.append(zh_tok_count)

            if en_tok_count > 0:
                ratios.append(zh_tok_count / en_tok_count)

        # Compute statistics
        avg_en = sum(en_tokens) / len(en_tokens) if en_tokens else 0
        avg_zh = sum(zh_tokens) / len(zh_tokens) if zh_tokens else 0
        avg_ratio = avg_zh / avg_en if avg_en > 0 else 0  # (Avg ZH) / (Avg EN)

        # Compute median
        sorted_en = sorted(en_tokens)
        sorted_zh = sorted(zh_tokens)
        n = len(en_tokens)
        median_en = sorted_en[n // 2] if n % 2 == 1 else (sorted_en[n // 2 - 1] + sorted_en[n // 2]) / 2
        median_zh = sorted_zh[n // 2] if n % 2 == 1 else (sorted_zh[n // 2 - 1] + sorted_zh[n // 2]) / 2
        median_ratio = median_zh / median_en if median_en > 0 else 0  # (Median ZH) / (Median EN)

        results[tok_name] = {
            "n_samples": len(en_tokens),
            "avg_en_tokens": avg_en,
            "avg_zh_tokens": avg_zh,
            "median_en_tokens": median_en,
            "median_zh_tokens": median_zh,
            "avg_ratio": avg_ratio,  # (Avg ZH) / (Avg EN)
            "median_ratio": median_ratio,  # (Median ZH) / (Median EN)
            "total_en_tokens": sum(en_tokens),
            "total_zh_tokens": sum(zh_tokens),
            "en_tokens_list": en_tokens,
            "zh_tokens_list": zh_tokens,
        }

        print(f"  Samples: {len(en_tokens)}")
        print(f"  Avg EN tokens: {avg_en:.1f}")
        print(f"  Avg ZH tokens: {avg_zh:.1f}")
        print(f"  Avg ZH/EN ratio: {avg_ratio:.3f} (= {avg_zh:.1f} / {avg_en:.1f})")
        print(f"  Median ZH/EN ratio: {median_ratio:.3f} (= {median_zh:.1f} / {median_en:.1f})")

    return results


def save_detailed_results(
    en_results: list[dict],
    zh_results: list[dict],
    tokenizers: dict,
    output_path: str,
    limit: Optional[int] = None,
    use_answer_only: bool = True,
    both_correct_only: bool = True,
):
    """Save detailed per-example token counts to a CSV file."""
    en_by_id = {r["id"]: r for r in en_results}
    zh_by_id = {r["id"]: r for r in zh_results}
    common_ids = set(en_by_id.keys()) & set(zh_by_id.keys())

    # Filter for both correct if requested
    if both_correct_only:
        common_ids = {
            qid for qid in common_ids
            if str(en_by_id[qid].get("correct", "0")) == "1"
            and str(zh_by_id[qid].get("correct", "0")) == "1"
        }

    common_ids = sorted(common_ids)

    if limit:
        common_ids = common_ids[:limit]

    # Build rows
    rows = []
    for qid in common_ids:
        en_output = en_by_id[qid]["raw_output"]
        zh_output = zh_by_id[qid]["raw_output"]

        if use_answer_only:
            en_text = extract_answer_portion(en_output)
            zh_text = extract_answer_portion(zh_output)
        else:
            en_text = en_output
            zh_text = zh_output

        row = {
            "id": qid,
            "en_correct": en_by_id[qid].get("correct", ""),
            "zh_correct": zh_by_id[qid].get("correct", ""),
            "en_char_len": len(en_text),
            "zh_char_len": len(zh_text),
        }

        for tok_name, tokenizer in tokenizers.items():
            en_tok = count_tokens(en_text, tokenizer)
            zh_tok = count_tokens(zh_text, tokenizer)
            row[f"{tok_name}_en_tokens"] = en_tok
            row[f"{tok_name}_zh_tokens"] = zh_tok
            row[f"{tok_name}_ratio"] = zh_tok / en_tok if en_tok > 0 else 0

        rows.append(row)

    # Write CSV
    ensure_dir(os.path.dirname(output_path))
    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDetailed results saved to: {output_path}")


def print_summary_table(results: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("TOKEN LENGTH COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Tokenizer':<20} {'Avg EN':>10} {'Avg ZH':>10} {'Avg Ratio':>12} {'Median Ratio':>14}")
    print("-" * 80)

    for tok_name, stats in results.items():
        print(f"{tok_name:<20} {stats['avg_en_tokens']:>10.1f} {stats['avg_zh_tokens']:>10.1f} "
              f"{stats['avg_ratio']:>12.3f} {stats['median_ratio']:>14.3f}")

    print("-" * 80)
    print("\nInterpretation:")
    print("  - Ratio < 1.0: Chinese uses fewer tokens than English")
    print("  - Ratio > 1.0: Chinese uses more tokens than English")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare token lengths of English vs Chinese outputs"
    )
    parser.add_argument(
        "--en-csv",
        type=str,
        default="results/gsm8k/gsm8k_qwen3-8b_en_direct_maxtok4096.csv",
        help="Path to English results CSV"
    )
    parser.add_argument(
        "--zh-csv",
        type=str,
        default="results/gsm8k/gsm8k_qwen3-8b_zh_direct_maxtok4096.csv",
        help="Path to Chinese results CSV"
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        choices=list(TOKENIZERS.keys()),
        default=list(TOKENIZERS.keys()),
        help="Tokenizers to use for comparison"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to analyze (for testing)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/token_analysis",
        help="Output directory for detailed results"
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Analyze full output including prompt (default: answer portion only)"
    )
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include examples where EN or ZH answer is incorrect (default: both must be correct)"
    )

    args = parser.parse_args()

    # Verify input files exist
    if not os.path.exists(args.en_csv):
        print(f"Error: English CSV not found: {args.en_csv}")
        sys.exit(1)
    if not os.path.exists(args.zh_csv):
        print(f"Error: Chinese CSV not found: {args.zh_csv}")
        sys.exit(1)

    print("=" * 80)
    print("Token Length Comparison: English vs Chinese Outputs")
    print("=" * 80)
    print(f"English CSV: {args.en_csv}")
    print(f"Chinese CSV: {args.zh_csv}")
    print(f"Tokenizers: {args.tokenizers}")
    print(f"Analyze: {'Full output' if args.full_output else 'Answer portion only'}")
    print(f"Filter: {'All examples' if args.include_incorrect else 'Both correct only'}")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    print()

    # Load results
    print("Loading evaluation results...")
    en_results = load_csv_results(args.en_csv)
    zh_results = load_csv_results(args.zh_csv)
    print(f"  English results: {len(en_results)} examples")
    print(f"  Chinese results: {len(zh_results)} examples")

    # Load tokenizers
    print("\nLoading tokenizers...")
    tokenizers = {}
    for tok_name in args.tokenizers:
        try:
            tokenizers[tok_name] = load_tokenizer(TOKENIZERS[tok_name])
        except Exception as e:
            print(f"  Warning: Failed to load {tok_name}: {e}")

    if not tokenizers:
        print("Error: No tokenizers loaded successfully")
        sys.exit(1)

    # Analyze token lengths
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    both_correct_only = not args.include_incorrect

    results = analyze_token_lengths(
        en_results=en_results,
        zh_results=zh_results,
        tokenizers=tokenizers,
        limit=args.limit,
        use_answer_only=not args.full_output,
        both_correct_only=both_correct_only,
    )

    # Print summary
    print_summary_table(results)

    # Save detailed results
    output_path = os.path.join(
        args.output_dir,
        f"token_comparison_{'full' if args.full_output else 'answer'}_{'all' if args.include_incorrect else 'correct'}.csv"
    )
    save_detailed_results(
        en_results=en_results,
        zh_results=zh_results,
        tokenizers=tokenizers,
        output_path=output_path,
        limit=args.limit,
        use_answer_only=not args.full_output,
        both_correct_only=both_correct_only,
    )

    # Save summary statistics
    summary_path = os.path.join(args.output_dir, "summary.txt")
    ensure_dir(args.output_dir)
    with open(summary_path, 'w') as f:
        f.write("Token Length Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"English CSV: {args.en_csv}\n")
        f.write(f"Chinese CSV: {args.zh_csv}\n")
        f.write(f"Analysis type: {'Full output' if args.full_output else 'Answer portion only'}\n")
        f.write(f"Filter: {'All examples' if args.include_incorrect else 'Both correct only'}\n\n")

        for tok_name, stats in results.items():
            f.write(f"\n{tok_name}:\n")
            f.write(f"  Samples: {stats['n_samples']}\n")
            f.write(f"  Avg EN tokens: {stats['avg_en_tokens']:.1f}\n")
            f.write(f"  Avg ZH tokens: {stats['avg_zh_tokens']:.1f}\n")
            f.write(f"  Median EN tokens: {stats['median_en_tokens']:.1f}\n")
            f.write(f"  Median ZH tokens: {stats['median_zh_tokens']:.1f}\n")
            f.write(f"  Avg ZH/EN ratio: {stats['avg_ratio']:.4f}\n")
            f.write(f"  Median ZH/EN ratio: {stats['median_ratio']:.4f}\n")
            f.write(f"  Total EN tokens: {stats['total_en_tokens']}\n")
            f.write(f"  Total ZH tokens: {stats['total_zh_tokens']}\n")

    print(f"Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
