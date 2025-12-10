#!/usr/bin/env python3
"""
Record token lengths for multilingual MMATH outputs using different tokenizers.

This script loads model outputs from MMATH evaluation results and records
the token lengths when tokenized with different tokenizers (Qwen, DeepSeek, GPT-4o, Llama).

The script processes four languages: English, Chinese, Spanish, and Thai.

Usage:
    python scripts/13_compare_token_lengths_mmath.py --model chatgpt-5.1
    python scripts/13_compare_token_lengths_mmath.py --model deepseek-v3.2 --limit 100
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


def load_multilang_results(model: str, base_dir: str = "results/mmath") -> dict:
    """
    Load evaluation results for all four languages.
    
    Args:
        model: Model name (e.g., 'chatgpt-5.1', 'deepseek-v3.2')
        base_dir: Base directory containing results
    
    Returns:
        Dictionary mapping language codes to list of result dictionaries
    """
    languages = ['en', 'zh', 'es', 'th']
    results = {}
    
    for lang in languages:
        csv_path = os.path.join(base_dir, f"mmath_{model}_{lang}_direct_maxtok8192.csv")
        if os.path.exists(csv_path):
            results[lang] = load_csv_results(csv_path)
            print(f"  Loaded {lang}: {len(results[lang])} examples")
        else:
            print(f"  Warning: File not found for {lang}: {csv_path}")
    
    return results


def record_token_lengths(
    multilang_results: dict,
    tokenizers: dict,
    limit: Optional[int] = None,
) -> dict:
    """
    Record token lengths for all languages.

    Args:
        multilang_results: Dictionary mapping language codes to result lists.
        tokenizers: Dictionary of tokenizer names to tokenizer objects.
        limit: Optional limit on number of examples to analyze.

    Returns:
        Dictionary with token count records for each language.
    """
    def _reg_id(id_):
        return id_.split("_")[1]
    
    # Create lookup by ID for each language
    results_by_id = {}
    all_ids = set()
    
    for lang, results in multilang_results.items():
        results_by_id[lang] = {_reg_id(r["id"]): r for r in results}
        all_ids.update(results_by_id[lang].keys())
    
    # Find common IDs across all languages
    common_ids = set.intersection(*[set(results_by_id[lang].keys()) for lang in multilang_results.keys()])
    print(f"\nFound {len(common_ids)} questions answered in all languages")
    print(f"Total unique IDs across all languages: {len(all_ids)}")
    
    # Filter for cases where ALL languages answered correctly
    all_correct_ids = set()
    for qid in common_ids:
        all_correct = True
        for lang in multilang_results.keys():
            if str(results_by_id[lang][qid].get("correct", "0")) != "1":
                all_correct = False
                break
        if all_correct:
            all_correct_ids.add(qid)
    
    print(f"Filtered to {len(all_correct_ids)} examples where all languages answered correctly")
    common_ids = all_correct_ids
    
    if limit:
        common_ids = sorted(list(common_ids))[:limit]
        print(f"Limiting analysis to {len(common_ids)} examples")
    
    # Record token counts for each tokenizer and language
    token_records = {}
    
    for tok_name, tokenizer in tokenizers.items():
        print(f"\nRecording tokens with {tok_name} tokenizer...")
        
        lang_tokens = {lang: [] for lang in multilang_results.keys()}
        
        for qid in sorted(common_ids):
            for lang in multilang_results.keys():
                if qid in results_by_id[lang]:
                    output = results_by_id[lang][qid]["raw_output"]
                    tok_count = count_tokens(output, tokenizer)
                    lang_tokens[lang].append(tok_count)
        
        # Compute statistics for each language
        token_records[tok_name] = {}
        for lang in multilang_results.keys():
            tokens = lang_tokens[lang]
            avg_tokens = sum(tokens) / len(tokens) if tokens else 0
            
            sorted_tokens = sorted(tokens)
            n = len(tokens)
            median_tokens = sorted_tokens[n // 2] if n % 2 == 1 else (sorted_tokens[n // 2 - 1] + sorted_tokens[n // 2]) / 2
            
            token_records[tok_name][lang] = {
                "n_samples": len(tokens),
                "avg_tokens": avg_tokens,
                "median_tokens": median_tokens,
                "total_tokens": sum(tokens),
                "tokens_list": tokens,
            }
            
            print(f"  {lang.upper()}: {len(tokens)} samples, avg={avg_tokens:.1f}, median={median_tokens:.1f}")
    
    return token_records, common_ids, results_by_id


def save_detailed_results(
    common_ids: list,
    results_by_id: dict,
    tokenizers: dict,
    output_path: str,
):
    """Save detailed per-example token counts to a CSV file."""
    languages = list(results_by_id.keys())
    
    # Build rows
    rows = []
    for qid in sorted(common_ids):
        row = {"id": qid}
        
        # Add character lengths and correctness for each language
        for lang in languages:
            if qid in results_by_id[lang]:
                output = results_by_id[lang][qid]["raw_output"]
                row[f"{lang}_char_len"] = len(output)
                row[f"{lang}_correct"] = results_by_id[lang][qid].get("correct", "")
        
        # Add token counts for each tokenizer and language
        for tok_name, tokenizer in tokenizers.items():
            for lang in languages:
                if qid in results_by_id[lang]:
                    output = results_by_id[lang][qid]["raw_output"]
                    tok_count = count_tokens(output, tokenizer)
                    row[f"{tok_name}_{lang}_tokens"] = tok_count
        
        rows.append(row)
    
    # Write CSV
    ensure_dir(os.path.dirname(output_path))
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\nDetailed results saved to: {output_path}")
    else:
        print(f"\nNo data to save to: {output_path}")


def print_summary_table(token_records: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("TOKEN LENGTH SUMMARY FOR ALL LANGUAGES")
    print("=" * 100)
    
    for tok_name, lang_stats in token_records.items():
        print(f"\n{tok_name.upper()} Tokenizer:")
        print("-" * 100)
        print(f"{'Language':<12} {'Samples':>10} {'Avg Tokens':>12} {'Median Tokens':>15} {'Total Tokens':>15}")
        print("-" * 100)
        
        for lang in sorted(lang_stats.keys()):
            stats = lang_stats[lang]
            print(f"{lang.upper():<12} {stats['n_samples']:>10} {stats['avg_tokens']:>12.1f} "
                  f"{stats['median_tokens']:>15.1f} {stats['total_tokens']:>15}")
    
    print("=" * 100)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record token lengths for multilingual MMATH outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["chatgpt-5.1", "deepseek-v3.2", "gemini-2.5"],
        help="Model name to analyze"
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        choices=list(TOKENIZERS.keys()),
        default=list(TOKENIZERS.keys()),
        help="Tokenizers to use for recording"
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
        "--base-dir",
        type=str,
        default="results/mmath",
        help="Base directory containing MMATH results"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Token Length Recording: Multilingual MMATH Outputs")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Languages: English, Chinese, Spanish, Thai")
    print(f"Tokenizers: {args.tokenizers}")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    print()

    # Load results for all languages
    print("Loading evaluation results...")
    multilang_results = load_multilang_results(args.model, args.base_dir)
    
    if not multilang_results:
        print("Error: No results loaded")
        sys.exit(1)

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

    # Record token lengths
    print("\n" + "=" * 80)
    print("RECORDING TOKEN LENGTHS")
    print("=" * 80)

    token_records, common_ids, results_by_id = record_token_lengths(
        multilang_results=multilang_results,
        tokenizers=tokenizers,
        limit=args.limit,
    )

    # Print summary
    print_summary_table(token_records)

    # Save detailed results
    output_path = os.path.join(
        args.output_dir,
        f"token_records_{args.model}_mmath.csv"
    )
    save_detailed_results(
        common_ids=common_ids,
        results_by_id=results_by_id,
        tokenizers=tokenizers,
        output_path=output_path,
    )

    # Save summary statistics
    summary_path = os.path.join(args.output_dir, f"summary_{args.model}_mmath.txt")
    ensure_dir(args.output_dir)
    with open(summary_path, 'w') as f:
        f.write("Token Length Recording Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Languages: en, zh, es, th\n")
        f.write(f"Base directory: {args.base_dir}\n\n")

        for tok_name, lang_stats in token_records.items():
            f.write(f"\n{tok_name}:\n")
            for lang in sorted(lang_stats.keys()):
                stats = lang_stats[lang]
                f.write(f"  {lang.upper()}:\n")
                f.write(f"    Samples: {stats['n_samples']}\n")
                f.write(f"    Avg tokens: {stats['avg_tokens']:.1f}\n")
                f.write(f"    Median tokens: {stats['median_tokens']:.1f}\n")
                f.write(f"    Total tokens: {stats['total_tokens']}\n")

    print(f"\nSummary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("Recording complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
