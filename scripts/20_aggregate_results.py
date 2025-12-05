#!/usr/bin/env python3
"""
Aggregate results from all evaluation runs.

Reads CSVs from results/gsm8k/ and results/mmath/,
computes summary statistics, and writes results/summary.csv.

Usage:
    python scripts/20_aggregate_results.py
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import ensure_dir


def find_result_csvs(base_dir: str) -> list[str]:
    """
    Find all CSV files in a directory and subdirectories.
    
    Args:
        base_dir: Base directory to search.
        
    Returns:
        List of paths to CSV files.
    """
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return csv_files


def load_csv(path: str) -> list[dict[str, Any]]:
    """
    Load a CSV file as a list of dictionaries.
    
    Args:
        path: Path to CSV file.
        
    Returns:
        List of row dictionaries.
    """
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            if 'correct' in row:
                row['correct'] = int(row['correct'])
            if 'max_tokens' in row:
                row['max_tokens'] = int(row['max_tokens'])
            if 'raw_output_length' in row:
                row['raw_output_length'] = int(row['raw_output_length'])
            rows.append(row)
    return rows


def aggregate_results(result_dirs: list[str]) -> list[dict[str, Any]]:
    """
    Aggregate results from multiple directories.
    
    Args:
        result_dirs: List of directories containing result CSVs.
        
    Returns:
        List of aggregated summary rows.
    """
    # Find all CSV files
    all_csvs = []
    for d in result_dirs:
        if os.path.exists(d):
            all_csvs.extend(find_result_csvs(d))
    
    if not all_csvs:
        print("No result CSV files found.")
        return []
    
    print(f"Found {len(all_csvs)} result files")
    
    # Load all results
    all_rows = []
    for csv_path in all_csvs:
        try:
            rows = load_csv(csv_path)
            all_rows.extend(rows)
            print(f"  Loaded {len(rows)} rows from {csv_path}")
        except Exception as e:
            print(f"  Error loading {csv_path}: {e}")
    
    if not all_rows:
        print("No result rows found.")
        return []
    
    # Group by (model, dataset, language, mode, max_tokens)
    groups = defaultdict(list)
    for row in all_rows:
        key = (
            row.get('model', 'unknown'),
            row.get('dataset', 'unknown'),
            row.get('language', 'unknown'),
            row.get('mode', 'direct'),
            row.get('max_tokens', 0)
        )
        groups[key].append(row)
    
    # Compute aggregates
    summaries = []
    for key, rows in groups.items():
        model, dataset, language, mode, max_tokens = key
        
        n = len(rows)
        correct_count = sum(r.get('correct', 0) for r in rows)
        accuracy = correct_count / n if n > 0 else 0.0
        
        total_output_length = sum(r.get('raw_output_length', 0) for r in rows)
        avg_output_length = total_output_length / n if n > 0 else 0.0
        
        summaries.append({
            'model': model,
            'dataset': dataset,
            'language': language,
            'mode': mode,
            'max_tokens': max_tokens,
            'n': n,
            'correct': correct_count,
            'accuracy': round(accuracy, 4),
            'avg_raw_output_length': round(avg_output_length, 2)
        })
    
    # Sort by dataset, model, language
    summaries.sort(key=lambda x: (x['dataset'], x['model'], x['language'], x['mode']))
    
    return summaries


def save_summary(summaries: list[dict], output_path: str):
    """
    Save summary to CSV.
    
    Args:
        summaries: List of summary dictionaries.
        output_path: Path to output CSV.
    """
    if not summaries:
        print("No summaries to save.")
        return
    
    ensure_dir(os.path.dirname(output_path))
    
    fieldnames = ['model', 'dataset', 'language', 'mode', 'max_tokens',
                  'n', 'correct', 'accuracy', 'avg_raw_output_length']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    
    print(f"Summary saved to: {output_path}")


def print_summary_table(summaries: list[dict]):
    """
    Print a formatted summary table.
    
    Args:
        summaries: List of summary dictionaries.
    """
    if not summaries:
        return
    
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    
    # Header
    header = f"{'Model':<20} {'Dataset':<10} {'Lang':<5} {'Mode':<25} {'MaxTok':<8} {'N':<6} {'Acc':<8} {'AvgLen':<10}"
    print(header)
    print("-" * 90)
    
    # Rows
    current_dataset = None
    for s in summaries:
        if s['dataset'] != current_dataset:
            if current_dataset is not None:
                print("-" * 90)
            current_dataset = s['dataset']
        
        row = (f"{s['model']:<20} {s['dataset']:<10} {s['language']:<5} "
               f"{s['mode']:<25} {s['max_tokens']:<8} {s['n']:<6} "
               f"{s['accuracy']:<8.4f} {s['avg_raw_output_length']:<10.1f}")
        print(row)
    
    print("=" * 90)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Aggregating Evaluation Results")
    print("=" * 60)
    print()
    
    # Directories to search
    result_dirs = [
        "results/gsm8k",
        "results/mmath"
    ]
    
    # Aggregate results
    summaries = aggregate_results(result_dirs)
    
    if summaries:
        # Save summary CSV
        output_path = "results/summary.csv"
        save_summary(summaries, output_path)
        
        # Print table
        print_summary_table(summaries)
        
        # Print quick stats
        print("\nQuick Stats:")
        datasets = set(s['dataset'] for s in summaries)
        for dataset in sorted(datasets):
            dataset_summaries = [s for s in summaries if s['dataset'] == dataset]
            models = set(s['model'] for s in dataset_summaries)
            print(f"  {dataset}: {len(dataset_summaries)} experiments across {len(models)} models")
    else:
        print("No results to aggregate.")
        print("Run evaluation scripts first:")
        print("  python scripts/10_eval_gsm8k_local.py")
        print("  python scripts/11_eval_mmath_api.py")
    
    print("\n" + "=" * 60)
    print("Aggregation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
