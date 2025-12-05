#!/usr/bin/env python3
"""
Download raw datasets (GSM8K, MMATH) from HuggingFace or original sources.

Usage:
    python scripts/00_download_data.py
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import ensure_dir, save_jsonl


def download_gsm8k():
    """
    Download GSM8K dataset from HuggingFace.
    
    TODO: Implement actual download. For now, creates placeholder files.
    """
    print("Downloading GSM8K dataset...")
    
    # TODO: Use HuggingFace datasets library
    # from datasets import load_dataset
    # dataset = load_dataset("openai/gsm8k", "main")
    
    raw_dir = "data/raw"
    ensure_dir(raw_dir)
    
    # Placeholder: Create sample data files
    # In production, replace with actual download
    
    sample_train = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "72"
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "10"
        }
    ]
    
    sample_test = [
        {
            "question": "Jesse and Mia are competing in a week long race. They have one week to run 30 miles. On the first three days Jesse averages (2/3) of a mile. On day four she runs 10 miles. Mia averages 3 miles a day over the first 4 days. What is the average of their declines on day 4 from their daily averages over the first three days?",
            "answer": "4.5"
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "answer": "3"
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "answer": "70000"
        }
    ]
    
    # Save placeholder files
    train_path = os.path.join(raw_dir, "gsm8k_train.jsonl")
    test_path = os.path.join(raw_dir, "gsm8k_test.jsonl")
    
    save_jsonl(train_path, sample_train)
    save_jsonl(test_path, sample_test)
    
    print(f"  Saved {len(sample_train)} train examples to {train_path}")
    print(f"  Saved {len(sample_test)} test examples to {test_path}")
    print("  NOTE: These are placeholder samples. Replace with full dataset download.")


def download_mmath():
    """
    Download MMATH dataset.
    
    TODO: Implement actual download from the MMATH source.
    """
    print("Downloading MMATH dataset...")
    
    raw_dir = "data/raw"
    ensure_dir(raw_dir)
    
    # Placeholder: Create sample data file
    sample_mmath = [
        {
            "problem_id": "000045",
            "question_en": "Let $f(x) = x^2 + bx + c$ where $b$ and $c$ are integers. If $f(f(1)) = f(f(2)) = 0$, find the value of $f(0)$.",
            "question_zh": "设 $f(x) = x^2 + bx + c$，其中 $b$ 和 $c$ 是整数。若 $f(f(1)) = f(f(2)) = 0$，求 $f(0)$ 的值。",
            "answer": "17",
            "subdomain": "algebra",
            "difficulty": "high"
        },
        {
            "problem_id": "000123",
            "question_en": "In triangle ABC, angle A = 60 degrees, and the sides opposite to angles A, B, C are a, b, c respectively. If a = 7 and b + c = 13, find the area of triangle ABC.",
            "question_zh": "在三角形ABC中，角A = 60度，角A、B、C的对边分别为a、b、c。若 a = 7 且 b + c = 13，求三角形ABC的面积。",
            "answer": "10√3",
            "subdomain": "geometry",
            "difficulty": "high"
        },
        {
            "problem_id": "000256",
            "question_en": "Find the number of positive integers n less than 1000 such that n^2 + 1 is divisible by 101.",
            "question_zh": "求小于1000的正整数n的个数，使得 n^2 + 1 能被101整除。",
            "answer": "20",
            "subdomain": "number_theory",
            "difficulty": "high"
        }
    ]
    
    mmath_path = os.path.join(raw_dir, "mmath_raw.jsonl")
    save_jsonl(mmath_path, sample_mmath)
    
    print(f"  Saved {len(sample_mmath)} examples to {mmath_path}")
    print("  NOTE: These are placeholder samples. Replace with full dataset download.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Downloading datasets for multilingual math evaluation")
    print("=" * 60)
    print()
    
    download_gsm8k()
    print()
    
    download_mmath()
    print()
    
    print("=" * 60)
    print("Download complete!")
    print()
    print("Next steps:")
    print("  1. Replace placeholder data with full datasets")
    print("  2. Run: python scripts/01_translate_gsm8k_zh.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
