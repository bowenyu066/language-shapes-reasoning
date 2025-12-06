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
from datasets import load_dataset

def download_gsm8k(debug: bool = False):
    """
    Download GSM8K dataset from HuggingFace.

    * Notes for debugging: `dataset` contains two splits, "train" and "test".
    Each split is a list of dictionaries, where each dictionary contains the
    keys "question" and "answer". The "answer" is in the format "....<solution>\n#### <answer>".
    """
    print("Downloading GSM8K dataset...")
    
    # Use HuggingFace datasets library
    dataset = load_dataset("openai/gsm8k", "main")
    
    raw_dir = "data/raw"
    ensure_dir(raw_dir)
    
    train_path = os.path.join(raw_dir, "gsm8k_train.jsonl")
    test_path = os.path.join(raw_dir, "gsm8k_test.jsonl")
    
    save_jsonl(train_path, dataset["train"].select(range(10)) if debug else dataset["train"])
    save_jsonl(test_path, dataset["test"].select(range(10)) if debug else dataset["test"])
    
    print(f"  Saved {len(dataset['train'])} train examples to {train_path}")
    print(f"  Saved {len(dataset['test'])} test examples to {test_path}")


def download_mmath(debug: bool = False):
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
    
    download_gsm8k(debug=True)
    print()
    
    # download_mmath()
    # print()
    
    print("=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
