#!/usr/bin/env python3
"""
Download raw datasets (GSM8K, MMATH) from HuggingFace or original sources.

Usage:
    python scripts/00_download_data.py
"""

import json
import os
import sys
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import ensure_dir, save_jsonl
from datasets import concatenate_datasets, load_dataset

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
    
    raw_dir = "data/raw/gsm8k"
    ensure_dir(raw_dir)
    
    train_path = os.path.join(raw_dir, "gsm8k_train.jsonl")
    test_path = os.path.join(raw_dir, "gsm8k_test.jsonl")
    
    save_jsonl(train_path, dataset["train"].select(range(10)) if debug else dataset["train"])
    save_jsonl(test_path, dataset["test"].select(range(10)) if debug else dataset["test"])
    
    print(f"  Saved {len(dataset['train'])} train examples to {train_path}")
    print(f"  Saved {len(dataset['test'])} test examples to {test_path}")


def download_mmath():
    """
    Download MMATH dataset.
    """
    print("Downloading MMATH dataset...")
    
    raw_dir = "data/raw/mmath"
    ensure_dir(raw_dir)

    # Download from original source
    content_url = "https://api.github.com/repos/RUCAIBox/MMATH/contents/mmath?ref=main"
    response = requests.get(content_url)
    response.raise_for_status()
    contents = response.json()
    
    # Download each file
    for content in contents:
        file_url = content["download_url"]
        file_response = requests.get(file_url)
        file_response.raise_for_status()
        
        file_name = content["name"]
        file_path = os.path.join(raw_dir, file_name)
        
        with open(file_path, "wb") as f:
            f.write(file_response.content)
        
        print(f"  Saved {file_name} to {file_path}")


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
    print("=" * 60)


if __name__ == "__main__":
    main()
