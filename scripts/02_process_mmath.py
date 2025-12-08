#!/usr/bin/env python3
"""
Process MMATH raw data into standardized multilingual format.

This script:
1. Loads data/raw/mmath/{language}.json
2. Converts to standardized format with separate language entries
3. Saves to data/processed/mmath/mmath_{language}.jsonl

Usage:
    python scripts/02_process_mmath.py
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import save_jsonl, ensure_dir


# Languages available in MMATH dataset
MMATH_LANGUAGES = ["ar", "en", "es", "fr", "ja", "ko", "pt", "th", "vi", "zh"]


def load_json(path: str) -> list[dict]:
    """Load a JSON array file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_answer_type(answer: str) -> str:
    """Determine the type of answer (int, float, or expression)."""
    try:
        float(answer)
        if "." in answer:
            return "float"
        return "int"
    except ValueError:
        return "expression"


def process_mmath_record(raw_record: dict) -> dict:
    """
    Convert a raw MMATH record to standardized format.
    
    Raw format (per-language JSON file):
        - question: The question text
        - answer: The answer
        - data_source: Source dataset (e.g., "AIME2024")
        - data_source_id: ID within that source
        - lang: Language code
        - gid: Global ID
    
    Output format:
        - id: Unique identifier (mmath_{gid}_{lang})
        - gid: Global problem ID (for cross-language matching)
        - source: "mmath"
        - language: Language code
        - question: Question text
        - answer: Answer string
        - answer_type: int/float/expression
        - meta: Additional metadata
    
    Args:
        raw_record: Raw MMATH record from JSON file.
        
    Returns:
        Processed record in standardized format.
    """
    gid = raw_record.get("gid", 0)
    lang = raw_record.get("lang", "unknown")
    answer = str(raw_record.get("answer", ""))
    
    return {
        "id": f"mmath_{gid}_{lang}",
        "gid": gid,
        "source": "mmath",
        "language": lang,
        "question": raw_record.get("question", ""),
        "answer": answer,
        "answer_type": get_answer_type(answer),
        "meta": {
            "data_source": raw_record.get("data_source", ""),
            "data_source_id": raw_record.get("data_source_id", 0)
        }
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Processing MMATH dataset")
    print("=" * 60)
    print()
    
    raw_dir = "data/raw/mmath"
    output_dir = "data/processed/mmath"
    
    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory not found at {raw_dir}")
        print("Please run: python scripts/00_download_data.py first")
        sys.exit(1)
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    total_records = 0
    lang_counts = {}
    
    # Process each language file
    for lang in MMATH_LANGUAGES:
        raw_path = os.path.join(raw_dir, f"{lang}.json")
        output_path = os.path.join(output_dir, f"mmath_{lang}.jsonl")
        
        if not os.path.exists(raw_path):
            print(f"  Skipping {lang}: {raw_path} not found")
            continue
        
        # Load raw data (JSON array)
        print(f"Processing {lang}...")
        raw_records = load_json(raw_path)
        print(f"  Loaded {len(raw_records)} raw records from {raw_path}")
        
        # Process records
        processed_records = [process_mmath_record(r) for r in raw_records]
        
        # Save to JSONL
        save_jsonl(output_path, processed_records)
        print(f"  Saved {len(processed_records)} records to {output_path}")
        
        lang_counts[lang] = len(processed_records)
        total_records += len(processed_records)
    
    # Summary
    print()
    print("-" * 60)
    print("Summary:")
    print(f"  Total records processed: {total_records}")
    print("  By language:")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
