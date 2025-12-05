#!/usr/bin/env python3
"""
Process MMATH raw data into standardized multilingual format.

This script:
1. Loads data/raw/mmath_raw.jsonl
2. Converts to standardized format with separate language entries
3. Saves to data/processed/mmath_multilingual.jsonl

Usage:
    python scripts/02_process_mmath.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_jsonl, save_jsonl, ensure_dir


def process_mmath_record(raw_record: dict) -> list[dict]:
    """
    Convert a raw MMATH record to standardized format.
    
    Raw format has question_en, question_zh, etc.
    Output format has separate records per language.
    
    Args:
        raw_record: Raw MMATH record.
        
    Returns:
        List of processed records (one per language).
    """
    problem_id = raw_record.get("problem_id", "unknown")
    answer = str(raw_record.get("answer", ""))
    subdomain = raw_record.get("subdomain", "general")
    difficulty = raw_record.get("difficulty", "unknown")
    
    # Determine answer type
    answer_type = "int"
    try:
        float(answer)
        if "." in answer:
            answer_type = "float"
    except ValueError:
        answer_type = "expression"  # For symbolic answers like "10√3"
    
    processed = []
    
    # Process each language
    language_fields = {
        "en": "question_en",
        "zh": "question_zh",
        "de": "question_de",
        "fr": "question_fr",
        "es": "question_es",
        "ru": "question_ru",
        "ja": "question_ja",
    }
    
    for lang, field in language_fields.items():
        if field in raw_record and raw_record[field]:
            processed.append({
                "id": f"mmath_{problem_id}_{lang}",
                "problem_id": problem_id,
                "source": "mmath",
                "level": "hard",
                "language": lang,
                "question": raw_record[field],
                "answer": answer,
                "answer_type": answer_type,
                "meta": {
                    "subdomain": subdomain,
                    "difficulty": difficulty
                }
            })
    
    return processed


def main():
    """Main entry point."""
    print("=" * 60)
    print("Processing MMATH dataset")
    print("=" * 60)
    print()
    
    # Paths
    raw_path = "data/raw/mmath_raw.jsonl"
    output_path = "data/processed/mmath_multilingual.jsonl"
    
    # Check if raw data exists
    if not os.path.exists(raw_path):
        print(f"Error: Raw data not found at {raw_path}")
        print("Please run: python scripts/00_download_data.py first")
        sys.exit(1)
    
    # Load raw data
    print(f"Loading raw data from {raw_path}...")
    raw_records = load_jsonl(raw_path)
    print(f"  Loaded {len(raw_records)} raw records")
    
    # Process records
    print("Processing records...")
    processed_records = []
    for raw_record in raw_records:
        processed_records.extend(process_mmath_record(raw_record))
    
    # Count by language
    lang_counts = {}
    for r in processed_records:
        lang = r["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print(f"  Created {len(processed_records)} processed records")
    print("  By language:")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
    
    # Save processed file
    ensure_dir("data/processed")
    save_jsonl(output_path, processed_records)
    print(f"  Saved to {output_path}")
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
