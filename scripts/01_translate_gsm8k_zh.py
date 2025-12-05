#!/usr/bin/env python3
"""
Translate GSM8K from English to Chinese.

This script:
1. Loads data/raw/gsm8k_test.jsonl
2. Translates questions to Chinese using an LLM
3. Saves both EN and ZH versions in processed format

Usage:
    python scripts/01_translate_gsm8k_zh.py
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_jsonl, save_jsonl, ensure_dir
from src.translate_utils import translate_questions_to_zh, DummyModelClient


def format_id(index: int) -> str:
    """Format index as zero-padded ID."""
    return f"gsm8k_test_{index:06d}"


def create_processed_record_en(raw_record: dict, index: int) -> dict:
    """
    Create a processed English record from raw GSM8K data.
    
    Args:
        raw_record: Raw record with 'question' and 'answer' fields.
        index: Record index for ID generation.
        
    Returns:
        Processed record in standard format.
    """
    # Extract numeric answer from GSM8K format
    # GSM8K answers are sometimes in format "#### 42"
    answer = raw_record.get("answer", "")
    if "####" in str(answer):
        answer = str(answer).split("####")[-1].strip()
    else:
        answer = str(answer).strip()
    
    return {
        "id": format_id(index),
        "source": "gsm8k",
        "split": "test",
        "level": "gsm8k",
        "language": "en",
        "question": raw_record["question"],
        "answer": answer,
        "answer_type": "float",
        "meta": {
            "original_index": index
        }
    }


def create_processed_record_zh(
    en_record: dict,
    question_zh: str,
    translation_source: str = "chatgpt-5.1"
) -> dict:
    """
    Create a processed Chinese record from English record and translation.
    
    Args:
        en_record: Processed English record.
        question_zh: Translated Chinese question.
        translation_source: Model used for translation.
        
    Returns:
        Processed Chinese record in standard format.
    """
    return {
        "id": en_record["id"],  # Same ID for pairing
        "source": "gsm8k",
        "split": "test",
        "level": "gsm8k",
        "language": "zh",
        "question": question_zh,
        "answer": en_record["answer"],
        "answer_type": "float",
        "meta": {
            "original_index": en_record["meta"]["original_index"],
            "translation_source": translation_source,
            "translation_time": datetime.utcnow().isoformat() + "Z"
        }
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Translating GSM8K from English to Chinese")
    print("=" * 60)
    print()
    
    # Paths
    raw_path = "data/raw/gsm8k_test.jsonl"
    en_output_path = "data/processed/gsm8k_en_test.jsonl"
    zh_output_path = "data/processed/gsm8k_zh_test.jsonl"
    
    # Check if raw data exists
    if not os.path.exists(raw_path):
        print(f"Error: Raw data not found at {raw_path}")
        print("Please run: python scripts/00_download_data.py first")
        sys.exit(1)
    
    # Load raw data
    print(f"Loading raw data from {raw_path}...")
    raw_records = load_jsonl(raw_path)
    print(f"  Loaded {len(raw_records)} records")
    
    # Create processed English records
    print("Creating processed English records...")
    en_records = [
        create_processed_record_en(r, i)
        for i, r in enumerate(raw_records)
    ]
    
    # Translate to Chinese
    print("Translating questions to Chinese...")
    
    # TODO: Replace DummyModelClient with actual API client
    # Example with OpenAI:
    # from src.model_interface import OpenAIChatModel
    # translator = OpenAIChatModel(name="translator", model_name="gpt-4")
    # class TranslatorWrapper:
    #     def __init__(self, model):
    #         self.model = model
    #     def chat(self, prompt):
    #         return self.model.generate(prompt, max_tokens=2000)
    # model_client = TranslatorWrapper(translator)
    
    model_client = DummyModelClient()
    
    # Prepare records for translation
    translation_input = [
        {"id": r["id"], "question": r["question"]}
        for r in en_records
    ]
    
    translated = translate_questions_to_zh(
        records=translation_input,
        model_client=model_client,
        batch_size=10
    )
    
    # Create translation lookup
    translation_lookup = {t["id"]: t.get("question_zh", t["question"]) for t in translated}
    
    # Create processed Chinese records
    print("Creating processed Chinese records...")
    zh_records = [
        create_processed_record_zh(
            en_record=r,
            question_zh=translation_lookup.get(r["id"], r["question"]),
            translation_source="dummy"  # TODO: Update when using real API
        )
        for r in en_records
    ]
    
    # Save processed files
    ensure_dir("data/processed")
    
    save_jsonl(en_output_path, en_records)
    print(f"  Saved {len(en_records)} English records to {en_output_path}")
    
    save_jsonl(zh_output_path, zh_records)
    print(f"  Saved {len(zh_records)} Chinese records to {zh_output_path}")
    
    print()
    print("=" * 60)
    print("Translation complete!")
    print()
    print("NOTE: Currently using DummyModelClient (copies EN to ZH).")
    print("TODO: Replace with actual API client for real translations.")
    print()
    print("Next steps:")
    print("  1. Configure OPENAI_API_KEY environment variable")
    print("  2. Update this script to use OpenAIChatModel")
    print("  3. Re-run to generate real translations")
    print("=" * 60)


if __name__ == "__main__":
    main()
