#!/usr/bin/env python3
"""
Process the GSM8K dataset.

This script:
1. Loads data/raw/gsm8k_train.jsonl and data/raw/gsm8k_test.jsonl
2. Extract the *final* answers from the "answer" keys (which are actually solutions)
3. Translates questions, solutions and answers to Chinese using an LLM
4. Saves both EN and ZH versions in processed format

Usage:
    python scripts/01_process_gsm8k.py
"""

import os
import sys
from datetime import datetime
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_jsonl, save_jsonl, ensure_dir
from src.translate_utils import translate_to_zh, DummyModelClient
from src.model_interface import OpenAIChatModel, GeminiModel


def format_id(index: int) -> str:
    """Format index as zero-padded ID."""
    return f"gsm8k_test_{index:06d}"


def extract_final_answer(answer: str) -> str:
    """
    Extract the final answer from the GSM8K answer format.
    GSM8K answers are in the format "#### 42"

    Args:
        answer: The answer string to extract the final answer from.
    
    Returns:
        The final answer as a string.
    """
    if "####" in str(answer):
        return str(answer).split("####")[-1].strip()
    else:
        raise ValueError(f"Invalid answer format: {answer}")


def create_processed_record_en(
    raw_record: dict,
    index: int,
    split: Literal["train", "test"],
) -> dict:
    """
    Create a processed English record from raw GSM8K data.
    
    Args:
        raw_record: Raw record with 'question' and 'answer' fields.
        index: Record index for ID generation.
        split: Split of the record (e.g., "train", "test").
        
    Returns:
        Processed record in standard format.
    """
    # Extract numeric answer from GSM8K format
    solution = raw_record["answer"]
    answer = extract_final_answer(solution)
    
    return {
        "id": format_id(index),
        "source": "gsm8k",
        "split": split,
        "level": "gsm8k",
        "language": "en",
        "question": raw_record["question"],
        "solution": solution,
        "answer": answer,
        "meta": {
            "original_index": index
        }
    }


def create_processed_record_zh(
    en_record: dict,
    question_zh: str,
    solution_zh: str,
    split: Literal["train", "test"],
    translation_source: Literal["chatgpt-5.1", "deepseek-v2"],
) -> dict:
    """
    Create a processed Chinese record from English record.
    
    Args:
        en_record: Processed English record.
        translation_source: Model used for translation.
        
    Returns:
        Processed Chinese record in standard format.
    """
    return {
        "id": en_record["id"],  # Same ID for pairing
        "source": "gsm8k",
        "split": split,
        "level": "gsm8k",
        "language": "zh",
        "question": question_zh,
        "solution": solution_zh,
        "answer": en_record["answer"],
        "meta": {
            "original_index": en_record["meta"]["original_index"],
            "translation_source": translation_source,
            "translation_time": datetime.now().isoformat() + "Z"
        }
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Translating GSM8K from English to Chinese")
    print("=" * 60)
    print()
    
    # Paths
    raw_path_train = "data/raw/gsm8k/gsm8k_train.jsonl"
    raw_path_test = "data/raw/gsm8k/gsm8k_test.jsonl"
    en_output_path_train = "data/processed/gsm8k/gsm8k_en_train.jsonl"
    en_output_path_test = "data/processed/gsm8k/gsm8k_en_test.jsonl"
    zh_output_path_train = "data/processed/gsm8k/gsm8k_zh_train.jsonl"
    zh_output_path_test = "data/processed/gsm8k/gsm8k_zh_test.jsonl"
    
    # Check if raw data exists
    if not os.path.exists(raw_path_train):
        print(f"Error: Raw data not found at {raw_path_train}")
        print("Please run: python scripts/00_download_data.py first")
        sys.exit(1)
    
    if not os.path.exists(raw_path_test):
        print(f"Error: Raw data not found at {raw_path_test}")
        print("Please run: python scripts/00_download_data.py first")
        sys.exit(1)
    
    # Load raw data
    print(f"Loading raw data from {raw_path_train}...")
    raw_records_train = load_jsonl(raw_path_train)
    print(f"  Loaded {len(raw_records_train)} records")
    
    print(f"Loading raw data from {raw_path_test}...")
    raw_records_test = load_jsonl(raw_path_test)
    print(f"  Loaded {len(raw_records_test)} records")
    
    # Create processed English records
    print("Creating processed English records...")
    en_records_train = [
        create_processed_record_en(r, i, "train")
        for i, r in enumerate(raw_records_train)
    ]
    
    en_records_test = [
        create_processed_record_en(r, i, "test")
        for i, r in enumerate(raw_records_test)
    ]
    
    # Translate to Chinese
    print("Translating questions to Chinese...")
    
    translator = OpenAIChatModel(
        name="translator",
        model_name="gpt-5-mini",
        use_temperature=False
    )
    class TranslatorWrapper:
        def __init__(self, model):
            self.model = model
        def chat(self, prompt):
            return self.model.generate(prompt)
    model_client = TranslatorWrapper(translator)
    
    # Prepare records for translation
    translation_input_train = [
        {"id": r["id"], "question": r["question"], "solution": r["solution"]}
        for r in en_records_train
    ]
    
    translation_input_test = [
        {"id": r["id"], "question": r["question"], "solution": r["solution"]}
        for r in en_records_test
    ]
    
    translated_train, failed_train = translate_to_zh(
        records=translation_input_train,
        model_client=model_client,
        batch_size=50
    )
    
    translated_test, failed_test = translate_to_zh(
        records=translation_input_test,
        model_client=model_client,
        batch_size=50
    )
    
    # Create translation lookup
    translation_lookup_train = {
        t["id"]: {
            "question_zh": t.get("question_zh"),
            "solution_zh": t.get("solution_zh"),
        }
        for t in translated_train
    }
    translation_lookup_test = {
        t["id"]: {
            "question_zh": t.get("question_zh"),
            "solution_zh": t.get("solution_zh"),
        }
        for t in translated_test
    }
    
    # Create processed Chinese records
    print("Creating processed Chinese records...")
    zh_records_train = [
        create_processed_record_zh(
            en_record=r,
            question_zh=translation_lookup_train.get(r["id"]).get("question_zh"),
            solution_zh=translation_lookup_train.get(r["id"]).get("solution_zh"),
            split="train",
            translation_source="gpt-5.1"
        )
        for r in en_records_train
    ]
    
    zh_records_test = [
        create_processed_record_zh(
            en_record=r,
            question_zh=translation_lookup_test.get(r["id"]).get("question_zh"),
            solution_zh=translation_lookup_test.get(r["id"]).get("solution_zh"),
            split="test",
            translation_source="gpt-5.1"
        )
        for r in en_records_test
    ]
    
    # Save processed files
    ensure_dir("data/processed")
    
    save_jsonl(en_output_path_train, en_records_train)
    print(f"  Saved {len(en_records_train)} English records to {en_output_path_train}")
    
    save_jsonl(en_output_path_test, en_records_test)
    print(f"  Saved {len(en_records_test)} English records to {en_output_path_test}")
    
    save_jsonl(zh_output_path_train, zh_records_train)
    print(f"  Saved {len(zh_records_train)} Chinese records to {zh_output_path_train}")
    print(f"Failed translations ids: {failed_train}")
    
    save_jsonl(zh_output_path_test, zh_records_test)
    print(f"  Saved {len(zh_records_test)} Chinese records to {zh_output_path_test}")
    print(f"Failed translations ids: {failed_test}")
    
    print()
    print("=" * 60)
    print("Translation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
