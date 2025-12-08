"""
Data utilities for loading and saving JSONL files.
"""

import json
import os
from pathlib import Path
from typing import *


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """
    Load a JSONL file and return a list of dictionaries.
    
    Args:
        path: Path to the JSONL file.
        
    Returns:
        List of dictionaries, one per line.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """
    Save a list of dictionaries to a JSONL file.
    
    Args:
        path: Path to the output JSONL file.
        rows: List of dictionaries to save.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory.
    """
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def filter_by_language(records: list[dict], language: str) -> list[dict]:
    """
    Filter records by language.
    
    Args:
        records: List of record dictionaries.
        language: Language code (e.g., "en", "zh").
        
    Returns:
        Filtered list of records.
    """
    return [r for r in records if r.get("language") == language]


def group_by_problem_id(records: list[dict]) -> dict[str, list[dict]]:
    """
    Group records by problem_id (useful for MMATH multilingual pairing).
    
    Args:
        records: List of record dictionaries.
        
    Returns:
        Dictionary mapping problem_id to list of records.
    """
    grouped = {}
    for r in records:
        pid = r.get("problem_id", r.get("id"))
        if pid not in grouped:
            grouped[pid] = []
        grouped[pid].append(r)
    return grouped


def load_gsm8k_sft_data(language: str, limit: int = None) -> List[Dict]:
    """
    Load processed GSM8K training data for SFT.
    
    Args:
        language: "en" or "zh"
        limit: Optional limit on number of samples
        
    Returns:
        List of training examples
    """
    data_path = f"data/processed/gsm8k/gsm8k_{language}_train.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data not found at {data_path}. "
            "Please run: python scripts/01_process_gsm8k.py first"
        )
    
    records = load_jsonl(data_path)
    if limit:
        records = records[:limit]
    
    print(f"Loaded {len(records)} {language.upper()} training samples")
    return records