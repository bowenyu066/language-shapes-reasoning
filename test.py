import re
from typing import Optional
import pandas as pd

from src.parsing import is_answer_correct
from src.data_utils import load_jsonl

# PATH = "results/mmath/mmath_gemini-2.5_en_direct_maxtok8192.csv"

# df = pd.read_csv(PATH)

# def extract_answer(text: str) -> Optional[str]:
#     """
#     Extract answer from text in one of two formats:
#     1. "#### <answer>" (always starting from a new line)
#     2. "$\\boxed{<answer>}$" (could be in some sentence)
    
#     If both forms are found, returns the answer from "#### <answer>".
#     If none is found, returns None.
#     """
#     # Try to find "#### <answer>" pattern (starting from a new line)
#     if not text:
#         return None
#     text = str(text)
#     hash_pattern = r'(?:^|\n)####\s*(.+?)(?:\n|$)'
#     hash_match = re.search(hash_pattern, text)
    
#     if hash_match:
#         return hash_match.group(1).strip()
    
#     # Try to find "$\boxed{<answer>}$" pattern
#     boxed_pattern = r'\$\\boxed\{(.+?)\}\$'
#     boxed_match = re.search(boxed_pattern, text)
    
#     if boxed_match:
#         return boxed_match.group(1).strip()
    
#     return None

# raw_outputs, ids = df["raw_output"], df["id"]
# answers = raw_outputs.apply(extract_answer)

# dataset = load_jsonl("data/processed/mmath/mmath_en.jsonl")

# valid_count = answers.notnull().sum()
# correct_count = 0
# for i, (id_, pred_answer) in enumerate(zip(ids, answers)):
#     record = dataset[i]
#     assert record["id"] == id_, f"ID mismatch at index {i}: {record['id']} != {id_}"
#     gold_answer = record.get("answer")
#     is_correct = is_answer_correct(pred_answer, gold_answer)
#     correct_count += int(is_correct)
    
# valid_accuracy = correct_count / valid_count if valid_count > 0 else 0.0
# total_accuracy = correct_count / len(dataset)


# print(f"Valid counts: {valid_count}/{len(dataset)}")
# print(f"Valid accuracy: {valid_accuracy:.4f}")
# print(f"Total accuracy: {total_accuracy:.4f}")

PATH = "results/gsm8k/gsm8k_qwen3-8b_en_direct_maxtok128.csv"
df = pd.read_csv(PATH)

raw_outputs = df["raw_output"]

# want to count if there's any chinese characters in the raw output
chinese_count = 0
for raw_output in raw_outputs:
    if re.search(r'[一-鿿]', raw_output):
        chinese_count += 1

print(f"Chinese count: {chinese_count}")
