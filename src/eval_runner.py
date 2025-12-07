"""
Evaluation runner for math reasoning experiments.
"""

import csv
import os
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from .data_utils import load_jsonl, ensure_dir
from .model_interface import BaseModel
from .parsing import extract_final_answer, is_answer_correct


# Prompt templates
PROMPT_TEMPLATES = {
    "gsm8k_en_direct": """You are a careful math tutor. Solve the following problem step by step.
At the end, output ONLY the final numeric answer on a line starting with:
#### <number>

Problem:
{question}""",

    "gsm8k_zh_direct": """你是一名细心的数学老师。请一步一步推理并解答下面这道题。
最后请只在单独一行输出最终的数字答案，格式为：
#### <number>

题目：
{question}""",

    "gsm8k_zh_translate_then_solve": """你是一名中英双语的数学老师。请先在心里把这道中文题翻译成英文，再用英文在心里推理并解答。
不要展示翻译过程，只展示完整的推理过程（可以用中文），
最后在单独一行输出最终的数字答案，格式为：
#### <number>

题目：
{question}""",

    "mmath_en_direct": """You are a careful math tutor. Solve the following problem step by step.
At the end, output ONLY the final numeric answer on a line starting with:
#### <number>

Problem:
{question}""",

    "mmath_zh_direct": """你是一名细心的数学老师。请一步一步推理并解答下面这道题。
最后请只在单独一行输出最终的数字答案，格式为：
#### <number>

题目：
{question}""",
}


def get_prompt_template(
    dataset_name: str,
    language: str,
    mode: str
) -> str:
    """
    Get the appropriate prompt template.
    
    Args:
        dataset_name: "gsm8k" or "mmath"
        language: "en" or "zh"
        mode: "direct" or "zh_translate_then_solve"
        
    Returns:
        Prompt template string with {question} placeholder.
    """
    if mode == "zh_translate_then_solve" and language == "zh":
        key = f"{dataset_name}_zh_translate_then_solve"
    else:
        key = f"{dataset_name}_{language}_direct"
    
    if key not in PROMPT_TEMPLATES:
        # Fallback to English direct
        key = f"{dataset_name}_en_direct"
    
    return PROMPT_TEMPLATES.get(key, PROMPT_TEMPLATES["gsm8k_en_direct"])


def build_prompt(
    question: str,
    dataset_name: str,
    language: str,
    mode: str,
    prompt_variant: Optional[str] = None
) -> str:
    """
    Build the full prompt for a question.
    
    Args:
        question: The math question text.
        dataset_name: "gsm8k" or "mmath"
        language: "en" or "zh"
        mode: "direct" or "zh_translate_then_solve"
        prompt_variant: Optional override for prompt template key.
        
    Returns:
        Complete prompt string.
    """
    if prompt_variant and prompt_variant in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[prompt_variant]
    else:
        template = get_prompt_template(dataset_name, language, mode)
    
    return template.format(question=question)


def run_experiment(
    model: BaseModel,
    dataset_path: str,
    dataset_name: str,
    language: str,
    mode: str,
    max_tokens: int,
    output_csv_path: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 40,
    prompt_variant: Optional[str] = None,
    limit: Optional[int] = None
) -> dict:
    """
    Run an evaluation experiment.
    
    Args:
        model: Model instance to evaluate.
        dataset_path: Path to the JSONL dataset file.
        dataset_name: "gsm8k" or "mmath"
        language: "en" or "zh"
        mode: "direct" or "zh_translate_then_solve"
        max_tokens: Maximum tokens for generation.
        output_csv_path: Path to save results CSV.
        prompt_variant: Optional override for prompt template.
        limit: Optional limit on number of examples to evaluate.
        
    Returns:
        Dictionary with summary statistics.
    """
    # Load dataset
    records = load_jsonl(dataset_path)
    
    # Filter by language if dataset contains multiple languages
    records = [r for r in records if r.get("language", language) == language]
    
    if limit:
        records = records[:limit]
    
    print(f"Running experiment: {model.name} on {dataset_name} ({language}, {mode})")
    print(f"Dataset size: {len(records)} examples")
    
    # Prepare output
    ensure_dir(os.path.dirname(output_csv_path))
    
    results = []
    correct_count = 0
    total_output_length = 0
    
    # Run evaluation
    for record in tqdm(records, desc=f"Evaluating {model.name}"):
        question = record["question"]
        gold_answer = str(record["answer"])

        # Build prompt
        prompt = build_prompt(
            question=question,
            dataset_name=dataset_name,
            language=language,
            mode=mode,
            prompt_variant=prompt_variant
        )

        # Generate response
        try:
            raw_output = model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        except Exception as e:
            print(f"Warning: Generation failed for {record['id']}: {e}")
            raw_output = f"ERROR: {e}"

        # Extract and evaluate answer
        pred_answer = extract_final_answer(raw_output)
        correct = is_answer_correct(pred_answer, gold_answer)

        correct_count += int(correct)
        total_output_length += len(raw_output)

        # Record result
        results.append({
            "id": record["id"],
            "model": model.name,
            "dataset": dataset_name,
            "language": language,
            "mode": mode,
            "max_tokens": max_tokens,
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": int(correct),
            "raw_output_length": len(raw_output),
            "raw_output": raw_output
        })
    
    # Save results to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {output_csv_path}")
    
    # Compute summary
    n = len(results)
    accuracy = correct_count / n if n > 0 else 0.0
    avg_output_length = total_output_length / n if n > 0 else 0.0

    print(f"Correct: {correct_count}; Total: {n}")
    
    summary = {
        "model": model.name,
        "dataset": dataset_name,
        "language": language,
        "mode": mode,
        "max_tokens": max_tokens,
        "n": n,
        "correct": correct_count,
        "accuracy": accuracy,
        "avg_raw_output_length": avg_output_length,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{n})")
    
    return summary


def generate_output_path(
    output_dir: str,
    dataset_name: str,
    model_name: str,
    language: str,
    mode: str,
    max_tokens: int
) -> str:
    """
    Generate a standardized output CSV path.
    
    Args:
        output_dir: Base output directory.
        dataset_name: "gsm8k" or "mmath"
        model_name: Model identifier.
        language: "en" or "zh"
        mode: "direct" or "zh_translate_then_solve"
        max_tokens: Maximum tokens setting.
        
    Returns:
        Full path to output CSV file.
    """
    # Sanitize model name for filename
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    filename = f"{dataset_name}_{safe_model_name}_{language}_{mode}_maxtok{max_tokens}.csv"
    return os.path.join(output_dir, filename)
