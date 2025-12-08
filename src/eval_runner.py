"""
Evaluation runner for math reasoning experiments.
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from .data_utils import load_jsonl, ensure_dir
from .model_interface import BaseModel, GeminiModel
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
    limit: Optional[int] = None,
    use_batch: bool = False,
    batch_poll_interval: int = 30,
    batch_chunk_size: Optional[int] = None,
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
        use_batch: If True and model is GeminiModel, use Batch API (50% cost savings).
        batch_poll_interval: Seconds between batch status checks.
        batch_chunk_size: If set, split batch into chunks for incremental progress.
        
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
    output_lengths = []
    valid_count = 0
    correct_count = 0
    total_output_length = 0
    
    # Check if we should use batch mode (only for GeminiModel)
    can_use_batch = use_batch and isinstance(model, GeminiModel)
    
    if can_use_batch:
        # ==================== BATCH MODE (Gemini only) ====================
        print(f"[Batch Mode] Preparing {len(records)} prompts for Gemini Batch API...")
        
        # Build all prompts first
        prompts = []
        for record in records:
            prompt = build_prompt(
                question=record["question"],
                dataset_name=dataset_name,
                language=language,
                mode=mode,
                prompt_variant=prompt_variant
            )
            prompts.append(prompt)
        
        # Determine chunking
        chunk_size = batch_chunk_size if batch_chunk_size and batch_chunk_size > 0 else len(prompts)
        num_chunks = (len(prompts) + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            print(f"[Batch Mode] Splitting into {num_chunks} chunks of up to {chunk_size} prompts each")
        
        all_raw_outputs = []
        running_correct = 0
        batch_start_time = time.time()
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(prompts))
            chunk_prompts = prompts[start_idx:end_idx]
            chunk_records = records[start_idx:end_idx]
            
            if num_chunks > 1:
                print(f"\n[Batch Mode] === Chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_prompts)} prompts) ===")
            
            try:
                chunk_outputs = model.batch_generate_async(
                    prompts=chunk_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    poll_interval=batch_poll_interval,
                    verbose=True
                )
                all_raw_outputs.extend(chunk_outputs)
                
                # Show running progress after each chunk
                if num_chunks > 1:
                    chunk_correct = 0
                    for rec, out in zip(chunk_records, chunk_outputs):
                        pred = extract_final_answer(out)
                        if is_answer_correct(pred, str(rec["answer"])):
                            chunk_correct += 1
                    running_correct += chunk_correct
                    
                    elapsed = time.time() - batch_start_time
                    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
                    
                    print(f"[Batch Mode] Chunk {chunk_idx + 1} accuracy: {chunk_correct}/{len(chunk_outputs)} "
                          f"({chunk_correct/len(chunk_outputs)*100:.1f}%)")
                    print(f"[Batch Mode] Running accuracy: {running_correct}/{len(all_raw_outputs)} "
                          f"({running_correct/len(all_raw_outputs)*100:.1f}%)")
                    print(f"[Batch Mode] Progress: {len(all_raw_outputs)}/{len(prompts)} | Elapsed: {elapsed_str}")
                    
            except Exception as e:
                print(f"[Batch Mode] Chunk {chunk_idx + 1} failed: {e}")
                print("[Batch Mode] Falling back to sequential for remaining prompts...")
                can_use_batch = False
                break
        
        # Process all batch results
        if can_use_batch and len(all_raw_outputs) == len(records):
            for record, raw_output in zip(records, all_raw_outputs):
                gold_answer = str(record["answer"])
                pred_answer = extract_final_answer(raw_output)
                correct = is_answer_correct(pred_answer, gold_answer)
                
                valid_count += int(pred_answer is not None)
                correct_count += int(correct)
                total_output_length += len(raw_output)
                output_lengths.append(len(raw_output))
                
                results.append({
                    "id": record["id"],
                    "model": model.name,
                    "dataset": dataset_name,
                    "language": language,
                    "mode": mode,
                    "max_tokens": max_tokens,
                    "gold": gold_answer,
                    "pred": pred_answer,
                    "valid": int(pred_answer is not None),
                    "correct": int(correct),
                    "raw_output_length": len(raw_output),
                    "raw_output": raw_output
                })
        else:
            can_use_batch = False  # Fall back to sequential
    
    if not can_use_batch:
        # ==================== SEQUENTIAL MODE (all models) ====================
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

            valid_count += int(pred_answer is not None)
            correct_count += int(correct)
            total_output_length += len(raw_output)
            output_lengths.append(len(raw_output))

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
                "valid": int(pred_answer is not None),
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
    valid_accuracy = correct_count / valid_count if valid_count > 0 else 0.0

    print(f"Valid count: {valid_count}; Correct: {correct_count}; Total: {n}")
    
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
        "valid_count": valid_count,
        "valid_accuracy": valid_accuracy,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "output_lengths": output_lengths
    }
    
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{n})")
    print(f"Valid accuracy: {valid_accuracy:.4f} ({correct_count}/{valid_count})")
    
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
