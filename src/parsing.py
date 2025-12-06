"""
Parsing utilities for extracting and comparing answers.
"""

import re
from typing import Optional


def extract_final_answer(output: str) -> Optional[str]:
    """
    Extract the final answer from model output.
    
    Looks for a line starting with '#### ' and returns the rest stripped.
    
    Args:
        output: Raw model output string.
        
    Returns:
        Extracted answer string, or None if not found.
    """
    if not output:
        return None
    
    # Look for "Final Answer:" pattern (case insensitive)
    pattern = r"(?i)^####\s*(.+)"
    
    for line in output.split('\n'):
        match = re.search(pattern, line.strip())
        if match:
            answer = match.group(1).strip()
            # Clean up common formatting issues
            answer = answer.rstrip('.')
            answer = answer.strip('$')
            answer = answer.strip()
            return answer
    
    return None


def extract_final_answer_batch(outputs: list[str]) -> list[Optional[str]]:
    """
    Extract the final answer from a batch of model outputs.
    
    Args:
        outputs: List of raw model output strings.
        
    Returns:
        List of extracted answer strings, or None if not found.
    """
    return [extract_final_answer(output) for output in outputs]


def normalize_number(s: str) -> Optional[float]:
    """
    Try to parse a string as a number.
    
    Handles common formats like:
    - "42"
    - "3.14"
    - "-7.5"
    - "1,234.56" (with commas)
    - "1/2" (simple fractions)
    
    Args:
        s: String to parse.
        
    Returns:
        Float value, or None if parsing fails.
    """
    if not s:
        return None
    
    s = s.strip()
    
    # Remove commas (thousand separators)
    s = s.replace(',', '')
    
    # Try direct float parsing
    try:
        return float(s)
    except ValueError:
        pass
    
    # Try fraction parsing (e.g., "1/2", "3/4")
    fraction_pattern = r"^(-?\d+)\s*/\s*(\d+)$"
    match = re.match(fraction_pattern, s)
    if match:
        try:
            num = float(match.group(1))
            denom = float(match.group(2))
            if denom != 0:
                return num / denom
        except ValueError:
            pass
    
    # Try mixed number (e.g., "1 1/2")
    mixed_pattern = r"^(-?\d+)\s+(\d+)\s*/\s*(\d+)$"
    match = re.match(mixed_pattern, s)
    if match:
        try:
            whole = float(match.group(1))
            num = float(match.group(2))
            denom = float(match.group(3))
            if denom != 0:
                sign = -1 if whole < 0 else 1
                return whole + sign * (num / denom)
        except ValueError:
            pass
    
    return None


def is_answer_correct(
    pred: Optional[str],
    gold: str,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if predicted answer matches gold answer.
    
    Strategy:
    1. Try to parse both as floats; if equal within tolerance, return True.
    2. Otherwise compare as stripped lowercase strings.
    
    Args:
        pred: Predicted answer (may be None).
        gold: Gold/correct answer.
        tolerance: Numerical tolerance for float comparison.
        
    Returns:
        True if answers match, False otherwise.
    """
    if pred is None:
        return False
    
    pred = pred.strip()
    gold = gold.strip()
    
    # Try numeric comparison first
    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)
    
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < tolerance
    
    # Fall back to string comparison (case-insensitive)
    return pred.lower() == gold.lower()


def is_answer_correct_batch(
    preds: list[Optional[str]],
    golds: list[str],
    tolerance: float = 1e-6
) -> list[bool]:
    """
    Check if predicted answers match gold answers for a batch.
    
    Args:
        preds: List of predicted answers (may contain None).
        golds: List of gold/correct answers.
        tolerance: Numerical tolerance for float comparison.
        
    Returns:
        List of boolean values indicating correct/incorrect for each example.
    """
    return [is_answer_correct(pred, gold, tolerance) for pred, gold in zip(preds, golds)]


def extract_boxed_answer(output: str) -> Optional[str]:
    """
    Extract answer from LaTeX \\boxed{} format.
    
    Some models output answers in LaTeX format like \\boxed{42}.
    
    Args:
        output: Raw model output string.
        
    Returns:
        Extracted answer string, or None if not found.
    """
    if not output:
        return None
    
    # Match \boxed{...} pattern
    pattern = r"\\boxed\{([^}]+)\}"
    matches = re.findall(pattern, output)
    
    if matches:
        # Return the last boxed answer (usually the final answer)
        return matches[-1].strip()
    
    return None


def extract_answer_flexible(output: str) -> Optional[str]:
    """
    Try multiple extraction strategies and return the first match.
    
    Order of attempts:
    1. Final Answer: format
    2. \\boxed{} format
    3. Last number in the output
    
    Args:
        output: Raw model output string.
        
    Returns:
        Extracted answer string, or None if not found.
    """
    # Try Final Answer format
    answer = extract_final_answer(output)
    if answer:
        return answer
    
    # Try boxed format
    answer = extract_boxed_answer(output)
    if answer:
        return answer
    
    # Try to find the last standalone number
    # This is a fallback and may not be reliable
    number_pattern = r"(?:^|[^\d])(-?\d+(?:,\d{3})*(?:\.\d+)?|\d+/\d+)(?:[^\d]|$)"
    matches = re.findall(number_pattern, output)
    if matches:
        return matches[-1].strip()
    
    return None
