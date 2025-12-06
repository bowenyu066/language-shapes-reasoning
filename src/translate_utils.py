"""
Translation utilities for converting English math problems to Chinese.
"""

import json
from typing import Any, Protocol, List, Dict, Tuple
from tqdm import tqdm


class ModelClient(Protocol):
    """Protocol for model clients that can generate text."""
    def chat(self, prompt: str) -> str:
        """Send a prompt and get a response."""
        ...


class DummyModelClient:
    """
    Dummy model client for testing.
    TODO: Replace with actual API client (OpenAI, etc.)
    """
    
    def chat(self, prompt: str) -> str:
        """
        Dummy implementation that just returns a placeholder.
        In production, this should call the actual API.
        """
        # TODO: Implement actual API call
        # For now, we'll try to parse the input and return mock translations
        try:
            # Extract the JSON array from the prompt
            start = prompt.find('[')
            end = prompt.rfind(']') + 1
            if start != -1 and end > start:
                input_data = json.loads(prompt[start:end])
                # Mock: just copy the question as-is with a note
                output = []
                for item in input_data:
                    output.append({
                        "id": item["id"],
                        "question_zh": item["question"]  # TODO: Replace with real translation
                    })
                return json.dumps(output, ensure_ascii=False)
        except Exception:
            pass
        return "[]"


TRANSLATION_PROMPT_TEMPLATE = """You are a professional translator proficient 
in English and Simplified Chinese.
Translate each math problem and its corresponding solutions from English to 
natural, fluent Simplified Chinese, preserving all numbers, conditions, and 
structure exactly.
Do NOT add explanations or hints. Do NOT solve the problems in advance.
Input is a JSON array containing batches of problems and solutions, each with 
fields `id`, `question`, and `solution`.
Output should also be a JSON array containing the translated problems and 
solutions, each with the same `id` and new fields `question_zh` and 
`solution_zh`.
Input:
{input_json}

Output (JSON only):
"""


def translate_to_zh(
    records: List[Dict[str, Any]],
    model_client: ModelClient,
    batch_size: int = 50,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Translate questions and solutions from English to Chinese using an LLM.
    
    Args:
        records: List of records with "id", "question", and "solution" fields.
        model_client: Object with a `chat(prompt: str) -> str` method.
        batch_size: Number of questions to translate per API call.
        
    Returns:
        List of records with added "question_zh" and "solution_zh" fields.
    """
    results = []
    failed_ids = []
    
    # Process in batches
    for i in tqdm(range(0, len(records), batch_size), desc="Translating"):
        batch = records[i:i + batch_size]
        
        # Prepare input for the batch
        batch_input = [{"id": r["id"], "question": r["question"], "solution": r["solution"]} for r in batch]
        input_json = json.dumps(batch_input, ensure_ascii=False, indent=2)
        
        # Build prompt
        prompt = TRANSLATION_PROMPT_TEMPLATE.format(input_json=input_json)
        
        # Call model
        try:
            response = model_client.chat(prompt)
            
            # Parse response
            translations = parse_translation_response(response)
            
            # Create a lookup for translations
            trans_lookup = {
                t["id"]: {
                    "question_zh": t.get("question_zh", ""),
                    "solution_zh": t.get("solution_zh", "")
                } for t in translations
            }
            
            # Merge translations into original records
            for r in batch:
                r_copy = r.copy()
                r_copy["question_zh"] = trans_lookup.get(r["id"]).get("question_zh")
                r_copy["solution_zh"] = trans_lookup.get(r["id"]).get("solution_zh")
                results.append(r_copy)
                
        except Exception as e:
            print(f"Warning: Translation failed for batch starting at {i}: {e}")
            for r in batch:
                failed_ids.append(r["id"])
    
    return results, failed_ids


def parse_translation_response(response: str) -> List[Dict[str, Any]]:
    """
    Parse the JSON response from the translation model.
    
    Args:
        response: Raw response string from the model.
        
    Returns:
        List of dictionaries with "id", "question_zh", and "solution_zh" fields.
    """
    # Try to find JSON array in the response
    response = response.strip()
    
    # Handle markdown code blocks
    if response.startswith("```"):
        lines = response.split('\n')
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.startswith("```")]
        response = '\n'.join(lines)
    
    # Find the JSON array
    start = response.find('[')
    end = response.rfind(']') + 1
    
    if start != -1 and end > start:
        json_str = response[start:end]
        return json.loads(json_str)
    
    return []
