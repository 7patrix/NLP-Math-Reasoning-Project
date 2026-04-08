"""
Evaluation metrics for mathematical reasoning.
"""

import re
from typing import Dict, List, Optional


def extract_final_answer(text: str, dataset_type: str = None) -> str:
    """
    Extract the final numerical answer from model output or reference answer.
    Handles various formats:
    - "\\boxed{42}" (MATH-500 format)
    - "The answer is 42", "Answer: 42", "answer is 42"
    - "= 0" or "= 42" within text
    - Just the number at the end

    Args:
        text: The text to extract answer from
        dataset_type: Dataset name (e.g., 'math500', 'gsm8k', 'aime2024')
                     to enable dataset-specific extraction strategies

    Returns:
        Extracted answer string, or original text if nothing found
    """
    if not text:
        return ""

    dataset_key = (dataset_type or "").lower().replace("-", "").replace("_", "")

    # --- 1. Try boxed answer (LaTeX) ---
    boxed_match = re.search(r'\\boxed\{([^{}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # --- 2. Prefer the last explicit final-answer declaration ---
    final_answer_matches = re.findall(
        r'(?:final\s+voted\s+answer|final\s+answer|the\s+answer\s+is|answer\s*:)\s*([^\n.]+)',
        text,
        re.IGNORECASE,
    )
    if final_answer_matches:
        return final_answer_matches[-1].strip()

    # --- 3. Dataset-specific extraction ---
    if dataset_key.startswith("gsm8k"):
        gsm8k_match = re.search(r'####\s*([^\n]+)', text)
        if gsm8k_match:
            return gsm8k_match.group(1).strip()

    if dataset_key.startswith("aime"):
        aime_match = re.search(r'\b(\d{1,3})\b\s*$', text.strip())
        if aime_match:
            return aime_match.group(1)

    # --- 4. Try "answer is X" / "answer: X" patterns ---
    answer_match = re.search(
        r'(?:the\s+)?answer\s+is\s*[:\s]*([^\n.]+)',
        text, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip()

    # --- 5. Prefer the last equation result instead of first ---
    equals_matches = re.findall(r'=\s*(-?[\d]+\.?[\d]*)', text)
    if equals_matches:
        return equals_matches[-1]

    # --- 6. Try number at end of text (after punctuation) ---
    end_number_match = re.search(r'[.,;:]\s*(-?[\d]+\.?[\d]*)\s*$', text.strip())
    if end_number_match:
        return end_number_match.group(1)

    # --- 7. Try last standalone number ---
    numbers = re.findall(r'-?[\d]+\.?[\d]*', text)
    if numbers:
        return numbers[-1]

    return text.strip()


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    Steps:
    1. Extract boxed answer if present
    2. For long text, take the last number
    3. Remove commas, dollar signs, spaces
    4. Convert fractions to decimal
    5. Lowercase and strip
    """
    if not answer:
        return ""

    # Extract boxed answer if present
    boxed_match = re.search(r'\\boxed\{([^{}]+)\}', answer)
    if boxed_match:
        answer = boxed_match.group(1).strip()
    elif len(answer) > 50:
        # Long answer text: take last number
        numbers = re.findall(r'-?[\d]+\.?[\d]*', answer)
        if numbers:
            answer = numbers[-1]

    # Remove formatting characters
    answer = re.sub(r'[,$\\%{}\s]', '', answer)

    # Convert fractions
    if '/' in answer:
        try:
            num, denom = answer.split('/', 1)
            answer = str(float(num) / float(denom))
        except (ValueError, ZeroDivisionError):
            pass

    # Canonicalize numeric strings to avoid mismatches like "025" vs "25".
    if re.fullmatch(r'-?\d+', answer):
        try:
            answer = str(int(answer))
        except ValueError:
            pass
    elif re.fullmatch(r'-?\d+\.\d+', answer):
        answer = answer.rstrip('0').rstrip('.')
        if answer in {"", "-0"}:
            answer = "0"

    return answer.lower().strip()


def compute_accuracy(
    predictions: List[str],
    references: List[str],
    dataset_type: Optional[str] = None
) -> float:
    """
    Compute accuracy by comparing predictions against references.

    Args:
        predictions: List of model predictions
        references: List of ground truth answers
        dataset_type: Optional dataset name for format-specific extraction

    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_answer(extract_final_answer(pred, dataset_type))
        ref_norm = normalize_answer(extract_final_answer(ref, dataset_type))
        if pred_norm == ref_norm:
            correct += 1

    return correct / len(predictions)


def compute_response_length(responses: List[str]) -> Dict[str, float]:
    """Compute response length statistics."""
    import statistics

    if not responses:
        return {
            "char_mean": 0.0, "char_median": 0.0,
            "char_min": 0, "char_max": 0,
            "word_mean": 0.0, "word_median": 0.0,
            "word_min": 0, "word_max": 0,
        }

    char_lengths = [len(r) for r in responses]
    word_lengths = [len(r.split()) for r in responses]

    return {
        "char_mean": statistics.mean(char_lengths),
        "char_median": statistics.median(char_lengths),
        "char_min": min(char_lengths),
        "char_max": max(char_lengths),
        "word_mean": statistics.mean(word_lengths),
        "word_median": statistics.median(word_lengths),
        "word_min": min(word_lengths),
        "word_max": max(word_lengths),
    }


def evaluate_all(
    predictions: List[str],
    references: List[str],
    dataset_type: Optional[str] = None
) -> Dict:
    """
    Compute all evaluation metrics.

    Args:
        predictions: List of model predictions
        references: List of ground truth answers
        dataset_type: Optional dataset name

    Returns:
        Dict with all metrics
    """
    return {
        "accuracy": compute_accuracy(predictions, references, dataset_type),
        "response_length": compute_response_length(predictions),
        "total_samples": len(predictions)
    }
