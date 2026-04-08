"""
Data loading utilities for MATH-500, GSM8K, and AIME-2024 datasets.
"""

import json
import os
import re
from typing import List, Dict

# Configure HuggingFace mirror for users behind the Great Firewall
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Resolve project root relative to this file
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


def _get_data_path(*parts: str) -> str:
    """Get path relative to project data directory."""
    return os.path.join(_DATA_DIR, *parts)


def extract_math_answer(text: str) -> str:
    """Extract answer from MATH dataset solution text."""
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    return text.strip()


def load_math500() -> List[Dict]:
    """Load MATH-500 dataset from local file or public Hugging Face split."""
    local_path = _get_data_path("MATH-500", "test.json")
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            for item in data:
                if 'solution' in item and 'answer' not in item:
                    item['answer'] = extract_math_answer(item['solution'])
            return data

    # Fallback: use public MATH-500 split from Hugging Face
    try:
        from datasets import load_dataset
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        samples = []
        for item in dataset:
            problem = item.get("problem") or item.get("question")
            answer = item.get("answer") or item.get("final_answer") or item.get("expected_answer")
            if not problem or not answer:
                continue
            samples.append({
                "problem": str(problem).strip(),
                "answer": str(answer).strip(),
                "level": item.get("level", ""),
                "type": item.get("type", "")
            })
        return samples
    except Exception as e:
        print(f"[WARNING] Could not load public MATH-500 split from HuggingFace: {e}")
        print("[WARNING] Please run: python scripts/download_data.py")
        return []


def load_gsm8k() -> List[Dict]:
    """Load GSM8K test set from local file or Hugging Face."""
    local_path = _get_data_path("GSM8K", "test.json")
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            return data

    # Fallback: try loading from Hugging Face
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        samples = []
        for item in dataset:
            answer_text = item["answer"]
            if "####" in answer_text:
                answer = answer_text.split("####")[-1].strip()
            else:
                answer = answer_text
            samples.append({
                "problem": item["question"],
                "answer": answer
            })

        # Save to local for future use
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        return samples
    except Exception as e:
        print(f"[WARNING] Could not load GSM8K from HuggingFace: {e}")
        print("[WARNING] Please run: python scripts/download_data.py")
        return []


def load_aime2024() -> List[Dict]:
    """Load AIME 2024 from local file or public Hugging Face split."""
    local_path = _get_data_path("AIME-2024", "aime2024.json")
    if os.path.exists(local_path):
        with open(local_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            raise RuntimeError(
                f"Invalid AIME-2024 file at {local_path}: expected a non-empty list of samples."
            )

        invalid_indices = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                invalid_indices.append(idx)
                continue

            problem = str(item.get("problem", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not problem or not answer:
                invalid_indices.append(idx)
                continue

            year = str(item.get("year", "")).strip()
            if year and year != "2024":
                invalid_indices.append(idx)

        if invalid_indices:
            preview = ", ".join(str(i) for i in invalid_indices[:10])
            raise RuntimeError(
                "Invalid AIME-2024 samples found (missing/empty fields or non-2024 year) "
                f"at indices: {preview}."
            )

        return data

    # Fallback: use public AIME-2024 split from Hugging Face
    try:
        from datasets import load_dataset
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        samples = []
        for item in dataset:
            problem = item.get("problem")
            answer = item.get("answer")
            if not problem or not answer:
                continue
            samples.append({
                "problem": str(problem).strip(),
                "answer": str(answer).strip(),
                "year": str(item.get("year", "2024")).strip() or "2024",
            })
        return samples
    except Exception as e:
        print(f"[WARNING] Could not load public AIME-2024 split from HuggingFace: {e}")
        print("[WARNING] Please run: python scripts/download_data.py")
        return []


def load_dataset_by_name(name: str) -> List[Dict]:
    """Load dataset by name."""
    name_lower = name.lower().replace("-", "").replace("_", "")
    loaders = {
        "math500": load_math500,
        "gsm8k": load_gsm8k,
        "aime2024": load_aime2024,
        "math": load_math500,
    }

    loader = None
    for key, fn in loaders.items():
        if name_lower.startswith(key) or key.startswith(name_lower):
            loader = fn
            break

    if loader is None:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available datasets: {list(loaders.keys())}"
        )

    data = loader()
    if not data:
        raise RuntimeError(
            f"Failed to load dataset '{name}'. "
            f"Please ensure data files exist or run: python scripts/download_data.py"
        )

    return data
