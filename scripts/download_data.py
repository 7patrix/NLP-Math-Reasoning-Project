"""
Download datasets from Hugging Face.

Run this script to download all required datasets:
    python scripts/download_data.py

The data will be saved to:
    data/GSM8K/test.json
    data/MATH-500/test.json
    (AIME-2024 needs manual download)
"""

import os
import json

# Use HuggingFace mirror for users behind the Great Firewall
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Resolve project root relative to this file
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(_project_root, "data", "GSM8K"), exist_ok=True)
os.makedirs(os.path.join(_project_root, "data", "MATH-500"), exist_ok=True)


def download_gsm8k():
    """Download and save GSM8K test set."""
    from datasets import load_dataset

    print("Downloading GSM8K...")
    dataset = load_dataset("gsm8k", "main")

    test_data = []
    for item in dataset["test"]:
        answer_text = item["answer"]
        if "####" in answer_text:
            answer = answer_text.split("####")[-1].strip()
        else:
            answer = answer_text

        test_data.append({
            "problem": item["question"],
            "answer": answer
        })

    out_path = os.path.join(_project_root, "data", "GSM8K", "test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"GSM8K saved: {len(test_data)} samples -> {out_path}")


def download_math():
    """Download and save MATH-500 test set."""
    from datasets import load_dataset
    import re

    print("Downloading MATH dataset...")
    dataset = load_dataset("hendrycks/competition_math")

    def extract_math_answer(text: str) -> str:
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()
        return text.strip()

    test_data = []
    for item in dataset["test"]:
        test_data.append({
            "problem": item["problem"],
            "solution": item["solution"],
            "answer": extract_math_answer(item["solution"]),
            "level": item.get("level", ""),
            "type": item.get("type", "")
        })

    out_path = os.path.join(_project_root, "data", "MATH-500", "test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_data[:500], f, indent=2, ensure_ascii=False)

    print(f"MATH-500 saved: {min(500, len(test_data))} samples -> {out_path}")


def download_aime2024():
    """
    AIME 2024 needs manual download.
    Provide instructions and expected format.
    """
    out_path = os.path.join(_project_root, "data", "AIME-2024", "aime2024.json")
    if os.path.exists(out_path):
        print(f"AIME-2024 already exists at {out_path}")
        return

    print(f"AIME-2024 dataset requires manual download.")
    print(f"Please download from official sources and place at: {out_path}")
    print("Expected format: [{\"problem\": \"...\", \"answer\": \"...\"}, ...]")
    print("No placeholder file will be created. Please provide real AIME-2024 data.")


if __name__ == "__main__":
    print("Downloading datasets...\n")
    print(f"Data directory: {os.path.join(_project_root, 'data')}")
    print()

    download_gsm8k()
    print()
    download_math()
    print()
    download_aime2024()

    print("\nDone!")
