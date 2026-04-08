"""
Download datasets from Hugging Face.

Run this script to download all required datasets:
    python scripts/download_data.py

The data will be saved to:
    data/GSM8K/test.json
    data/MATH-500/test.json
    data/AIME-2024/aime2024.json
"""

import os
import json

# Use HuggingFace mirror for users behind the Great Firewall
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Resolve project root relative to this file
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(_project_root, "data", "GSM8K"), exist_ok=True)
os.makedirs(os.path.join(_project_root, "data", "MATH-500"), exist_ok=True)
os.makedirs(os.path.join(_project_root, "data", "AIME-2024"), exist_ok=True)


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
    """Download and save MATH-500 public test split."""
    from datasets import load_dataset

    print("Downloading MATH-500 public split...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    test_data = []
    for item in dataset:
        problem = item.get("problem") or item.get("question")
        answer = item.get("answer") or item.get("final_answer") or item.get("expected_answer")
        if not problem or not answer:
            continue

        test_data.append({
            "problem": str(problem).strip(),
            "answer": str(answer).strip(),
            "level": item.get("level", ""),
            "type": item.get("type", "")
        })

    out_path = os.path.join(_project_root, "data", "MATH-500", "test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"MATH-500 saved: {len(test_data)} samples -> {out_path}")


def download_aime2024():
    """Download and save official AIME-2024 public split."""
    from datasets import load_dataset

    print("Downloading AIME-2024 public split...")
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

    test_data = []
    for item in dataset:
        problem = item.get("problem")
        answer = item.get("answer")
        if not problem or not answer:
            continue

        test_data.append({
            "problem": str(problem).strip(),
            "answer": str(answer).strip(),
            "year": str(item.get("year", "2024")).strip() or "2024"
        })

    out_path = os.path.join(_project_root, "data", "AIME-2024", "aime2024.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"AIME-2024 saved: {len(test_data)} samples -> {out_path}")


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
