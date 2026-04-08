"""
Main experiment runner.
"""

import json
import os
import re
import sys
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from models.loader import load_model
from data.loader import load_dataset_by_name
from evaluation.metrics import evaluate_all
from prompts.cot import solve_cot
from prompts.self_refine import solve_self_refine
from prompts.self_consistency import solve_self_consistency


PROMPT_METHODS = {
    "cot": solve_cot,
    "self_refine": solve_self_refine,
    "self_consistency": solve_self_consistency,
}


def _sanitize_filename(name: str) -> str:
    """Remove or replace characters unsafe for filenames."""
    return re.sub(r'[<>:"/\\|?*\s]', '_', name)


def _ensure_results_dir(output_file: str) -> None:
    """Create results directory if it doesn't exist."""
    directory = os.path.dirname(output_file)
    if directory:
        os.makedirs(directory, exist_ok=True)


def run_experiment(
    model_name: str,
    dataset_name: str,
    prompt_method: str,
    output_file: str = None,
    limit: int = None,
    skip_existing: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Run a single experiment.

    Args:
        model_name: "qwen2.5-math-1.5b" or "deepseek-r1-qwen-1.5b"
        dataset_name: "math500", "gsm8k", or "aime2024"
        prompt_method: "cot", "self_refine", or "self_consistency"
        output_file: Path to save results. Auto-generated if None.
        limit: Limit number of samples for quick testing
        skip_existing: If True and output_file exists, skip running and load from disk
        seed: Global random seed for reproducibility

    Returns:
        Dict with results and metrics
    """
    # Auto-generate output file name if not provided
    if output_file is None:
        output_file = os.path.join(
            "results",
            f"{_sanitize_filename(model_name)}_{_sanitize_filename(dataset_name)}_{_sanitize_filename(prompt_method)}.json"
        )

    _ensure_results_dir(output_file)

    # Skip if already completed
    if skip_existing and os.path.exists(output_file):
        print(f"[SKIP] Results already exist at {output_file}")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"[WARN] Could not read existing file, re-running...")

    print(f"\n{'='*60}")
    print(f"Running: {model_name} + {dataset_name} + {prompt_method}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    run_start = time.time()
    run_start_iso = datetime.now().isoformat(timespec="seconds")

    # Set global random seeds for reproducible runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Validate prompt method
    if prompt_method not in PROMPT_METHODS:
        raise ValueError(
            f"Unknown prompt method: '{prompt_method}'. "
            f"Available: {list(PROMPT_METHODS.keys())}"
        )

    # Load model
    model, tokenizer = load_model(model_name)

    # Load dataset
    dataset = load_dataset_by_name(dataset_name)
    if not dataset:
        raise RuntimeError(f"Dataset '{dataset_name}' is empty or could not be loaded.")

    if limit:
        dataset = dataset[:limit]

    prompt_fn = PROMPT_METHODS[prompt_method]

    # Run inference
    predictions = []
    references = []

    for sample in tqdm(dataset, desc=f"{model_name}/{dataset_name}/{prompt_method}"):
        problem = sample["problem"]
        answer = sample["answer"]

        try:
            if prompt_method == "self_consistency":
                prediction = prompt_fn(model, tokenizer, problem, dataset_type=dataset_name)
            else:
                prediction = prompt_fn(model, tokenizer, problem)
            predictions.append(prediction)
        except Exception as e:
            print(f"[WARN] Error on sample: {e}")
            predictions.append("")

        references.append(answer)

    # Evaluate
    metrics = evaluate_all(predictions, references, dataset_type=dataset_name)

    duration_seconds = time.time() - run_start

    if prompt_method == "self_consistency":
        decoding = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_samples": 5,
        }
    else:
        decoding = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        }

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "prompt_method": prompt_method,
        "run_metadata": {
            "seed": seed,
            "started_at": run_start_iso,
            "duration_seconds": round(duration_seconds, 2),
            "device": str(getattr(model, "device", "unknown")),
            "decode_config": decoding,
            "max_new_tokens": 1024,
        },
        "metrics": metrics,
        "samples": [
            {"problem": d["problem"], "prediction": p, "reference": r}
            for d, p, r in zip(dataset, predictions, references)
        ]
    }

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")

    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Avg Response Length: {metrics['response_length']['char_mean']:.1f} chars")
    print(f"Duration: {duration_seconds:.1f}s")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    return results


def run_all_experiments():
    """Run all combinations of experiments."""
    models = ["qwen2.5-math-1.5b", "deepseek-r1-qwen-1.5b"]
    datasets = ["math500", "gsm8k", "aime2024"]
    methods = ["cot", "self_refine", "self_consistency"]

    all_results = []

    for model in models:
        for dataset in datasets:
            for method in methods:
                output_file = os.path.join(
                    "results",
                    f"{_sanitize_filename(model)}_{_sanitize_filename(dataset)}_{_sanitize_filename(method)}.json"
                )
                result = run_experiment(model, dataset, method, output_file)
                all_results.append(result)

    # Save combined results
    os.makedirs("results", exist_ok=True)
    with open("results/all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run math reasoning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/runner.py --model qwen2.5-math-1.5b --dataset math500 --method cot
  python experiments/runner.py --model deepseek-r1-qwen-1.5b --dataset gsm8k --method self_refine --limit 10
  python experiments/runner.py --all
"""
    )
    parser.add_argument("--model", default="qwen2.5-math-1.5b",
                        choices=["qwen2.5-math-1.5b", "deepseek-r1-qwen-1.5b"])
    parser.add_argument("--dataset", default="math500",
                        choices=["math500", "gsm8k", "aime2024"])
    parser.add_argument("--method", default="cot",
                        choices=["cot", "self_refine", "self_consistency"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples for quick testing")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even if results file already exists")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for reproducible runs")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        run_all_experiments()
    else:
        run_experiment(
            args.model,
            args.dataset,
            args.method,
            output_file=args.output,
            limit=args.limit,
            skip_existing=not args.no_skip,
            seed=args.seed,
        )
