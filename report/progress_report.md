# Mid-Term Progress Report - Mathematical Reasoning Ability of Large Language Models

**Course**: CS6493 Natural Language Processing  
**Project**: Topic 1 - Mathematical Reasoning Ability of Large Language Models  
**Date**: March 25, 2026

## 1. Research Background and Objectives

Large language models (LLMs) have shown strong performance on general NLP tasks, but mathematical reasoning remains challenging, especially on datasets such as GSM8K, MATH-500, and AIME-2024. This project studies how prompt engineering improves reasoning quality for small-scale (1.5B) math-capable LLMs.

The project objectives are:

1. Build a standardized pipeline for dataset loading and preprocessing for GSM8K, MATH-500, and AIME-2024.
2. Implement three prompt methods (CoT, Self-Refine, Self-Consistency) under a unified inference/evaluation framework.
3. Run 18 controlled experiments across 2 models x 3 datasets x 3 prompt methods.
4. Analyze performance using Accuracy and Average Response Length.

## 2. Methods and Experimental Design

### 2.1 Models

- Qwen2.5-Math-1.5B
- DeepSeek-R1-Qwen-1.5B

Implementation note: the CLI model key `deepseek-r1-qwen-1.5b` maps to Hugging Face model `deepseek-ai/DeepSeek-R1-Qwen-1.5B` in the current codebase.

### 2.2 Datasets

- GSM8K test set (~1319 samples), downloaded and normalized via script.
- MATH-500 (500 samples), loaded from the public split `HuggingFaceH4/MATH-500` (test split).
- AIME-2024 (30 samples), sourced from public split `HuggingFaceH4/aime_2024` and stored in `data/AIME-2024/aime2024.json`.

### 2.3 Prompt Methods

1. Chain of Thought (CoT)
2. Self-Refine (iterative self-feedback refinement)
3. Self-Consistency (5 sampled reasoning paths with majority voting)

### 2.4 Evaluation Metrics

- Accuracy: ratio of correctly solved problems to total problems.
- Average Response Length: mean character length of model outputs.

## 3. Work Completed So Far

### 3.1 Environment and Data Pipeline

1. Built a Python 3.10 environment and installed required dependencies.
2. Implemented automatic data download for GSM8K and MATH-500.
3. Implemented unified dataset loader with consistent `{"problem", "answer"}` interface.
4. Added strict AIME-2024 validation to prevent empty/invalid samples from being used.

### 3.2 Core Framework

1. Implemented model loader for both model families with configurable cache location.
2. Implemented CoT, Self-Refine, and Self-Consistency prompt modules.
3. Implemented experiment runner and batch runner with skip-existing behavior for resume.
4. Implemented metrics pipeline (answer extraction, normalization, accuracy, response length).
5. Built a Streamlit UI for single-question testing, result browsing, and comparison.

### 3.3 Experiment Progress

The 18-experiment plan has been set up in code and execution has started.

- Completed: Experiment ID 4 (Qwen2.5-Math-1.5B + MATH-500 + CoT)
- Remaining experiments are ready to run through `run_batch.py`.

## 4. Current Challenges

1. Runtime efficiency: Self-Consistency is the most expensive setting due to multi-sample decoding.
2. AIME-style answer matching: high-difficulty competition answers can be brittle under simple normalization.
3. Prompt specificity: current prompts are general-purpose and can be further specialized by dataset type.

## 5. Next-Step Plan

1. Complete all 18 experiments with multi-machine parallelization when available.
2. Analyze results by model, dataset, and prompt method.
3. Improve evaluation robustness for AIME-like answers and run targeted re-check experiments.
4. Finalize report and presentation materials before the final deadline.

## 6. Mid-Term Summary

The project has completed framework development, data pipeline setup, and initial experiment verification. The current implementation is reproducible and extensible, and is ready for full-scale experiment execution and final analysis.
