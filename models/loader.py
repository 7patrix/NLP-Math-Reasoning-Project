"""
Model loading and configuration for Qwen2.5-Math and DeepSeek-R1.
"""

import os
import sys

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load user config (each team member edits config.py to set their own paths)
try:
    from config import HF_HOME as _CONFIG_HF_HOME
except ImportError:
    _CONFIG_HF_HOME = None

# ─── Cache directory resolution ─────────────────────────────────────────────
# Priority: (1) config.HF_HOME  →  (2) HF_HOME env var  →  (3) local ./cache
#
# Team members: edit config.py to set your model cache path!
_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

if _CONFIG_HF_HOME:
    DEFAULT_CACHE = _CONFIG_HF_HOME
elif os.getenv("HF_HOME"):
    DEFAULT_CACHE = os.getenv("HF_HOME")
else:
    DEFAULT_CACHE = _DEFAULT_CACHE

os.environ["HF_HOME"] = DEFAULT_CACHE

# Configure HuggingFace mirror for users behind the Great Firewall
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print(f"[loader] HF_HOME set to: {DEFAULT_CACHE}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Local model paths (auto-discovered from cache directory)
LOCAL_MODEL_PATHS = {
    "qwen2.5-math-1.5b": os.path.join(DEFAULT_CACHE, "hub", "models--Qwen--Qwen2.5-Math-1.5B", "snapshots"),
    "deepseek-r1-qwen-1.5b": os.path.join(DEFAULT_CACHE, "hub", "models--deepseek-ai--DeepSeek-R1-Qwen-1.5B", "snapshots"),
}

MODEL_CONFIGS = {
    "qwen2.5-math-1.5b": {
        "name": "Qwen/Qwen2.5-Math-1.5B",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "deepseek-r1-qwen-1.5b": {
        "name": "deepseek-ai/DeepSeek-R1-Qwen-1.5B",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
}


def find_model_snapshot(base_path: str) -> str:
    """Find the actual model snapshot folder."""
    if not os.path.exists(base_path):
        return None
    for item in os.listdir(base_path):
        snapshot_path = os.path.join(base_path, item)
        if os.path.isdir(snapshot_path):
            if any(f in os.listdir(snapshot_path) for f in ['config.json', 'model.safetensors', 'pytorch_model.bin', 'model.bin']):
                return snapshot_path
    return None


def load_model(model_key: str, use_local: bool = True):
    """
    Load model and tokenizer.

    Args:
        model_key: One of "qwen2.5-math-1.5b" or "deepseek-r1-qwen-1.5b"
        use_local: Whether to prefer locally cached model

    Returns:
        model, tokenizer
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_key]
    device = config["device"]

    if use_local and model_key in LOCAL_MODEL_PATHS:
        local_path = find_model_snapshot(LOCAL_MODEL_PATHS[model_key])
        if local_path:
            print(f"Loading local model from: {local_path}")
            model_name = local_path
            local_files_only = True
        else:
            print(f"Local model not found, falling back to HuggingFace Hub: {config['name']}")
            model_name = config["name"]
            local_files_only = False
    else:
        print(f"Loading from HuggingFace Hub: {config['name']}")
        model_name = config["name"]
        local_files_only = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if device == "cpu":
        model = model.to("cpu")

    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate response from model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (used when do_sample=True)
        top_p: Nucleus sampling parameter (used when do_sample=True)
        do_sample: Whether to use stochastic sampling

    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()

    return response
