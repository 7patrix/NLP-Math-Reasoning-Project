"""
Chain of Thought (CoT) prompting method.

Reference:
Wei J, Wang X, Schuurmans D, et al. Chain-of-thought prompting elicits 
reasoning in large language models. NeurIPS 2022.
"""

from models.loader import generate_response


COT_TEMPLATE = """Solve the following math problem step by step.

Problem: {problem}

Let's think through this step by step:
"""


def solve_cot(model, tokenizer, problem: str, max_new_tokens: int = 1024) -> str:
    """
    Solve problem using Chain of Thought prompting.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        problem: Math problem text
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Model response with reasoning steps
    """
    prompt = COT_TEMPLATE.format(problem=problem)
    response = generate_response(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return response
