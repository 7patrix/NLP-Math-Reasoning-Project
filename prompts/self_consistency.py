"""
Self-Consistency prompting method.

Reference:
Wang X, Wei J, Schuurmans D, et al. Self-consistency improves chain of
thought reasoning in language models. arXiv:2203.11171, 2022.
"""

from models.loader import generate_response
from evaluation.metrics import extract_final_answer
from collections import Counter


SELF_CONSISTENCY_TEMPLATE = """Solve the following math problem:

Problem: {problem}

Let's think step by step:
"""


def solve_self_consistency(
    model,
    tokenizer,
    problem: str,
    dataset_type: str = None,
    num_samples: int = 5,
    max_new_tokens: int = 1024
) -> str:
    """
    Solve problem using Self-Consistency (majority voting).

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        problem: Math problem text
        dataset_type: Dataset name for answer extraction
        num_samples: Number of reasoning paths to sample (default 5)
        max_new_tokens: Maximum tokens per sample

    Returns:
        The most consistent answer, along with the shortest supporting response
    """
    prompt = SELF_CONSISTENCY_TEMPLATE.format(problem=problem)

    answers = []
    for _ in range(num_samples):
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        answer = extract_final_answer(response, dataset_type=dataset_type)
        answers.append((answer, response))

    # Majority voting on extracted numerical answers
    answer_counts = Counter([a[0] for a in answers])
    most_common = answer_counts.most_common(1)[0][0]
    vote_count = answer_counts[most_common]

    print(f"[Self-Consistency] {vote_count}/{num_samples} paths agree on answer: {most_common}")

    # Return one representative reasoning path and append a canonical voted answer line.
    # This keeps the final answer used by evaluation aligned with majority voting.
    majority_responses = [resp for ans, resp in answers if ans == most_common]
    best_response = min(majority_responses, key=len)

    return f"{best_response}\n\nFinal voted answer: {most_common}\nThe answer is {most_common}"
