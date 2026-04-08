"""
Self-Refine prompting method.

Reference:
Madaan A, Tandon N, Gupta P, et al. Self-refine: Iterative refinement 
with self-feedback. NeurIPS 2024.
"""

from models.loader import generate_response


SELF_REFINE_INITIAL = """Solve the following math problem:

Problem: {problem}

Solution:
"""

SELF_REFINE_FEEDBACK = """Review your solution to the following problem:

Problem: {problem}

Your solution:
{solution}

Please identify any errors or areas for improvement, then provide a refined solution.

Feedback and Refined Solution:
"""


def solve_self_refine(
    model, 
    tokenizer, 
    problem: str, 
    max_iterations: int = 3,
    max_new_tokens: int = 1024
) -> str:
    """
    Solve problem using Self-Refine prompting.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        problem: Math problem text
        max_iterations: Maximum refinement iterations
        max_new_tokens: Maximum tokens to generate per step
    
    Returns:
        Final refined solution
    """
    # Initial solution
    initial_prompt = SELF_REFINE_INITIAL.format(problem=problem)
    solution = generate_response(
        model,
        tokenizer,
        initial_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    
    # Iterative refinement
    for i in range(max_iterations):
        feedback_prompt = SELF_REFINE_FEEDBACK.format(
            problem=problem,
            solution=solution
        )
        refined = generate_response(
            model,
            tokenizer,
            feedback_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
        # Extract refined solution (after "Feedback and Refined Solution:")
        if "refined solution:" in refined.lower():
            solution = refined.split(":", 1)[-1].strip()
        else:
            solution = refined
    
    return solution
