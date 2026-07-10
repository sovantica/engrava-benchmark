"""Official LongMemEval judge prompts — UPSTREAM-VERBATIM, pinned.

Mirrors ``get_anscheck_prompt`` from the official LongMemEval
``src/evaluation/evaluate_qa.py`` at commit
``9e0b455f4ef0e2ab8f2e582289761153549043fc`` (see ``UPSTREAM.md``). The prompt
strings are the official scoring contract and are reproduced **byte-faithfully** so
the judge asks the exact official question. Do NOT edit the prompt wording — any
change would diverge from the pinned scorer and invalidate ``scorer_version``.

The judge model is ``gpt-4o-2024-08-06`` via ``api.openai.com`` direct, called with
``temperature=0, max_tokens=10``; the official label is
``'yes' in completion.strip().lower()`` (the runner's judge applies that rule).
"""

from __future__ import annotations

# The 6 official question types whose judge uses the shared "Correct Answer"
# template (single-session-preference, knowledge-update, temporal-reasoning, and
# abstention items override this below).
_SHARED_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains all "
    "the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the "
    "answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}"
    "\n\nIs the model response correct? Answer yes or no only."
)

_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains all "
    "the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the "
    "answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days. If the question asks for the number of days/weeks/months, etc., "
    "and the model makes off-by-one errors (e.g., predicting 19 days when the answer "
    "is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect "
    "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes "
    "or no only."
)

_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}"
    "\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the "
    "points in the rubric. The response is correct as long as it recalls and "
    "utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: "
    "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some "
    "other information is given but the asked information is not.\n\nQuestion: {}\n\n"
    "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the "
    "question as unanswerable? Answer yes or no only."
)


def get_anscheck_prompt(
    task: str, question: str, answer: str, response: str, *, abstention: bool = False
) -> str:
    """Build the official judge prompt for one question (upstream-verbatim).

    Mirrors the official ``get_anscheck_prompt(task, question, answer, response,
    abstention)``. The second format slot is the correct answer / rubric /
    explanation depending on the branch.

    Args:
        task: The official ``question_type``.
        question: The question text.
        answer: The gold answer (or rubric for preference / explanation for
            abstention).
        response: The model's answer to judge.
        abstention: Whether this is an abstention item (``_abs`` qid).

    Returns:
        The official judge prompt string.

    Raises:
        ValueError: If ``task`` is not a recognized official question type.

    """
    if abstention:
        return _ABSTENTION_TEMPLATE.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        template = _SHARED_TEMPLATE
    elif task == "temporal-reasoning":
        template = _TEMPORAL_TEMPLATE
    elif task == "knowledge-update":
        template = _KNOWLEDGE_UPDATE_TEMPLATE
    elif task == "single-session-preference":
        template = _PREFERENCE_TEMPLATE
    else:
        msg = f"unknown official question_type: {task!r}"
        raise ValueError(msg)
    return template.format(question, answer, response)


def build_judge_prompt(
    *, question: str, answer: str, gold: str, question_type: str, abstention: bool = False
) -> str:
    """Adapter over :func:`get_anscheck_prompt` with keyword arguments.

    Args:
        question: The question text.
        answer: The model's answer to judge.
        gold: The gold answer (rubric/explanation for preference/abstention).
        question_type: The official ``question_type``.
        abstention: Whether this is an abstention item.

    Returns:
        The official judge prompt string.

    """
    return get_anscheck_prompt(question_type, question, gold, answer, abstention=abstention)
