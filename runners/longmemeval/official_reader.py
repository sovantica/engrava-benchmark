"""Official LongMemEval reader prompt + context assembly — UPSTREAM-VERBATIM, pinned.

Replicates the reader (answer-generation) protocol of the official LongMemEval
``src/generation/run_generation.py`` at commit
``9e0b455f4ef0e2ab8f2e582289761153549043fc`` (the same commit as the scorer; see
``official_scorer/UPSTREAM.md``). The public runner is the source of the published
number, so its reader must be the unmodified official one — these strings and the
assembly steps are reproduced byte-faithfully, not re-authored.

This module is **runner-owned** and uniform for every system: it consumes the
adapter's ranked neutral unit ids plus runner-internal session content, and never
reads gold / answer / evidence flags (the ``has_answer`` key is popped exactly as
upstream does). The adapter never touches it — equal footing is preserved.

Official default reading method (``con``) resolves to ``--cot true`` (the
step-by-step extract-then-reason prompt, A.1 below); the separate chain-of-note
extraction pass (``--con true``) is the non-default ``con-separate`` path and is
not the LongMemEval-S default, so it is not replicated here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- VERBATIM reader prompt (run_generation.py:55, cot=true default) ---------
# The three ``{}`` are filled .format(history_string, question_date, question).
# Single user message; no system role; no per-question-type variation (upstream).
COT_READER_PROMPT_TEMPLATE = (
    "I will give you several history chats between you and a user. Please answer "
    "the question based on the relevant chat history. Answer the question step by "
    "step: first extract all the relevant information, and then reason over the "
    "information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n"
    "Question: {}\nAnswer (step by step):"
)

# Official reader generation length for cot=true (run_generation.py:340-343).
COT_GEN_LENGTH = 800
# gpt-4o context window (run_generation.py model2maxlength table).
GPT4O_MAX_LENGTH = 128000
# History-token budget = model_max - gen_length - 1000 (run_generation.py:343).
MAX_RETRIEVAL_LENGTH = GPT4O_MAX_LENGTH - COT_GEN_LENGTH - 1000  # 126200
# Encoding for gpt-4o (run_generation.py:329-331).
OPENAI_ENCODING = "o200k_base"


def assemble_history(
    ranked_rounds: Sequence[tuple[str, list[dict[str, str]]]],
    *,
    top_k: int,
    encoding_name: str = OPENAI_ENCODING,
    max_retrieval_length: int = MAX_RETRIEVAL_LENGTH,
) -> str:
    r"""Build the official ``history_string`` from ranked, round-expanded chunks.

    Replicates run_generation.py for ``flat-turn`` + ``history_format == "json"``:

    1. **top-K by rank** — keep the first ``top_k`` ranked chunks (``[:topk_context]``).
    2. **chronological re-sort** — sort the kept chunks ascending by their date
       (``x[0]``), so the prompt is oldest-first.
    3. **JSON formatting** — each chunk is ``"\\n" + json.dumps(turn_dicts)`` wrapped
       as ``"### Session {i+1}:\\nSession Date: {date}\\nSession Content:\\n{sess}\\n"``.
    4. **tiktoken truncation** — encode with ``o200k_base``; if over budget keep the
       FIRST ``max_retrieval_length`` tokens (head-keep, drop the newest sessions).

    Args:
        ranked_rounds: ``(session_date, round_turns)`` pairs in retrieval-rank order
            (best first). ``round_turns`` is the round's list of ``{"role","content"}``
            dicts (already round-expanded, ``has_answer`` popped).
        top_k: The official ``topk_context`` — number of ranked chunks to keep.
        encoding_name: tiktoken encoding (``o200k_base`` for gpt-4o).
        max_retrieval_length: Token budget for the history string.

    Returns:
        The assembled, possibly-truncated ``history_string``.

    """
    import tiktoken  # noqa: PLC0415 - heavy optional dep, imported on use

    kept = list(ranked_rounds)[:top_k]
    # Chronological re-sort, ascending by date string (run_generation.py:225).
    kept.sort(key=lambda x: x[0])

    history_string = ""
    for i, (chunk_date, round_turns) in enumerate(kept):
        sess_string = "\n" + json.dumps(round_turns)
        history_string += (
            f"\n### Session {i + 1}:\nSession Date: {chunk_date}\nSession Content:\n{sess_string}\n"
        )

    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(history_string, allowed_special={"<|endoftext|>"})
    if len(tokens) > max_retrieval_length:
        history_string = tokenizer.decode(tokens[:max_retrieval_length])
    return history_string


def build_reader_prompt(question: str, question_date: str, history_string: str) -> str:
    """Build the official reader user message (verbatim CoT template).

    Args:
        question: The question text.
        question_date: The question's ``question_date`` string (may be empty).
        history_string: The assembled history (see :func:`assemble_history`).

    Returns:
        The single user-message string the official reader sends (no system role).

    """
    return COT_READER_PROMPT_TEMPLATE.format(history_string, question_date, question)
