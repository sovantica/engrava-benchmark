"""Tests for the official LongMemEval reader prompt + context-assembly stages.

These are deterministic and offline (tiktoken loads a local encoding; no network),
so they run in the standard suite and exercise the assembly that the smoke path
uses. Each official stage — round-expansion, chronological re-sort, JSON
formatting, tiktoken truncation — is asserted, plus the leakage invariant that the
reader path reads no gold/evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken

from runners.longmemeval import official_reader
from runners.longmemeval import run as runner

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "longmemeval_smoke.json"


def _question_with_sessions() -> runner.Question:
    item = {
        "question_id": "q",
        "question_type": "multi-session",
        "question": "What pet?",
        "question_date": "2026-03-01",
        "answer": "a beagle",
        "haystack_dates": ["2026-02-01", "2026-01-01"],  # session 0 is NEWER than 1
        "haystack_session_ids": ["s_new", "s_old"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "NEW: I adopted a beagle.", "has_answer": True},
                {"role": "assistant", "content": "Nice, a beagle!"},
                {"role": "user", "content": "NEW: tail filler."},
            ],
            [
                {"role": "user", "content": "OLD: I like jazz.", "has_answer": False},
                {"role": "assistant", "content": "Jazz is great."},
            ],
        ],
    }
    path = FIXTURE.parent / "_tmp_reader_item.json"
    path.write_text(json.dumps([item]), encoding="utf-8")
    try:
        return runner.load_questions(path)[0]
    finally:
        path.unlink()


# --- prompt ------------------------------------------------------------------ #
def test_reader_prompt_is_verbatim_cot() -> None:
    prompt = official_reader.build_reader_prompt("Q?", "2026-01-01", "HISTORY")
    assert prompt.startswith("I will give you several history chats between you and a user.")
    assert "Answer the question step by step" in prompt
    assert "History Chats:\n\nHISTORY" in prompt
    assert "Current Date: 2026-01-01" in prompt
    assert prompt.endswith("Question: Q?\nAnswer (step by step):")


def test_reader_budget_constants() -> None:
    # 128000 - 800 - 1000 = 126200, o200k_base for gpt-4o.
    assert official_reader.MAX_RETRIEVAL_LENGTH == 126200
    assert official_reader.OPENAI_ENCODING == "o200k_base"
    assert official_reader.COT_GEN_LENGTH == 800


# --- round-expansion --------------------------------------------------------- #
def test_round_expansion_includes_following_turn() -> None:
    q = _question_with_sessions()
    # The first user turn of s_new is unit "<sid>#0"; its round = [turn0, turn1].
    new_sid = next(t.session_id for t in q.corpus if t.text.startswith("NEW: I adopted"))
    ctx = runner.assemble_context([f"{new_sid}#0"], q, top_k=10)
    chunk = json.loads(ctx.split("Session Content:\n", 1)[1].strip())
    assert chunk == [
        {"role": "user", "content": "NEW: I adopted a beagle."},
        {"role": "assistant", "content": "Nice, a beagle!"},
    ]


def test_round_expansion_capped_at_session_boundary() -> None:
    q = _question_with_sessions()
    # The LAST user turn of s_old (index 0, followed only by the assistant at 1).
    old_sid = next(t.session_id for t in q.corpus if t.text.startswith("OLD:"))
    ctx = runner.assemble_context([f"{old_sid}#0"], q, top_k=10)
    chunk = json.loads(ctx.split("Session Content:\n", 1)[1].strip())
    assert chunk[-1] == {"role": "assistant", "content": "Jazz is great."}


# --- chronological re-sort --------------------------------------------------- #
def test_chronological_resort_oldest_first() -> None:
    q = _question_with_sessions()
    new_sid = next(t.session_id for t in q.corpus if t.text.startswith("NEW: I adopted"))
    old_sid = next(t.session_id for t in q.corpus if t.text.startswith("OLD:"))
    # Rank NEW first, OLD second; assembly must re-sort to OLD (2026-01-01) first.
    ctx = runner.assemble_context([f"{new_sid}#0", f"{old_sid}#0"], q, top_k=10)
    assert ctx.index("2026-01-01") < ctx.index("2026-02-01")
    assert ctx.index("OLD:") < ctx.index("NEW:")


# --- JSON format ------------------------------------------------------------- #
def test_json_history_format() -> None:
    q = _question_with_sessions()
    old_sid = next(t.session_id for t in q.corpus if t.text.startswith("OLD:"))
    ctx = runner.assemble_context([f"{old_sid}#0"], q, top_k=10)
    assert "### Session 1:" in ctx
    assert "Session Date: 2026-01-01" in ctx
    assert "Session Content:" in ctx
    # The content is a JSON array of {role, content} dicts.
    payload = ctx.split("Session Content:\n", 1)[1].strip()
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    assert all(set(t) == {"role", "content"} for t in parsed)


# --- top-K by rank ----------------------------------------------------------- #
def test_top_k_keeps_first_by_rank() -> None:
    q = _question_with_sessions()
    new_sid = next(t.session_id for t in q.corpus if t.text.startswith("NEW: I adopted"))
    old_sid = next(t.session_id for t in q.corpus if t.text.startswith("OLD:"))
    # top_k=1 keeps only the first-ranked chunk (NEW), even though OLD is older.
    ctx = runner.assemble_context([f"{new_sid}#0", f"{old_sid}#0"], q, top_k=1)
    assert "NEW:" in ctx
    assert "OLD:" not in ctx


# --- tiktoken truncation ----------------------------------------------------- #
def test_tiktoken_truncation_boundary() -> None:
    rounds = [
        (f"2026-01-{i:02d}", [{"role": "user", "content": "word " * 50}]) for i in range(1, 20)
    ]
    budget = 40
    out = official_reader.assemble_history(rounds, top_k=100, max_retrieval_length=budget)
    enc = tiktoken.get_encoding(official_reader.OPENAI_ENCODING)
    assert len(enc.encode(out, allowed_special={"<|endoftext|>"})) <= budget


def test_no_truncation_under_budget() -> None:
    rounds = [("2026-01-01", [{"role": "user", "content": "short"}])]
    out = official_reader.assemble_history(rounds, top_k=10)
    assert "short" in out


# --- leakage guard on the reader path --------------------------------------- #
def test_assembled_context_carries_no_evidence_flag() -> None:
    q = _question_with_sessions()
    ids = [t.unit_id for t in q.corpus]
    ctx = runner.assemble_context(ids, q, top_k=10)
    # has_answer is popped at load; the assembled history never carries it.
    assert "has_answer" not in ctx
    # Assembly injects no gold beyond what a retrieved turn literally says: the only
    # occurrence of the gold token here is inside the real user turn "I adopted a
    # beagle." — never a separately-injected gold/answer field.
    for line in ctx.splitlines():
        if "beagle" in line:
            assert "I adopted a beagle" in line


def test_session_content_has_no_evidence_flag() -> None:
    q = _question_with_sessions()
    for session in q.sessions.values():
        for turn in session.turns:
            assert set(turn) == {"role", "content"}  # no has_answer / gold


def test_malformed_unit_id_skipped() -> None:
    q = _question_with_sessions()
    # An id with no '#'/non-numeric index is skipped, not crashed on.
    ctx = runner.assemble_context(["not-a-valid-id", "sess-x#notanumber"], q, top_k=10)
    assert ctx == ""
