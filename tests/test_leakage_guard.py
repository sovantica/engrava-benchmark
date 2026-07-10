"""Tests for the equal-footing leakage guard + dataset-loading robustness.

The decisive invariant: nothing an adapter can reach may reveal which turns are
gold (evidence). The official corpus id encodes evidence (``answer`` kept vs
``noans``); the adapter must only ever see a NEUTRAL, opaque unit id. These tests
build a dataset with realistic answer-bearing session ids and assert no evidence
signal crosses the adapter boundary, while the runner keeps the official mapping
internally.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from adapters.base import CorpusTurn
from runners.longmemeval import run as runner

if TYPE_CHECKING:
    from pathlib import Path

# A haystack with the official answer-bearing session-id convention: an `answer_*`
# session whose evidence turn (has_answer) keeps `answer` and whose non-evidence
# turns become `noans` in the OFFICIAL id. The neutral adapter-facing id must show
# none of that.
_DATASET = [
    {
        "question_id": "q_evidence",
        "question_type": "single-session-user",
        "question": "Which city did the user move to?",
        "answer": "Porto",
        "haystack_dates": ["2026-01-01"],
        "haystack_session_ids": ["answer_abc123"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I just moved to Porto.", "has_answer": True},
                {"role": "assistant", "content": "Enjoy Porto!"},
                {"role": "user", "content": "The weather is mild.", "has_answer": False},
            ]
        ],
    }
]


# A multi-session haystack whose answer-bearing session is NOT first, used to prove
# the adapter-facing id is independent of haystack position (ordinal).
def _multi_session_item(session_order: list[str]) -> dict:
    bodies = {
        "answer_evidence": [
            {"role": "user", "content": "I just moved to Porto.", "has_answer": True},
            {"role": "user", "content": "The weather is mild.", "has_answer": False},
        ],
        "plain_a": [{"role": "user", "content": "I like jazz."}],
        "plain_b": [{"role": "user", "content": "I run on Sundays."}],
    }
    return {
        "question_id": "q_multi",
        "question_type": "multi-session",
        "question": "Where did the user move?",
        "answer": "Porto",
        "haystack_dates": ["2026-01-01"] * len(session_order),
        "haystack_session_ids": list(session_order),
        "haystack_sessions": [bodies[s] for s in session_order],
    }


def _write(tmp_path: Path, item: dict | None = None) -> Path:
    path = tmp_path / "haystack.json"
    path.write_text(json.dumps([item] if item else _DATASET), encoding="utf-8")
    return path


def test_adapter_facing_ids_carry_no_evidence_signal(tmp_path: Path) -> None:
    questions = runner.load_questions(_write(tmp_path))
    corpus = questions[0].corpus
    for turn in corpus:
        assert "answer" not in turn.unit_id
        assert "noans" not in turn.unit_id
        # The session surrogate must also leak nothing from the real `answer_*` id.
        assert "answer" not in turn.session_id
        assert "noans" not in turn.session_id


def test_adapter_facing_ids_carry_no_haystack_ordinal(tmp_path: Path) -> None:
    """Shuffling the haystack must not change a session's adapter-facing ids.

    A positional id (e.g. ``s{haystack_index}#...``) would change when the same
    session moves position; a session-identity id does not. This closes the
    order-correlation channel: an adapter cannot infer evidence placement from the
    id.
    """
    order_a = ["answer_evidence", "plain_a", "plain_b"]
    order_b = ["plain_a", "plain_b", "answer_evidence"]  # evidence moved to the end

    qa = runner.load_questions(_write(tmp_path, _multi_session_item(order_a)))[0]
    qb = runner.load_questions(_write(tmp_path, _multi_session_item(order_b)))[0]

    def ids_by_text(q: object) -> dict[str, str]:
        return {t.text: t.unit_id for t in q.corpus}  # type: ignore[attr-defined]

    a_ids, b_ids = ids_by_text(qa), ids_by_text(qb)
    # The same session content yields the same adapter id regardless of position.
    assert a_ids == b_ids
    # And nothing in any id encodes the haystack ordinal (0/1/2) of the evidence
    # session: its id is identical across the two orderings.
    evidence_text = "I just moved to Porto."
    assert a_ids[evidence_text] == b_ids[evidence_text]


def test_neutral_session_id_is_content_derived() -> None:
    turns = [
        {"role": "user", "content": "I just moved to Porto."},
        {"role": "assistant", "content": "Enjoy Porto!"},
    ]
    # Stable: same content -> same id.
    assert runner._neutral_session_id(turns) == runner._neutral_session_id(list(turns))
    # Distinct: different content -> different id.
    other = [{"role": "user", "content": "I run on Sundays."}]
    assert runner._neutral_session_id(turns) != runner._neutral_session_id(other)
    # Opaque, evidence-free shape.
    nid = runner._neutral_session_id(turns)
    assert nid.startswith("sess-")
    assert "answer" not in nid
    assert "noans" not in nid
    # Full sha256 hex digest (64 chars) — not truncated, so no birthday collisions.
    assert len(nid) == len("sess-") + 64


def test_neutral_session_id_is_role_independent() -> None:
    """The same turns hash identically whether marked as evidence or a distractor.

    Content derivation ignores ``has_answer`` entirely, so a session presented as
    the evidence session and the SAME session presented as a plain distractor get
    the same id — the evidence role does not leak through the id.
    """
    as_evidence = [
        {"role": "user", "content": "I just moved to Porto.", "has_answer": True},
        {"role": "user", "content": "The weather is mild.", "has_answer": False},
    ]
    as_distractor = [
        {"role": "user", "content": "I just moved to Porto."},
        {"role": "user", "content": "The weather is mild."},
    ]
    assert runner._neutral_session_id(as_evidence) == runner._neutral_session_id(as_distractor)


def test_neutral_session_id_no_delimiter_collision() -> None:
    # Length-prefixed serialisation: two distinct turn splittings must not collide.
    a = [{"role": "user", "content": "ab"}, {"role": "user", "content": "c"}]
    b = [{"role": "user", "content": "a"}, {"role": "user", "content": "bc"}]
    assert runner._neutral_session_id(a) != runner._neutral_session_id(b)


def test_id_string_marker_does_not_collide_or_leak(tmp_path: Path) -> None:
    """Regression: a distractor whose session-id string contains the marker.

    The old substring-normalising derivation could (a) over-collapse a genuinely
    different session whose id contained ``answer``/``noans`` in its remainder and
    (b) collide a normalised ``answer_*`` with a real ``session_*``. Content
    derivation ignores the id string entirely, so distinct content stays distinct
    and the marker never reaches the adapter id.
    """
    item = {
        "question_id": "q_marker",
        "question_type": "multi-session",
        "question": "?",
        "answer": "x",
        "haystack_dates": ["2026-01-01", "2026-01-02", "2026-01-03"],
        # Distinct content, but ids that would confuse a string-parsing derivation:
        "haystack_session_ids": ["answer_real", "noans_distractor", "session_real"],
        "haystack_sessions": [
            [{"role": "user", "content": "alpha", "has_answer": True}],
            [{"role": "user", "content": "beta"}],
            [{"role": "user", "content": "gamma"}],
        ],
    }
    q = runner.load_questions(_write(tmp_path, item))[0]
    session_ids = {t.session_id for t in q.corpus}
    # Three distinct-content sessions -> three distinct adapter session ids.
    assert len(session_ids) == 3
    # And no id-string marker leaked to any adapter-facing id.
    for sid in session_ids:
        assert "answer" not in sid
        assert "noans" not in sid


def test_official_ids_kept_runner_internal(tmp_path: Path) -> None:
    questions = runner.load_questions(_write(tmp_path))
    q = questions[0]
    # The runner-internal map still holds the OFFICIAL evidence-encoding ids.
    official_ids = set(q.id_map.values())
    assert any("answer" in oid for oid in official_ids)  # the evidence turn
    assert any("noans" in oid for oid in official_ids)  # the rewritten non-evidence turn
    # Every neutral id resolves to an official id; the map is the only place the
    # official id lives.
    assert set(q.id_map) == {t.unit_id for t in q.corpus}


def test_neutral_ids_unique_within_a_question(tmp_path: Path) -> None:
    questions = runner.load_questions(_write(tmp_path))
    ids = [t.unit_id for t in questions[0].corpus]
    assert len(ids) == len(set(ids))


def test_corpus_turn_has_no_gold_field() -> None:
    names = {f.name for f in dataclasses.fields(CorpusTurn)}
    assert names.isdisjoint({"has_answer", "answer", "gold", "is_evidence", "label"})


def test_question_gold_not_on_corpus(tmp_path: Path) -> None:
    # Gold lives on Question (runner-internal), never on a CorpusTurn the adapter sees.
    questions = runner.load_questions(_write(tmp_path))
    assert questions[0].answer == "Porto"
    assert not any(hasattr(t, "answer") for t in questions[0].corpus)


# --- dataset-loading robustness (P2) ---------------------------------------- #
def test_limit_zero_yields_no_questions(tmp_path: Path) -> None:
    assert runner.load_questions(_write(tmp_path), limit=0) == []


def test_limit_none_loads_all(tmp_path: Path) -> None:
    assert len(runner.load_questions(_write(tmp_path), limit=None)) == 1


def test_session_length_mismatch_raises(tmp_path: Path) -> None:
    bad = [
        {
            "question_id": "q_bad",
            "question_type": "single-session-user",
            "question": "?",
            "answer": "x",
            "haystack_dates": ["2026-01-01"],
            "haystack_session_ids": ["s0", "s1"],  # 2 ids
            "haystack_sessions": [[{"role": "user", "content": "hi"}]],  # 1 session
        }
    ]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(runner.DatasetError):
        runner.load_questions(path)


def test_byte_identical_sessions_ingest_both_first_occurrence_date(tmp_path: Path) -> None:
    """Byte-identical duplicate sessions LOAD, ingest both, keep first-occ date.

    Byte-identical duplicate sessions are a legitimate LongMemEval-S case (~13
    questions). ``load_questions`` INGESTS BOTH (both turns appended to the corpus,
    mirroring the official flat-list so FTS corpus statistics match), while
    ``id_map`` and the assembly session content stay FIRST-occurrence. The two
    haystack occurrences may carry different dates; the surviving date must be the
    first. No ``DatasetError`` is raised.
    """
    same_session = [{"role": "user", "content": "identical content"}]
    item = {
        "question_id": "q_dup",
        "question_type": "multi-session",
        "question": "?",
        "answer": "x",
        "haystack_dates": ["2026-01-01", "2026-01-02"],  # different dates
        "haystack_session_ids": ["s_a", "s_b"],  # distinct ids, identical content
        "haystack_sessions": [list(same_session), list(same_session)],
    }
    path = tmp_path / "dup.json"
    path.write_text(json.dumps([item]), encoding="utf-8")

    questions = runner.load_questions(path)

    assert len(questions) == 1
    q = questions[0]
    unit_ids = [t.unit_id for t in q.corpus]
    # Ingest-both: the corpus carries BOTH copies of the duplicate turn.
    assert len(unit_ids) == 2
    assert len(set(unit_ids)) == 1  # ...sharing one content-derived unit_id
    # id_map stays first-occurrence: one entry for the collapsed unit_id.
    assert list(q.id_map) == [unit_ids[0]]
    # Session content (and its date) is FIRST-occurrence, not overwritten by the dup.
    neutral_sid = q.corpus[0].session_id
    assert q.sessions[neutral_sid].date == "2026-01-01"
