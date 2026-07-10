"""OpenAI-compatible reader + judge for the LongMemEval runner.

These are concrete implementations of the runner's ``Reader`` / ``Judge`` Protocol
seams (``run.py``). They talk to any OpenAI-compatible Chat Completions API via the
official ``openai`` Python SDK — ``api.openai.com`` for the canonical headline, or a
local OpenAI-compatible server (e.g. Ollama at ``http://<host>:11434/v1``) for a
free, offline-capable run. The adapter never touches them — equal footing is
preserved because the reader/judge live entirely on the runner side.

Cost note
---------
Constructing these objects is free; only :meth:`OpenAIReader.answer` and
:meth:`OpenAIJudge.score` make API calls. The free smoke path (``mock_models.py``)
is used for local end-to-end testing without any LLM call.

Endpoint handling
-----------------
The ``endpoint`` may carry a scheme. ``http://``/``https://`` endpoints are used as
the base URL directly (``/v1`` appended if missing), so a local Ollama endpoint
``http://<host>:11434`` resolves to ``http://<host>:11434/v1``. A bare host such as
``api.openai.com`` keeps the default ``https://<host>/v1`` — the canonical headline
is unchanged.

Key handling
------------
The API key is read from an environment variable (default ``OPENAI_API_KEY``) and
is **never** hard-coded or logged. When the env var is unset, a dummy non-empty
placeholder (``"ollama"``) is used so the SDK initialises against a local server
that ignores auth — this is never a real credential.

D9 (canonical headline)
-----------------------
The canonical headline runs the **reader** at ``api.openai.com`` with snapshot
``gpt-4o-2024-08-06`` (temperature 0.0) so the row lands in the canonical
comparability segment; the **judge** is OpenAI-direct ``gpt-4o-2024-08-06``
regardless. Both are config-driven (``config/default.json``); these classes just
honor what the config declares. A local-LLM run records its actual endpoint/model,
so it correctly lands in a non-canonical segment.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from runners.longmemeval import official_reader

# Dummy non-empty placeholder used when no API key env var is set (e.g. a local
# Ollama server that ignores auth). NOT a real credential.
_LOCAL_DUMMY_KEY = "ollama"


def resolve_base_url(endpoint: str) -> str:
    """Resolve an ``endpoint`` to an OpenAI-compatible base URL.

    A scheme-bearing endpoint (``http://`` / ``https://``) is used as the base URL
    directly, with ``/v1`` appended if it is not already the path. A bare host gets
    the default ``https://<host>/v1`` (the canonical ``api.openai.com`` is unchanged).

    Args:
        endpoint: A bare host (``api.openai.com``) or a full URL
            (``http://host:11434`` / ``http://host:11434/v1``).

    Returns:
        The base URL to pass to the OpenAI SDK.

    """
    if endpoint.startswith(("http://", "https://")):
        base = endpoint.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base
    return f"https://{endpoint}/v1"


def make_client(endpoint: str, api_key_env: str) -> Any:  # noqa: ANN401 - external SDK type
    """Build an OpenAI SDK client for an endpoint, with a local-friendly key fallback.

    Args:
        endpoint: The endpoint (bare host or full URL); see :func:`resolve_base_url`.
        api_key_env: Env var holding the API key. If unset, a dummy non-empty
            placeholder is used (a local server ignores auth). Never a real key,
            never logged.

    Returns:
        A configured ``openai.OpenAI`` client.

    """
    from openai import OpenAI  # noqa: PLC0415 - lazy: only when a run actually executes

    api_key = os.environ.get(api_key_env) or _LOCAL_DUMMY_KEY
    return OpenAI(api_key=api_key, base_url=resolve_base_url(endpoint))


class OpenAIReader:
    """Reader that answers from assembled context via OpenAI-direct chat.

    Implements the runner's ``Reader`` Protocol.
    """

    def __init__(
        self,
        *,
        model_snapshot: str,
        endpoint: str,
        sampling: Mapping[str, Any],
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        """Initialize the reader.

        Args:
            model_snapshot: The exact model snapshot (e.g. ``gpt-4o-2024-08-06``).
            endpoint: The API host (canonical headline requires ``api.openai.com``).
            sampling: Sampling params (e.g. ``{"temperature": 0.0}``). An optional
                ``max_tokens`` here overrides the reader generation length; when
                absent the official 800 is used.
            api_key_env: Env var holding the API key (never hard-coded).

        """
        self._model = model_snapshot
        self._endpoint = endpoint
        self._sampling = dict(sampling)
        self._api_key_env = api_key_env
        self._client: Any = None

    def _ensure_client(self) -> Any:  # noqa: ANN401 - external SDK type
        if self._client is None:
            self._client = make_client(self._endpoint, self._api_key_env)
        return self._client

    def answer(self, question: str, context: str, *, question_date: str = "") -> str:
        """Answer ``question`` from the assembled history via a paid OpenAI call.

        Uses the official LongMemEval reader prompt (a single user message; no
        system role), exactly as ``run_generation.py`` calls the API. The generation
        length is the ``sampling.max_tokens`` value declared in the reader config when
        present, else the official cot generation length
        (:data:`official_reader.COT_GEN_LENGTH`, 800). The canonical config declares
        no ``max_tokens``, so the canonical headline is unchanged; a bring-your-own
        reader can raise it (e.g. a reasoning model that needs more room).

        The visible answer is read from ``message.content``. Some reasoning models
        return their chain-of-thought in a separate field and leave ``content`` empty
        or ``None``; that is handled without special-casing any model name — an empty
        content simply yields ``""`` (no crash).

        Args:
            question: The question text.
            context: The official assembled history string.
            question_date: The question's date (the official ``Current Date`` field).

        Returns:
            The model's answer string (stripped, as upstream does); ``""`` if the
            model returned empty/absent content.

        """
        prompt = official_reader.build_reader_prompt(question, question_date, context)
        # max_tokens is config-driven (sampling.max_tokens) with the official 800 as
        # the default; pop it out of the sampling kwargs so it is not passed twice.
        sampling = dict(self._sampling)
        max_tokens = sampling.pop("max_tokens", official_reader.COT_GEN_LENGTH)
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **sampling,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class OpenAIJudge:
    """Judge that scores a model answer against the gold answer (OpenAI-direct).

    Implements the runner's ``Judge`` Protocol. The judgment delegates the actual
    correctness rubric to the official scorer's judge prompt
    (``scorer.official_judge_prompt``) so the wording is the unmodified official
    one; this class only performs the API call and parses yes/no.
    """

    def __init__(
        self,
        *,
        model_snapshot: str,
        endpoint: str,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        """Initialize the judge.

        Args:
            model_snapshot: The judge snapshot (canonical: ``gpt-4o-2024-08-06``).
            endpoint: The API host (canonical: ``api.openai.com``).
            api_key_env: Env var holding the API key (never hard-coded).

        """
        self._model = model_snapshot
        self._endpoint = endpoint
        self._api_key_env = api_key_env
        self._client: Any = None

    def _ensure_client(self) -> Any:  # noqa: ANN401 - external SDK type
        if self._client is None:
            self._client = make_client(self._endpoint, self._api_key_env)
        return self._client

    def score(
        self,
        question: str,
        gold: str,
        answer: str,
        *,
        question_type: str,
        question_id: str,
    ) -> bool:
        """Judge whether ``answer`` is correct via the official judge prompt.

        Args:
            question: The question text.
            gold: The gold answer.
            answer: The model's answer to judge.
            question_type: The official question type (selects the official
                judge prompt template).
            question_id: The question id (detects abstention items).

        Returns:
            ``True`` iff the judge rules the answer correct.

        """
        from runners.longmemeval import scorer  # noqa: PLC0415 - avoid import cycle

        prompt = scorer.official_judge_prompt(
            question=question,
            gold=gold,
            answer=answer,
            question_type=question_type,
            question_id=question_id,
        )
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        # Official label rule: 'yes' in the completion (case-insensitive).
        verdict = (response.choices[0].message.content or "").strip().lower()
        return "yes" in verdict
