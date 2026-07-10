# Official LongMemEval reader — upstream pin

The reader prompt and context-assembly used by this runner are the **official,
unmodified** LongMemEval generation protocol.

- **Upstream repo:** https://github.com/xiaowu0162/LongMemEval
- **Pinned commit:** `9e0b455f4ef0e2ab8f2e582289761153549043fc` (same commit as the scorer).
- **Official files:**
  - `src/generation/run_generation.py` — the reader (answer/hypothesis generation):
    the CoT prompt, the flat-turn round-expansion, the chronological re-sort, the JSON
    history formatting, and the tiktoken truncation.
  - `src/generation/run_generation.sh` — the official wrapper / CLI defaults.

Replicated verbatim in `official_reader.py`; the round-expansion lives in
`run.py :: assemble_context` (it needs runner-internal session content). Recorded as
`reader_version` in every emitted result:
`longmemeval@9e0b455f4ef0e2ab8f2e582289761153549043fc`.

## What is replicated (byte-faithful)

- **Reader prompt (default).** The official default reading method `con` resolves to
  `--cot true` (the step-by-step extract-then-reason prompt). A single user message,
  no system role, no per-question-type variation. `COT_READER_PROMPT_TEMPLATE` in
  `official_reader.py` is the verbatim string from `run_generation.py`, filled
  `.format(history_string, question_date, question)`. (The separate chain-of-note
  extraction pass — `--con true`, only reached by the non-default `con-separate` — is
  not the LongMemEval-S default and is not replicated.)
- **Context assembly** (`flat-turn`, `history_format=json`):
  1. **round-expansion** — each retrieved user turn expands to `[turn, next_turn]`
     (the user turn + its assistant reply), capped at the session boundary.
  2. **top-K by rank** then **chronological re-sort** ascending by session date.
  3. **JSON formatting** — each chunk is `"\n" + json.dumps(round_turns)` wrapped as
     `### Session {i+1}:\nSession Date: {date}\nSession Content:\n{sess}\n`.
  4. **tiktoken truncation** — encode with `o200k_base`; if over budget keep the
     **first** `MAX_RETRIEVAL_LENGTH` tokens (head-keep, drops the newest sessions).
     Budget = `128000 - 800 - 1000 = 126200` (model window − cot gen length − 1000).
- **Model call** — `temperature=0`, `n=1`, `max_tokens=800` (cot), single user
  message, output `.strip()`-ed. Model/endpoint are config-driven (`gpt-4o-2024-08-06`
  via `api.openai.com` per the canonical-headline constraint).

## Equal-footing / leakage guard

The reader and its context assembly are **runner-owned and uniform** for every
system — an adapter never touches them. The assembly consumes the adapter's ranked
neutral unit ids plus runner-internal session content (`Question.sessions`, which
carries only `role` + `content`; the evidence `has_answer` flag is popped at load,
exactly as upstream does). It reads no gold / answer / evidence.

## Fully-faithful reproduction (maintainer / third party)

To generate against the *exact* upstream script, clone the pinned commit:

```bash
git clone https://github.com/xiaowu0162/LongMemEval.git
cd LongMemEval && git checkout 9e0b455f4ef0e2ab8f2e582289761153549043fc
cd src/generation
bash run_generation.sh RETRIEVAL_LOG_FILE gpt-4o flat-<retriever>-turn 1000 json false con
```
(`con` → `--cot true`; `history_format=json`; `useronly=false`.)
