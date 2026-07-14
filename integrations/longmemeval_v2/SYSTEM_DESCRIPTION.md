# Engrava 0.5.0

Engrava is an open-source memory system evaluated here as a LongMemEval-V2
memory backend. The submission uses the public PyPI package `engrava==0.5.0`
and the public repository <https://github.com/sovantica/engrava>.

## Memory Backend

The LongMemEval-V2 backend is implemented in `engrava_memory.py`.

Each inserted trajectory is converted into one Engrava observation per trajectory
state. A state observation includes the trajectory goal, outcome, state index,
URL, action, internal thought text when present, and accessibility-tree/text
observation. Screenshots are not inserted into Engrava; the backend returns text
memory context items.

At query time, the backend calls Engrava's public `search_hybrid()` API and
returns the retrieved state observations as LongMemEval-V2 text memory context
items. Reader, judge, prompts, scoring, token truncation, and leaderboard
packaging are all owned by the upstream LongMemEval-V2 harness.

## Configuration

The canonical operating point uses:

- system: `Engrava`
- version: `0.5.0`
- memory type: `engrava`
- embedding backend: OpenAI-compatible endpoint
- embedding model: `Qwen/Qwen3-Embedding-8B`
- reader model: `Qwen/Qwen3.5-9B`
- judge model: `gpt-5.2`
- operating point: `canonical`
- tier: `small`

The adapter includes a deterministic embedding backend only for local smoke
tests. It must not be used for publishable results.

## Limitations

This first operating point is text-only. It does not return screenshot memory
items. Questions may still include their original question image because that is
handled by the upstream reader harness, not by Engrava.
