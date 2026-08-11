# runners/longmemeval_v2/

A thin wrapper that runs **LongMemEval-V2 through its own upstream harness**, with this
repository's run-mode surface in front of it.

The distinction matters for reading any number that comes out of here: the retrieval is
Engrava's, but the questions, the haystacks, the prompt assembly and the scoring all belong to
the upstream project. This directory does not reimplement them, and a result produced here is
comparable to other LongMemEval-V2 results only to the extent the upstream harness makes it so.

## Files

- `run.py` — resolves `--mode`, builds the upstream harness command, runs it, and (in
  `retrieval` mode) extracts a ranked retrieval log from what the harness wrote.
- `retrieval_log.py` — reduces the harness's `prompt_rows.jsonl` to the benchmark-agnostic
  `{question_id: [item_key, ...]}` shape that `runners/retrieval_diff.py` compares.

## Requirements

The upstream LongMemEval-V2 checkout is **not** vendored here. Point the runner at yours:

```bash
export LME_V2_HARNESS_DIR=/path/to/LongMemEval-V2   # or pass --harness-dir
```

Run it with that checkout's own environment. It carries multimodal prompt-build dependencies this
repository does not declare, so a run from an environment lacking them fails during prompt
assembly — after the embedding work is already done and paid for. The runner puts this repository
on `sys.path` itself, so it does not need to be installed into that environment.

**Install the memory shim into the harness first.** The harness loads Engrava through
`memory_modules/engrava_memory.py`, which this repository owns but does not vendor into your
checkout — see [the integration's README](../../integrations/longmemeval_v2/README.md) for the
copy step. A stale copy is the failure to watch for: it fails deep inside a run, after the
retrieval work, with an error about the memory backend rather than about the copy.

## Modes

The modes mean what they mean in `runners/longmemeval/README.md`. Three notes specific to this
runner:

- **Cheap does not mean quick here.** The mode table in that README reads cost as LLM spend, and on
  that axis `smoke` and `plumbing` are free. On wall clock they are not: the upstream harness loads
  the whole `trajectories.jsonl` on every run — currently about **1.2 GB** — regardless of how many
  questions you asked for. A two-question `smoke` therefore takes minutes, and on a loaded machine
  considerably longer. It is a wiring check, not a fast one; if it seems to hang, check the load
  before assuming it has.

- **`retrieval`** is the mode this wrapper exists for: real embeddings, no reader, and a ranked
  retrieval log written next to the harness output. It is how an Engrava change is screened for a
  retrieval regression without paying for a reader.
- **`score`** hands the reader and judge flags through to the upstream harness, which owns them.
  Only the env-var *names* are forwarded, never key values.

```bash
# screen a retrieval change against a stored baseline
python runners/longmemeval_v2/run.py --mode retrieval --domain web \
    --output-dir runs/web-candidate --baseline runs/web-reference/retrieval_log.json
```

The command exits non-zero when the diff verdict is RED, so it can gate a pipeline directly.

## What this runner refuses to do

- **Extract an artifact it did not produce.** The harness writes into the output directory and a
  zero exit does not prove it rewrote anything; a `prompt_rows.jsonl` older than the run is
  rejected rather than screened as this run's retrieval.
- **Turn a missing field into a result.** A row without `memory_context` is an error, not a
  question that retrieved nothing, and question ids are required to be strings rather than
  coerced.
- **Emit from a non-official mode.** Only `score` may, and the mode table rejects any combination
  that claims otherwise.

## Retrieval identity

An item's identity is its parsed `Trajectory ID` + `State index`; when both headers are absent the
item falls back to a content hash. Requiring both keeps two distinct states of one trajectory from
collapsing onto the same key, which would make a real ranking change invisible to the diff.
