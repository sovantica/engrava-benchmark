# Official LongMemEval scorer — upstream pin

The judge prompts and metric semantics used by this runner are the **official,
unmodified** LongMemEval evaluation contract.

- **Upstream repo:** https://github.com/xiaowu0162/LongMemEval
- **Pinned commit:** `9e0b455f4ef0e2ab8f2e582289761153549043fc`
- **Official files:**
  - `src/evaluation/evaluate_qa.py` — the GPT-4o judge + per-question-type prompts
    (`get_anscheck_prompt`).
  - `src/evaluation/print_qa_metrics.py` — overall-micro / task-macro / abstention
    aggregation.

This is the value recorded as `scorer_version` in every emitted result:
`longmemeval@9e0b455f4ef0e2ab8f2e582289761153549043fc`.

## How "unmodified" is honored

The runner does **not** fork or alter the official scoring logic:

- `evaluate_qa.py` in this directory carries the **verbatim** per-question-type judge
  prompt builder (`get_anscheck_prompt`) from the pinned commit — the prompt strings
  are the official scoring contract and must be byte-faithful, so the judge asks the
  exact official question. They are reproduced here (not re-authored) and marked as
  upstream-verbatim.
- The metric aggregation (`../scorer.py :: OfficialScorer.aggregate`) reproduces
  `print_qa_metrics.py`'s counting exactly: micro overall (flat per-question mean,
  abstention items included), unweighted macro over the 6 categories, and the
  abstention subset (`_abs` question-id suffix) as a cross-cutting group.

## Fully-faithful reproduction (maintainer / third party)

To score against the *exact* upstream scripts end-to-end (rather than this faithful
in-repo mirror), clone the pinned commit and run the official scripts on the runner's
hypothesis file:

```bash
git clone https://github.com/xiaowu0162/LongMemEval.git
cd LongMemEval && git checkout 9e0b455f4ef0e2ab8f2e582289761153549043fc
python src/evaluation/evaluate_qa.py gpt-4o <hypothesis_file> <reference_file>
python src/evaluation/print_qa_metrics.py <hypothesis_file> <reference_file>
```

The judge is `gpt-4o-2024-08-06` via `api.openai.com` **direct** (no broker), with
`temperature=0, max_tokens=10`; the label is `'yes' in completion.strip().lower()`.
