# RAG Evaluation

This project includes a lightweight retrieval evaluation script and a small JSONL dataset scaffold.

## Dataset

Edit `data/rag_eval.jsonl` and fill in:
- `expected_repo_ids`: list of repo IDs expected for repo queries.
- `expected_profile_sources`: list of profile document `source` values expected for profile queries.
- `expect_repo_empty`: set `true` when you expect no repo matches.
- `expect_profile_empty`: set `true` when you expect no profile matches.
- `expected_plan`: optional dict of plan flags you expect the planner to set.
- `expected_plan_any`: optional list of acceptable plan dicts.

Entries with empty expected lists are skipped unless you set `expect_repo_empty` or
`expect_profile_empty`.

## Run Retrieval Eval

```bash
python scripts/eval_rag.py --print-items
```

Optional controls:

```bash
python scripts/eval_rag.py --repo-k 5 --profile-k 5
```

## Reranking

Reranking is enabled by default to improve precision. Disable it for baseline runs:

```bash
export RERANK_ENABLED=false
```

Optional knobs:

```bash
export RERANK_MAX_CANDIDATES=8
export MAX_RERANK_TEXT_CHARS=320
export RERANK_INCLUDE_SNIPPETS=false
```

Plan checks (requires OpenAI access since it runs the planner):

```bash
python scripts/eval_rag.py --check-plan
```

Print mismatched planner outputs:

```bash
python scripts/eval_rag.py --check-plan --print-plan
```

## LangSmith Tracing

Set these environment variables to trace runs:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_PROJECT=portfolio-rag-eval
```

If tracing is enabled, the script prints a notice when it starts.

## RAGAS (Optional)

Run with:

```bash
python scripts/eval_rag.py --ragas
```

RAGAS is optional and requires additional setup. If not installed or not configured
with reference answers/contexts, the script will skip RAGAS gracefully.
