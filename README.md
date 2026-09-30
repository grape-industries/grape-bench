# Grape Benchmark

Grape (lexical search over a trigram index) against Vector RAG, with the same documents, questions and LLM. Only the way passages are found differs.

**Full report, with every answer and the method: https://grape-industries.github.io/grape-bench/**

## Results

Claude Haiku, thinking off, each suite run three times; means shown. Grape runs with its server reranker and sends 5 passages. A question counts when it is answered correctly or, if off-topic, refused.

| Set | Grape | Vector RAG |
|---|---|---|
| English documents, 28 questions | 28 / 28 | 24 / 28 (bge-small) |
| Large corpus: 300 articles, 18 MB, 29 questions | 27.7 / 29 | 29 / 29 (bge-small) |
| Gujarati documents and questions, 28 | 24.7 / 28 | 27 / 28 (e5-large) |
| Gujarati questions on English documents, 28 | 26.7 / 28 | 26 / 28 (e5-large) |
| ...against an English embedder | 28 / 28 | 5 / 28 (bge-small) |

| Cost and speed | Grape | Vector RAG |
|---|---|---|
| Ready after a change, 12 articles | 0.1 s | 98 s |
| Ready, 18 MB (CPU) | 1.1 s | 46 min |
| Cost per English question | $0.00148 | $0.00155 |
| Input tokens per question, Gujarati documents | 1,684 | 4,137 |

Retrieval alone (SQuAD 2.0 dev, 400 questions, no LLM): right paragraph ranked first 78.2% vs 69.8%, search time 10 ms vs 28 ms.

**Where Grape loses:**
- With 5 passages, a question whose two halves sit in different articles sometimes misses one half (Gujarati documents, 18 MB corpus).
- A question in another language than the documents costs one extra small LLM call to rewrite it, so it costs more than a multilingual Vector RAG.
- Ranking was tuned on the English and Gujarati-question sets; the other two were run only after tuning.
- The question sets are small and written by us, so read one-question gaps as ties.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and a signed-in [Claude Code](https://claude.com/claude-code) (`claude -p`). Documents and questions are committed in `corpus/` and `questions/` (Wikipedia and SQuAD, CC BY-SA 4.0), so there is nothing to download first.

Grape setup, the same as for any app that searches Grape:
1. Sign in at https://grape-industries.vercel.app/app and create a project for the corpus you want to run, then upload the files from `corpus/<set>/` (for example `corpus/fresh/`).
2. Copy your search-only API key from **API keys** (it starts with `grape_`). The project URL is `https://grape-industries.vercel.app/grape/` plus the **Project ID** shown on the project page (the search example there uses the same URL).

```sh
export GRAPE_KEY=grape_...
export GRAPE_PROJECT_URL=https://grape-industries.vercel.app/grape/<project-id>   # the project holding corpus/fresh/

F="--no-thinking --server-rerank --server-limit 5"
uv run bench.py --corpus fresh $F                                            # English
uv run bench.py --corpus fresh --questions questions_fresh_gu.json $F        # Gujarati questions, English docs
uv run bench.py --corpus fresh --questions questions_fresh_gu.json --embed bge-small $F   # ...against an English embedder
uv run bench.py --corpus fresh --only grape -n 5                             # quick check

# other corpora need their own project; point GRAPE_PROJECT_URL at it first
uv run bench.py --corpus gujarati $F                                         # Gujarati docs and questions
uv run bench.py --corpus large $F                                            # 300 articles, 18 MB
uv run tools/recall.py                                                       # retrieval only, no LLM (corpus/squad/)
```

`tools/fetch_corpus.py <set>` and `tools/fetch_squad.py` download the documents again.

- **LLM:** `bench.py` uses your Claude Code login through `claude -p` (Haiku by default, `--model` to change). Cost is the API list price; on a subscription the run uses plan quota. Each call spends about 3.5 s starting up, so compare the `api` column for model time only.
- **Output:** each run writes `results/<corpus>-<timestamp>.json` and rebuilds `site/` with `report.py`. The first run per corpus also embeds it for Vector RAG, cached in `cache/<corpus>/`.
- **Publishing:** `.github/workflows/pages.yml` rebuilds the report on every push that changes `results/`.

## Layout

```
bench.py     runs one suite
report.py    builds site/ from results/*.json
questions/   one JSON file per suite
corpus/      the documents
results/     raw results, one file per run
tools/       fetch_corpus.py, fetch_squad.py, recall.py, gate.py, export_site.py
assets/      chart, icons and font for the report
```
