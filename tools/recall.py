"""Retrieval recall on SQuAD v2 (no LLM, $0): does the gold paragraph land in the top-k?

Compares grape (keyword/BM25 + brief) and vector search, each with and without a cross-encoder
reranker over its top-20 candidates. Unanswerable questions show whether the top reranker score
can tell "not in the documents" apart. Run `uv run fetch_squad.py` first.
Usage: uv run recall.py [n_answerable] [n_unanswerable] [n_jina]
"""

import pathlib
import json
import os
import random
import statistics
import sys
import time

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # bench.py lives in the repo root
import bench  # noqa: E402

N_ANS, N_IMP, N_JINA = (int(a) for a in (sys.argv[1:] + ["400", "200", "150"][len(sys.argv) - 1:])[:3])
CANDS, KS = 20, (1, 5, 8)
RERANKERS = {"minilm": "Xenova/ms-marco-MiniLM-L-6-v2", "jina": "jinaai/jina-reranker-v2-base-multilingual"}
corpus, cache = bench.ROOT / "corpus" / "squad", bench.ROOT / "cache" / "squad"
cache.mkdir(parents=True, exist_ok=True)
bench.grape_session()

qs = json.loads((bench.QUESTION_DIR / "squad_questions.json").read_text())
rng = random.Random(0)
sample = rng.sample([q for q in qs if not q["impossible"]], N_ANS) + rng.sample([q for q in qs if q["impossible"]], N_IMP)
paras = [(f.name, i, t) for f in sorted(corpus.glob("*.txt")) for i, t in enumerate(f.read_text().splitlines(), 1)]

# grape: project per corpus, reindexed so the brief is current
gid, _ = bench.setup_grape(corpus, cache)
bench.http.post(f"{bench.GRAPE}/{gid}/index").raise_for_status()


def grape_cands(q):
    r = bench.http.post(f"{bench.GRAPE}/{gid}/search", json={"query": q, "limit": CANDS, "cutoff": 0})
    r.raise_for_status()
    return [(h["path"], h["line"], h["text"]) for h in r.json()["hits"]]


# vector: one embedding per paragraph (SQuAD's unit), cached
emb = TextEmbedding(bench.EMBED_MODEL)
vec_file = cache / "para_vecs.npy"
if vec_file.exists():
    vecs = np.load(vec_file)
else:
    t = time.perf_counter()
    vecs = np.array(list(emb.passage_embed([p[2] for p in paras])))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(vec_file, vecs)
    print(f"embedded {len(paras)} paragraphs in {time.perf_counter() - t:.0f}s")


def vector_cands(q):
    qv = np.array(list(emb.query_embed(q))[0])
    top = np.argsort(vecs @ (qv / np.linalg.norm(qv)))[::-1][:CANDS]
    return [paras[i] for i in top]


rerankers = {k: TextCrossEncoder(m, threads=os.cpu_count()) for k, m in RERANKERS.items()}


def rerank(model, q, cands):
    if not cands:
        return [], float("-inf")
    scores = list(model.rerank(q, [c[2] for c in cands], batch_size=CANDS))
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    return [cands[i] for i in order], max(scores)


# setup -> per-question {"rank": position of gold or None, "top": top rerank score}
results, timing = {}, {}
for ret_name, ret in (("grape", grape_cands), ("vector", vector_cands)):
    for n, item in enumerate(sample):
        t = time.perf_counter()
        cands = ret(item["q"])
        timing.setdefault(ret_name, []).append(time.perf_counter() - t)
        variants = [(ret_name, cands, None)]
        for rk, model in rerankers.items():
            if rk == "jina" and n >= N_JINA:
                continue
            t = time.perf_counter()
            ranked, top = rerank(model, item["q"], cands)
            timing.setdefault(f"{rk} rerank", []).append(time.perf_counter() - t)
            variants.append((f"{ret_name}+{rk}", ranked, top))
        for name, ranked, top in variants:
            gold = next((i for i, c in enumerate(ranked) if (c[0], c[1]) == (item["file"], item["line"])), None)
            results.setdefault(name, []).append({"rank": gold, "top": top, "impossible": item["impossible"], "n": n})
        if n % 100 == 99:
            print(f"  {ret_name}: {n + 1}/{len(sample)}", flush=True)

print(f"\nSQuAD v2 dev, {len(paras)} paragraphs from {len(list(corpus.glob('*.txt')))} articles; "
      f"{N_ANS} answerable questions (jina: first {N_JINA} of the sample)\n")
print(f"{'setup':<16}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'R@' + str(CANDS):>8}{'n':>6}")
for name, rs in results.items():
    ans = [r for r in rs if not r["impossible"]]
    row = "".join(f"{sum(r['rank'] is not None and r['rank'] < k for r in ans) / len(ans):>8.1%}" for k in KS)
    print(f"{name:<16}{row}{sum(r['rank'] is not None for r in ans) / len(ans):>8.1%}{len(ans):>6}")

print("\n'Not in the documents' signal: top reranker score, answerable vs unanswerable")
for name, rs in results.items():
    if rs[0]["top"] is None:
        continue
    a = [r["top"] for r in rs if not r["impossible"]]
    u = [r["top"] for r in rs if r["impossible"]]
    if not u:
        continue
    # best single threshold: answerable above, unanswerable below
    best = max((sum(x >= th for x in a) + sum(x < th for x in u)) / (len(a) + len(u)) for th in a + u)
    print(f"  {name:<16} median answerable {statistics.median(a):6.2f}  unanswerable {statistics.median(u):6.2f}"
          f"  best threshold accuracy {best:.1%}")

print("\nLatency per question (CPU):")
for k, v in timing.items():
    print(f"  {k:<14} {statistics.mean(v) * 1000:7.0f} ms")
