"""Calibrate a no-LLM "off-topic" gate: grape coverage + reranker top score (no LLM, $0).

A gate rule rejects a question when every chosen signal is below its threshold. We want the
rule that rejects the most out-of-scope questions while rejecting zero in-scope ones.
Questions with no English function words (e.g. Spanish, Hindi) are never gated: the rerankers
here can't judge them, and grape's route call handles them.
Usage: uv run gate.py
"""

import pathlib
import sys
import itertools
import json
import os
import random
import re
import time

from fastembed.rerank.cross_encoder import TextCrossEncoder

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # bench.py lives in the repo root
import bench  # noqa: E402

SETS = [("space", "questions.json"), ("mixed", "questions_mixed.json"), ("fresh", "questions_fresh.json")]
SQUAD_ANS, SQUAD_IMP = 400, 200
MAX_SQUAD_REJECT = 0.01  # at most 1% of answerable SQuAD questions may be wrongly rejected
english_like = bench.english_like


def grape_cands(gid, q):
    body = {"query": q, "limit": bench.CANDIDATES, "cutoff": 0, "snippet": bench.SNIPPET}
    r = bench.http.post(f"{bench.GRAPE}/{gid}/search", json=body)
    r.raise_for_status()
    d = r.json()
    return d["coverage"], [h["text"] for h in d["hits"]]


bench.grape_session()
models = {k: TextCrossEncoder(m, threads=os.cpu_count()) for k, m in bench.RERANKERS.items()}
lat = {k: [] for k in models}


def top_score(name, q, texts):
    if not texts:
        return -99.0
    t = time.perf_counter()
    s = max(models[name].rerank(q, texts, batch_size=len(texts)))
    lat[name].append(time.perf_counter() - t)
    return s


score_file = bench.ROOT / "cache" / "gate_scores.json"
if score_file.exists():
    rows, squad_rows = json.loads(score_file.read_text())
else:
    rows = []  # one per question: label "in" (must never be rejected) or "out" (should be rejected)
    for corpus, qfile in SETS:
        gid = bench.setup_grape(bench.ROOT / "corpus" / corpus, bench.ROOT / "cache" / corpus)[0]
        for item in json.loads((bench.QUESTION_DIR / qfile).read_text()):
            cov, texts = grape_cands(gid, item["q"])
            rows.append({
                "set": corpus, "q": item["q"], "label": "out" if item.get("none") else "in", "en": english_like(item["q"]),
                "cov": cov, "minilm": top_score("minilm", item["q"], texts), "jina": top_score("jina", item["q"], texts),
            })
        print(f"  scored {corpus}", flush=True)

    squad = json.loads((bench.QUESTION_DIR / "squad_questions.json").read_text())
    rng = random.Random(0)
    sq = rng.sample([q for q in squad if not q["impossible"]], SQUAD_ANS) + rng.sample([q for q in squad if q["impossible"]], SQUAD_IMP)
    sgid = bench.setup_grape(bench.ROOT / "corpus" / "squad", bench.ROOT / "cache" / "squad")[0]
    squad_rows = []
    for item in sq:
        cov, texts = grape_cands(sgid, item["q"])
        squad_rows.append({"cov": cov, "minilm": top_score("minilm", item["q"], texts), "impossible": item["impossible"],
                           "en": english_like(item["q"])})
    print("  scored squad\n")

    score_file.write_text(json.dumps([rows, squad_rows]))

RULES = {"coverage": ["cov"], "minilm": ["minilm"], "coverage+minilm": ["cov", "minilm"],
         "jina": ["jina"], "coverage+minilm+jina (both models)": ["cov", "minilm", "jina"]}


def rejects(r, keys, th):
    return r["en"] and all(r[k] < th[k] for k in keys)


ins = [r for r in rows if r["label"] == "in"]
outs = [r for r in rows if r["label"] == "out"]
print(f"our sets: {len(ins)} in-scope (incl. non-English), {len(outs)} out-of-scope\n")
print(f"{'rule':<36}{'out rejected':>14}{'in rejected':>13}{'SQuAD ans rej':>15}  thresholds")
best = {}
for name, keys in RULES.items():
    # candidate thresholds: just above each in-scope value, so no in-scope question falls below
    grids = [[float("-inf")] + sorted({r[k] for r in ins if r["en"]}) for k in keys]  # -inf = never reject
    top = None
    for combo in itertools.product(*grids):
        th = dict(zip(keys, combo))  # reject only strictly below the lowest in-scope value -> 0 false rejects
        if any(rejects(r, keys, th) for r in ins):
            continue
        if "jina" not in keys and sum(rejects(r, keys, th) for r in squad_rows if not r["impossible"]) > MAX_SQUAD_REJECT * SQUAD_ANS:
            continue
        caught = sum(rejects(r, keys, th) for r in outs)
        if top is None or caught > top[0]:
            top = (caught, th)
    caught, th = top
    best[name] = th
    sq_rej = (f"{sum(rejects(r, keys, th) for r in squad_rows if not r['impossible']) / SQUAD_ANS:>14.1%}"
              if "jina" not in keys else f"{'n/a':>14}")
    print(f"{name:<36}{f'{caught}/{len(outs)}':>14}{f'0/{len(ins)}':>13}{sq_rej}  "
          + ", ".join(f"{k}<{v:.2f}" for k, v in th.items()))

print("\nPer out-of-scope question (rejected by coverage+minilm?):")
th = best["coverage+minilm"]
for r in outs:
    print(f"  {'REJECT' if rejects(r, ['cov', 'minilm'], th) else 'to LLM':<7} cov={r['cov']:.2f} minilm={r['minilm']:6.2f}"
          f" jina={r['jina']:5.2f}  [{r['set']}] {r['q']}")
if any(lat.values()):
    print("\nLatency per question:", ", ".join(f"{k} {sum(v) / len(v) * 1000:.0f} ms" for k, v in lat.items()))
(bench.ROOT / "cache" / "gate.json").write_text(json.dumps(best["coverage+minilm"]))
print("saved coverage+minilm thresholds to cache/gate.json")
