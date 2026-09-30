"""SQuAD v2 dev set -> corpus/squad/<article>.txt (one paragraph per line) + questions/squad_questions.json.

Each question keeps its gold paragraph (file + line), so retrieval can be checked exactly,
and SQuAD v2's unanswerable questions test the "not in the documents" case.
Usage: uv run tools/fetch_squad.py [n_articles]
"""

import json
import pathlib
import re
import sys
import urllib.request

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
ROOT = pathlib.Path(__file__).resolve().parents[1]
n_articles = int(sys.argv[1]) if len(sys.argv) > 1 else 15

cache = ROOT / "cache" / "squad-dev-v2.0.json"
cache.parent.mkdir(exist_ok=True)
if not cache.exists():
    cache.write_bytes(urllib.request.urlopen(URL).read())
data = json.loads(cache.read_text())["data"][:n_articles]

out = ROOT / "corpus" / "squad"
out.mkdir(parents=True, exist_ok=True)
questions = []
for art in data:
    name = re.sub(r"[^\w-]", "_", art["title"]) + ".txt"
    paras = [" ".join(p["context"].split()) for p in art["paragraphs"]]
    (out / name).write_text("\n".join(paras) + "\n")
    for line, p in enumerate(art["paragraphs"], 1):
        for qa in p["qas"]:
            questions.append({
                "q": qa["question"].strip(),
                "file": name,
                "line": line,
                "answers": sorted({a["text"] for a in qa["answers"]}),
                "impossible": qa["is_impossible"],
            })

(ROOT / "questions" / "squad_questions.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False))
ans = sum(not q["impossible"] for q in questions)
print(f"{len(data)} articles, {sum(1 for _ in out.glob('*.txt'))} files, "
      f"{len(questions)} questions ({ans} answerable, {len(questions) - ans} unanswerable)")
