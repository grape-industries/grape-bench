"""Benchmark: grape (lexical search + brief) vs vector RAG, same LLM, same questions."""

import argparse
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import httpx
import numpy as np

# Reference prices (USD per 1M tokens) for the free nvidia/cloudflare providers: Cloudflare's
# published llama-3.3-70b-fp8-fast rate. The claude provider uses the cost the CLI reports.
PRICE_IN, PRICE_OUT = 0.293, 2.253

PROVIDERS = {
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "meta/llama-3.3-70b-instruct"),
    "cloudflare": (
        "https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1",
        "CF_API_TOKEN",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    ),
}
QUESTIONS = {
    "space": "questions.json", "mixed": "questions_mixed.json", "fresh": "questions_fresh.json",
    "gujarati": "questions_gu.json", "large": "questions_large.json",
}
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# RAG gets a multilingual embedder on non-English sets, so the comparison stays fair.
# e5 wants "query: " / "passage: " prefixes; its 512-token window fits a 1000-char chunk.
EMBEDS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "", ""),
    "e5-large": ("intfloat/multilingual-e5-large", "query: ", "passage: "),
}
CORPUS_EMBED = {"gujarati": "e5-large"}
# Gujarati questions on the English corpus: RAG also gets the multilingual embedder (--embed bge-small shows
# what a default English embedder does).
QUESTION_EMBED = {"questions_fresh_gu.json": "e5-large"}
Q_PREFIX, P_PREFIX = "", ""
# grape's 8 x 600 characters of excerpts is about RAG's 5 x 1000: the same context budget.
LIMIT, SNIPPET, TOP_K, CHUNK_CHARS = 8, 600, 5, 1000
# With --rerank: take CANDIDATES from the retriever, a cross-encoder keeps the best TOP_K.
CANDIDATES = 20
# --server-rerank: grape's own reranker and off-topic gate (GRAPE_RERANK_MODEL on the server) pick the passages.
server_rerank = False
server_limit = 6
RERANKERS = {"minilm": "Xenova/ms-marco-MiniLM-L-6-v2", "jina": "jinaai/jina-reranker-v2-base-multilingual"}
reranker = None
# --gate: reject clearly off-topic questions with no LLM call. Thresholds come from tools/gate.py
# (most out-of-scope rejected with zero in-scope rejected on our question sets).
gate, gate_model = None, None
# English-only function words (none shared with French/Spanish/Portuguese, e.g. no "a", "on", "do").
EN_FUNCTION = set("""the is are was were been does did what which who whom whose when where why how of to
for with by from and not this that these those its there can could should would will many much about into
me my you your""".split())


def english_like(q):
    """The gate's reranker only judges English; other languages go to the LLM route instead."""
    return any(w in EN_FUNCTION for w in re.findall(r"[a-z']+", q.lower()))
COVERAGE_OK = 0.5  # below this grape's excerpts are likely off-target: route instead of answering
NOT_FOUND = "Not in the documents."
NONE_PHRASES = ["not in the documents", "don't know", "do not know", "no information", "not contain",
                "doesn't contain", "not mention", "not available", "not covered", "cannot answer",
                # Gujarati: "not in the documents", "no information", "not available", "not mentioned"
                "દસ્તાવેજોમાં નથી", "માહિતી નથી", "ઉપલબ્ધ નથી", "ઉલ્લેખ નથી"]

ROOT = pathlib.Path(__file__).parent
QUESTION_DIR = ROOT / "questions"
RESULTS = ROOT / "results"
GRAPE = os.environ.get("GRAPE_URL", "http://localhost:8080/grape").rstrip("/")
http = httpx.Client(timeout=180)


def grape_session():
    """grape needs a signed-in user: log in (or sign up once) and send the token on every call."""
    base = GRAPE.rsplit("/grape", 1)[0]
    local = re.match(r"https?://(localhost|127\.)", base)
    if not local and not os.environ.get("GRAPE_PASSWORD"):
        sys.exit("Set GRAPE_EMAIL and GRAPE_PASSWORD for a grape server that is not on this machine.")
    email = os.environ.get("GRAPE_EMAIL", "bench@grape.local")
    password = os.environ.get("GRAPE_PASSWORD", "grape-bench-local")
    r = http.post(f"{base}/auth/login", json={"email": email, "password": password})
    if r.status_code == 401:
        r = http.post(f"{base}/auth/signup", json={"email": email, "password": password})
    r.raise_for_status()
    http.headers["Authorization"] = f"Bearer {r.json()['token']}"


class ClaudeCLI:
    """Uses the local Claude Code login via `claude -p`; cost is what the CLI reports (API list price)."""

    def __init__(self, model, effort=None, thinking=True):
        if not shutil.which("claude"):
            sys.exit("claude CLI not found; install Claude Code and log in.")
        self.model, self.api_s = model or "haiku", 0.0
        self.extra = (["--effort", effort] if effort else []) + (
            [] if thinking else ["--settings", json.dumps({"alwaysThinkingEnabled": False})]
        )

    def chat(self, messages):
        system = messages[0]["content"]
        # -p takes one prompt, so earlier turns are replayed as a transcript (the API resends them too).
        turns = [f"{m['role'].upper()}:\n{m['content']}" for m in messages[1:]]
        prompt = messages[1]["content"] if len(turns) == 1 else "\n\n".join(turns) + "\n\nASSISTANT:"
        # Big prompts go through stdin and a file: one argv string is capped at 128 KB on Linux.
        if len(system) > 100_000:
            f = ROOT / "cache" / f"system-{hashlib.sha1(system.encode()).hexdigest()[:12]}.txt"
            if not f.exists():
                f.write_text(system)
            sys_args = ["--system-prompt-file", str(f)]
        else:
            sys_args = ["--system-prompt", system]
        cmd = [
            "claude", "-p", *sys_args, "--model", self.model, "--tools", "",
            "--output-format", "json", "--no-session-persistence", "--setting-sources", "", "--strict-mcp-config",
            *self.extra,
        ]
        for wait in (5, 15, 45, 0):
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=ROOT)
            try:
                d = json.loads(p.stdout)
            except json.JSONDecodeError:
                d = {"is_error": True, "result": p.stderr or p.stdout}
            if not d.get("is_error"):
                break
            if not wait:
                raise RuntimeError(f"claude failed: {d.get('result')}")
            time.sleep(wait)
        self.api_s += d.get("duration_api_ms", 0) / 1000
        u = d["usage"]
        tin = u["input_tokens"] + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        return d["result"], tin, u["output_tokens"], d["total_cost_usd"]


class LLM:
    def __init__(self, provider, model):
        base, key_var, default_model = PROVIDERS[provider]
        key = os.environ.get(key_var)
        if not key:
            sys.exit(f"Set {key_var} to use provider '{provider}'.")
        if "{CF_ACCOUNT_ID}" in base:
            if not os.environ.get("CF_ACCOUNT_ID"):
                sys.exit("Set CF_ACCOUNT_ID to use provider 'cloudflare'.")
            base = base.format(CF_ACCOUNT_ID=os.environ["CF_ACCOUNT_ID"])
        self.url, self.model = base + "/chat/completions", model or default_model
        self.headers = {"Authorization": f"Bearer {key}"}
        self.api_s = 0.0

    def chat(self, messages):
        """Returns (text, input_tokens, output_tokens, cost); retries on rate limits and 5xx."""
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 512}
        for wait in (3, 6, 12, 24, 48, 0):
            t = time.perf_counter()
            r = http.post(self.url, json=body, headers=self.headers)
            self.api_s += time.perf_counter() - t
            if r.status_code in (429, 500, 502, 503, 504) and wait:
                time.sleep(wait)
                continue
            r.raise_for_status()
            d = r.json()
            u = d.get("usage") or {}
            tin, tout = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
            return d["choices"][0]["message"]["content"] or "", tin, tout, cost(tin, tout)


# ---------- setup ----------

def setup_grape(corpus, cache):
    id_file = cache / "grape_id"
    if id_file.exists():
        gid = id_file.read_text().strip()
        if http.get(f"{GRAPE}/{gid}/metadata").status_code == 200:
            return gid, json.loads((cache / "grape_setup.json").read_text())["seconds"]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(corpus.glob("*.txt")):
            tar.add(f, arcname=f.name)
    t = time.perf_counter()
    r = http.post(f"{GRAPE}/projects", content=buf.getvalue(), headers={"Content-Type": "application/gzip"})
    r.raise_for_status()
    secs = time.perf_counter() - t
    gid = r.json()["id"]
    id_file.write_text(gid)
    (cache / "grape_setup.json").write_text(json.dumps({"seconds": secs}))
    return gid, secs


def chunk_corpus(corpus):
    chunks = []
    for f in sorted(corpus.glob("*.txt")):
        cur = ""
        for para in (p.strip() for p in f.read_text().splitlines()):
            if not para:
                continue
            if cur and len(cur) + len(para) > CHUNK_CHARS:
                chunks.append({"file": f.name, "text": cur})
                cur = ""
            cur = f"{cur}\n{para}" if cur else para
        if cur:
            chunks.append({"file": f.name, "text": cur})
    return chunks


def embedder(name):
    from fastembed import TextEmbedding

    local = ROOT / "cache" / "models" / name.split("/")[-1]
    if local.exists():
        return TextEmbedding(name, specific_model_path=str(local))
    try:
        return TextEmbedding(name)
    except Exception as e:
        # Newer onnxruntime refuses weights stored as a symlink (the HF cache layout); load a real copy.
        if "escapes model directory" not in str(e):
            raise
        snap = next(pathlib.Path("/tmp/fastembed_cache").glob(f"models--*{local.name}*/snapshots/*"))
        shutil.copytree(snap, local)
        return TextEmbedding(name, specific_model_path=str(local))


def setup_vectors(corpus, cache):
    model = embedder(EMBED_MODEL)
    vec_file, chunk_file, meta_file = cache / "vectors.npy", cache / "chunks.json", cache / "vector_setup.json"
    if vec_file.exists():
        return model, np.load(vec_file), json.loads(chunk_file.read_text()), json.loads(meta_file.read_text())["seconds"]
    chunks = chunk_corpus(corpus)
    t = time.perf_counter()
    # Small batches for big models: attention memory grows with batch x sequence^2.
    batch = 8 if "large" in EMBED_MODEL else 32
    vecs = np.array(list(model.passage_embed([P_PREFIX + c["text"] for c in chunks], batch_size=batch)))
    secs = time.perf_counter() - t
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(vec_file, vecs)
    chunk_file.write_text(json.dumps(chunks))
    meta_file.write_text(json.dumps({"seconds": secs}))
    return model, vecs, chunks, secs


# ---------- modes ----------

# Same answer instruction for both modes, so the comparison stays fair.
ANSWER = f"""Answer in 1-2 sentences using only the excerpts, in the language of the question.
If the excerpts don't contain the answer, reply exactly: {NOT_FOUND}"""
GRAPE_ANSWER = """Answer in 1-2 sentences using only the excerpts, in the language of the question.
The excerpts come from a collection with these files:
{titles}
If the excerpts don't contain the answer:
- if these files could still cover it, reply only: SEARCH: <the question as a short plain question,
  in the language of the file names above, with the words the documents would use next to the answer>
  (for a question with parts found in different places, one SEARCH line per part)
- otherwise reply exactly: """ + NOT_FOUND
ROUTE = """A document collection contains these files:
{titles}
Reply with one line only:
SEARCH: <question>   if these documents could answer the question: the question rewritten as a short
                     plain question in the language of the file titles, with the words the documents
                     would use next to the answer (for a question with several parts, one SEARCH
                     line per part)
NONE                       if the question cannot be answered from these documents"""


class Grape:
    def __init__(self, gid):
        self.gid = gid
        files = http.get(f"{GRAPE}/{gid}/metadata").json().get("files", {})
        self.titles = "\n".join(f"- {f['title']}" for f in list(files.values())[:60])

    def search(self, queries):
        """grape picks terms from free text, ranks paragraphs (BM25 + brief), returns snippets + coverage.
        Several queries (a multi-part question) are searched separately and merged by grape."""
        if server_rerank:
            body = {"limit": server_limit, "snippet": SNIPPET}
        elif reranker:
            body = {"limit": CANDIDATES, "cutoff": 0, "snippet": SNIPPET, "rerank": False}
        else:
            body = {"limit": LIMIT, "snippet": SNIPPET, "rerank": False}
        body |= {"query": queries[0]} if len(queries) == 1 else {"queries": queries}
        r = http.post(f"{GRAPE}/{self.gid}/search", json=body)
        r.raise_for_status()
        d = r.json()
        self.refused = bool(d.get("refused"))
        hits = d["hits"]
        if reranker:
            # Rerank each part against its own candidates, so every part keeps its best excerpts.
            per_part = [[h for h in hits if h.get("query", queries[0]) == sub] for sub in queries]
            hits = [h for sub, hs in zip(queries, per_part) for h in rerank(sub, hs, lambda h: h.get("text", ""))[: max(2, TOP_K // len(queries) + 1)]]
        text = "\n".join(f"[{h['path']}:{h.get('line', '?')}] {h.get('text', '')}" for h in hits)
        return text or "(no excerpts found)", d["coverage"]

    def gated(self, q):
        if not gate or not english_like(q):
            return False
        body = {"query": q, "limit": CANDIDATES, "cutoff": 0, "snippet": SNIPPET}
        d = http.post(f"{GRAPE}/{self.gid}/search", json=body).json()
        if d["coverage"] >= gate["cov"]:
            return False
        texts = [h["text"] for h in d["hits"]]
        return not texts or max(gate_model.rerank(q, texts, batch_size=len(texts))) < gate["minilm"]

    def run(self, llm, q):
        st = {"in": 0, "out": 0, "cost": 0, "calls": 0, "route": "direct"}
        if self.gated(q):
            st["route"] = "gated"
            return {"answer": NOT_FOUND, **st}

        def ask(system, user):
            text, i, o, c = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
            st["in"], st["out"], st["cost"], st["calls"] = st["in"] + i, st["out"] + o, st["cost"] + c, st["calls"] + 1
            return text.strip()

        def answer_from(queries, again):
            """Answer from a search; with `again`, the answer may ask for one more search in other words."""
            st.setdefault("searched", []).append(queries)
            excerpts, _ = self.search(queries)
            answer = ask(GRAPE_ANSWER.format(titles=self.titles) if again else ANSWER, f"Question: {q}\n\nExcerpts:\n{excerpts}")
            parts = re.findall(r"\bSEARCH:\s*(.+)", answer) if again else []
            if parts:
                # Keep what was found: the new search adds to the old one (a two-part question needs both).
                st["route"] += "+again"
                return answer_from(queries + [p for p in parts if p not in queries], False)
            return answer

        excerpts, cov = self.search([q])
        st["coverage"] = cov
        if self.refused:
            # The server's off-topic gate: no passage covers it, so no LLM call.
            st["route"] = "refused"
            return {"answer": NOT_FOUND, **st}
        if cov >= COVERAGE_OK:
            # The answer call itself decides: answer, other words (vocabulary gap), or not covered.
            answer = ask(GRAPE_ANSWER.format(titles=self.titles), f"Question: {q}\n\nExcerpts:\n{excerpts}")
            parts = re.findall(r"\bSEARCH:\s*(.+)", answer)
            if parts:
                st["route"] = "answer-search"
                return {"answer": answer_from([q, *parts], True), **st}
            return {"answer": answer, **st}

        # Weak match (other language, off-topic, vocabulary gap): a tiny call with file titles only.
        reply = ask(ROUTE.format(titles=self.titles), f"Question: {q}")
        parts = re.findall(r"\bSEARCH:\s*(.+)", reply)
        if parts:
            st["route"] = "route-search"
            return {"answer": answer_from(parts, True), **st}
        st["route"] = "route-none"
        return {"answer": NOT_FOUND, **st}


def rerank(query, items, text_of):
    """Cross-encoder reads query and each candidate together; keeps the TOP_K best."""
    if len(items) <= 1:
        return items
    scores = list(reranker.rerank(query, [text_of(x) for x in items], batch_size=len(items)))
    return [items[i] for i in sorted(range(len(items)), key=lambda i: -scores[i])][:TOP_K]


def run_rag(llm, model, vecs, chunks, q):
    qv = np.array(list(model.query_embed(Q_PREFIX + q))[0])
    top = list(np.argsort(vecs @ (qv / np.linalg.norm(qv)))[::-1][: CANDIDATES if reranker else TOP_K])
    if reranker:
        top = rerank(q, top, lambda i: chunks[i]["text"])
    ctx = "\n\n".join(f"[{chunks[i]['file']}] {chunks[i]['text']}" for i in top)
    text, i, o, c = llm.chat([
        {"role": "system", "content": ANSWER},
        {"role": "user", "content": f"Question: {q}\n\nExcerpts:\n{ctx}"},
    ])
    return {"answer": text.strip(), "in": i, "out": o, "cost": c, "calls": 1}


# Fallback grader when no keyword matches: names written in another script have endless spellings
# ("Coubertin" is કુબર્ટિન, કુબર્તે, ...). Used the same way for every mode; judged rows are marked.
JUDGE = """You grade answers against expected facts. Reply with one word: YES if the answer states every expected
fact (in any language, spelling or transliteration; extra detail is fine), otherwise NO."""


def judge(llm, q, item, answer):
    """(verdict, cost): keyword check failed, so ask the LLM whether the answer still states the facts."""
    facts = "; ".join(" or ".join(e) if isinstance(e, list) else e for e in item["expect"])
    text, _, _, c = llm.chat([
        {"role": "system", "content": JUDGE},
        {"role": "user", "content": f"Question: {q}\nExpected facts: {facts}\nAnswer: {answer}"},
    ])
    return text.strip().upper().startswith("YES"), c


def correct(answer, item):
    a = answer.lower()
    if item.get("none"):
        return any(p in a for p in NONE_PHRASES)
    return all(any(alt.lower() in a for alt in (e if isinstance(e, list) else [e])) for e in item["expect"])


# ---------- report ----------

def cost(tin, tout):
    return tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT


def summary(rows, setup, price_note):
    print(f"\nCost: {price_note}\n")
    head = f"{'mode':<6}{'acc':>8}{'avg in':>9}{'avg out':>9}{'tot in':>9}{'tot out':>9}{'calls':>7}{'wall':>7}{'api':>7}{'$ total':>10}{'$ / q':>11}{'setup s':>9}"
    print(head + "\n" + "-" * len(head))
    for mode in ("grape", "rag"):
        rs = [r[mode] for r in rows if mode in r]
        if not rs:
            continue
        tin, tout, usd = sum(r["in"] for r in rs), sum(r["out"] for r in rs), sum(r["cost"] for r in rs)
        acc = sum(r["correct"] for r in rs)
        print(
            f"{mode:<6}{f'{acc}/{len(rs)}':>8}{tin / len(rs):>9.0f}{tout / len(rs):>9.0f}{tin:>9}{tout:>9}"
            f"{statistics.mean(r['calls'] for r in rs):>7.1f}{statistics.mean(r['seconds'] for r in rs):>6.1f}s"
            f"{statistics.mean(r['api_seconds'] for r in rs):>6.1f}s"
            f"{usd:>10.5f}{usd / len(rs):>11.6f}{setup.get(mode, 0):>9.2f}"
        )
    if "grape_update" in setup and "rag" in setup:
        print(f"\nUpdate after a doc change: grape reindex {setup['grape_update']:.2f}s, RAG full re-embed {setup['rag']:.2f}s")
    routes = [r["grape"]["route"] for r in rows if "grape" in r]
    if routes:
        print("grape routes: " + ", ".join(f"{k} {routes.count(k)}" for k in sorted(set(routes))))
    print("\nBy question type (correct, avg $ / q):")
    for t in sorted({r["type"] for r in rows}):
        parts = []
        for mode in ("grape", "rag"):
            rs = [r[mode] for r in rows if r["type"] == t and mode in r]
            if rs:
                parts.append(f"{mode} {sum(x['correct'] for x in rs)}/{len(rs)} ${statistics.mean(x['cost'] for x in rs):.5f}")
        print(f"  {t:<13}" + "   ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=QUESTIONS, default="space")
    ap.add_argument("--provider", choices=["claude", *PROVIDERS], default="claude")
    ap.add_argument("--model")
    ap.add_argument("--only", choices=["grape", "rag"])
    ap.add_argument("--embed", choices=EMBEDS, help="RAG embedding model (default: bge-small; e5-large is multilingual)")
    ap.add_argument("--effort", choices=["low", "medium", "high"], help="claude provider: thinking effort")
    ap.add_argument("--no-thinking", action="store_true", help="claude provider: disable extended thinking")
    ap.add_argument("--rerank", choices=RERANKERS, help="cross-encoder over top-20 candidates (both modes)")
    ap.add_argument("--gate", action="store_true", help="grape: reject off-topic questions without an LLM call")
    ap.add_argument("--server-rerank", action="store_true", help="grape: the server's reranker and off-topic gate pick passages")
    ap.add_argument("--server-limit", type=int, default=6, help="with --server-rerank: passages per search (default 6)")
    ap.add_argument("--questions", help="question file (default: the corpus's own set)")
    ap.add_argument("-n", type=int, help="run only the first N questions")
    ap.add_argument("--no-judge", action="store_true", help="keyword scoring only (no LLM judge fallback)")
    args = ap.parse_args()

    grape_session()
    global EMBED_MODEL, Q_PREFIX, P_PREFIX
    if args.questions:
        args.questions = pathlib.Path(args.questions).name  # "questions/x.json" and "x.json" both work
    default_embed = QUESTION_EMBED.get(args.questions or "", CORPUS_EMBED.get(args.corpus, "bge-small"))
    embed = args.embed or default_embed
    EMBED_MODEL, Q_PREFIX, P_PREFIX = EMBEDS[embed]
    corpus, cache = ROOT / "corpus" / args.corpus, ROOT / "cache" / args.corpus
    if not any(corpus.glob("*.txt")):
        sys.exit(f"No corpus in {corpus}; run: uv run tools/fetch_corpus.py {args.corpus}")
    cache.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    if args.provider == "claude":
        llm = ClaudeCLI(args.model, args.effort, thinking=not args.no_thinking)
        price_note = f"Claude API list price reported by the claude CLI ({llm.model}, thinking {'off' if args.no_thinking else 'on'})"
    else:
        llm = LLM(args.provider, args.model)
        price_note = f"reference ${PRICE_IN}/M in, ${PRICE_OUT}/M out"
    questions = json.loads((QUESTION_DIR / (args.questions or QUESTIONS[args.corpus])).read_text())[: args.n]
    if args.rerank:
        global reranker
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        reranker = TextCrossEncoder(RERANKERS[args.rerank], threads=os.cpu_count())
        price_note += f", rerank {args.rerank}"
    if args.gate:
        global gate, gate_model
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        gate = json.loads((ROOT / "cache" / "gate.json").read_text())
        gate_model = reranker if args.rerank == "minilm" else TextCrossEncoder(RERANKERS["minilm"], threads=os.cpu_count())
        price_note += f", gate {gate}"
    if args.server_rerank:
        global server_rerank, server_limit
        server_rerank, server_limit = True, args.server_limit
        price_note += f", grape server rerank + gate, {server_limit} passages"
    modes = [args.only] if args.only else ["grape", "rag"]

    setup = {}
    if "grape" in modes:
        gid, setup["grape"] = setup_grape(corpus, cache)
        t = time.perf_counter()
        http.post(f"{GRAPE}/{gid}/index").raise_for_status()
        setup["grape_update"] = time.perf_counter() - t
        grape = Grape(gid)
        print(f"grape project {gid}, indexed in {setup['grape']:.2f}s")
    if "rag" in modes:
        # Each non-default model keeps its own vectors next to the corpus's default ones.
        vec_cache = cache if embed == CORPUS_EMBED.get(args.corpus, "bge-small") else cache / embed
        vec_cache.mkdir(exist_ok=True)
        emb, vecs, chunks, setup["rag"] = setup_vectors(corpus, vec_cache)
        print(f"vector index: {len(chunks)} chunks, embedded in {setup['rag']:.2f}s ({EMBED_MODEL}, local)")

    rows = []
    for n, item in enumerate(questions, 1):
        row = {"q": item["q"], "type": item["type"], "expect": item.get("expect", "NOT FOUND")}
        for mode in modes:
            t, api0 = time.perf_counter(), llm.api_s
            if mode == "grape":
                res = grape.run(llm, item["q"])
            else:
                res = run_rag(llm, emb, vecs, chunks, item["q"])
            res["seconds"] = round(time.perf_counter() - t, 2)
            res["api_seconds"] = round(llm.api_s - api0, 2)
            res["correct"] = correct(res["answer"], item)
            refused = any(p in res["answer"].lower() for p in NONE_PHRASES)
            if not res["correct"] and not item.get("none") and not refused and not args.no_judge:
                res["correct"], res["judge_cost"] = judge(llm, item["q"], item, res["answer"])
                res["judged"] = True
            row[mode] = res
        rows.append(row)
        status = "  ".join(f"{m} {'ok ' if row[m]['correct'] else 'MISS'} {row[m]['in']:>5}+{row[m]['out']:<4}tok" for m in modes)
        print(f"[{n:>2}/{len(questions)}] {status}  {item['q'][:60]}")

    tag = f"-{pathlib.Path(args.questions).stem}" if args.questions else ""
    out = RESULTS / f"{args.corpus}{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    files = sorted(corpus.glob("*.txt"))
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "questions": args.questions or QUESTIONS[args.corpus],
        "embed_key": embed,
        "modes": modes,
        "corpus_files": len(files),
        "corpus_bytes": sum(f.stat().st_size for f in files),
        "chunks": len(chunks) if "rag" in modes else None,
        "settings": {"limit": LIMIT, "snippet": SNIPPET, "top_k": TOP_K, "chunk_chars": CHUNK_CHARS},
        "thinking": args.provider != "claude" or not args.no_thinking,
        "rerank": args.rerank, "gate": args.gate, "server_rerank": args.server_rerank, "server_limit": args.server_limit, "judge": not args.no_judge,
        "grape_version": grape_version(),
        "bench_commit": bench_commit(),
    }
    out.write_text(json.dumps({
        "corpus": args.corpus, "provider": args.provider, "model": llm.model, "cost_basis": price_note, "embed": EMBED_MODEL,
        "meta": meta, "setup_seconds": setup, "rows": rows,
    }, indent=2, ensure_ascii=False))
    summary(rows, setup, price_note)
    print(f"\nDetails: {out}")
    import report

    print(f"Report: {report.build()}")


def grape_version():
    try:
        return http.get(GRAPE.rsplit("/grape", 1)[0] + "/health").json().get("version")
    except (httpx.HTTPError, ValueError):
        return None


def bench_commit():
    p = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout
    return (p.stdout.strip() + ("-dirty" if dirty.strip() else "")) if p.returncode == 0 else None


if __name__ == "__main__":
    main()
