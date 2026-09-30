"""Build the benchmark report (site/) from results/*.json. Standard library only, so CI can run it.

Usage: python3 report.py            # writes site/index.html, site/suite/<suite>.html, site/data.json
"""

import html
import os
import json
import pathlib
import re
import shutil
import statistics

ROOT = pathlib.Path(__file__).parent
RESULTS, SITE = ROOT / "results", ROOT / "site"
# Top-bar links; CI or a fork can point them elsewhere.
REPO = os.environ.get("BENCH_SOURCE_URL", "https://github.com/grape-industries/grape-bench")
GRAPE_URL = os.environ.get("BENCH_GRAPE_URL", "GRAPE_URL")
NOT_FOUND = "NOT FOUND"

# Suites in page order: (corpus, question file, RAG embedder) -> title, what it tests.
E5, BGE = "intfloat/multilingual-e5-large", "BAAI/bge-small-en-v1.5"
SUITES = {
    ("fresh", "questions_fresh.json", BGE): (
        "English",
        "28 questions on 12 English Wikipedia articles (tea, octopus, Marie Curie, ...), 630 KB. Four are asked in "
        "Hindi, Spanish or French, and five are off-topic.",
    ),
    ("fresh", "questions_fresh_gu.json", E5): (
        "Gujarati questions, English documents",
        "The same 12 English articles, every question asked in Gujarati. Lexical search cannot match across "
        "languages, so Grape's first LLM call rewrites the question in the documents' language. Vector RAG uses a "
        "large multilingual embedding model (multilingual-e5-large), which embeds both languages in one space.",
    ),
    ("fresh", "questions_fresh_gu.json", BGE): (
        "Gujarati questions, English documents, default English embedder",
        "The same Gujarati questions, but Vector RAG uses the English embedding model most tutorials start with "
        "(bge-small-en). Shown to make the point that a vector setup only reaches across languages if its embedder does.",
    ),
    ("gujarati", "questions_gu.json", E5): (
        "Gujarati documents",
        "Gujarati questions on 12 Gujarati Wikipedia articles (Ahmedabad, Gir, Narmada, ...). Vector RAG uses "
        "multilingual-e5-large.",
    ),
    ("large", "questions_large.json", BGE): (
        "Large context: 300 articles, 18 MB",
        "300 long English Wikipedia articles, about 4.6 million tokens: far more than any model's context window, so "
        "the model only sees what retrieval hands it. Shows answer quality and the cost of getting the index ready.",
    ),
}
DEFAULT_QUESTIONS = {"space": "questions.json", "mixed": "questions_mixed.json", "fresh": "questions_fresh.json",
                     "gujarati": "questions_gu.json", "large": "questions_large.json"}
MODES = {
    "grape": ("Grape", "Lexical search (BM25 + brief) picks passages; the LLM may search again in other words.", "#963d97"),
    "rag": ("Vector RAG", "Embed 1000-character chunks; the 5 nearest go to the LLM.", "#009ddc"),
}
esc = html.escape


def load():
    runs = []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        if "corpus" not in d:  # runs older than the corpus field
            continue
        meta = d.get("meta", {})
        m = re.match(r"[a-z]+-(questions[\w]*)-\d{8}-\d{6}$", f.stem)
        questions = meta.get("questions") or (m.group(1) + ".json" if m else DEFAULT_QUESTIONS.get(d["corpus"]))
        d.update(name=f.stem, file=f.name, suite=(d["corpus"], questions, d.get("embed")), meta=meta)
        runs.append(d)
    return runs


def stats(run, mode):
    rows = [r for r in run["rows"] if mode in r]
    if not rows:
        return None
    ans = [r for r in rows if r["expect"] != NOT_FOUND]
    off = [r for r in rows if r["expect"] == NOT_FOUND]
    x = [r[mode] for r in rows]
    return {
        "answered": sum(r[mode]["correct"] for r in ans), "answerable": len(ans),
        "refused": sum(r[mode]["correct"] for r in off), "off_topic": len(off),
        "tokens": statistics.mean(r["in"] + r["out"] for r in x),
        "tokens_in": statistics.mean(r["in"] for r in x), "tokens_out": statistics.mean(r["out"] for r in x),
        "cost": statistics.mean(r["cost"] for r in x), "seconds": statistics.mean(r["seconds"] for r in x),
        "calls": statistics.mean(r["calls"] for r in x),
        "setup": run.get("setup_seconds", {}).get(mode),
    }


def spread(values, fmt):
    """Mean, and the range when runs differ."""
    lo, hi, avg = min(values), max(values), statistics.mean(values)
    return fmt(avg) if lo == hi else f"{fmt(avg)} <small>({fmt(lo)} to {fmt(hi)})</small>"


def seconds(s):
    if s is None:
        return "-"
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s" if s < 90 else f"{s / 60:.1f} min"


def bar(value, top, kind):
    width = 0 if not top else max(1.5, 100 * value / top)
    return f'<span class="bar {kind}"><i style="width:{width:.1f}%"></i></span>'


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def fmt_n(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def page(title, sidebar, body, depth=0):
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><meta name="description" content="Grape vs Vector RAG: same documents, same questions, same LLM.">
<link rel="icon" type="image/svg+xml" href="{up}favicon.svg">
<link rel="icon" type="image/png" sizes="192x192" href="{up}icon-light-192.png" media="(prefers-color-scheme: light)">
<link rel="icon" type="image/png" sizes="192x192" href="{up}icon-dark-192.png" media="(prefers-color-scheme: dark)">
<link rel="stylesheet" href="{up}style.css">
<script>try{{document.documentElement.dataset.theme=localStorage.getItem("grape-bench-theme")||"light"}}catch(e){{document.documentElement.dataset.theme="light"}}</script></head>
<body>
<header class="top"><a class="brand" href="{up}index.html"><span class="mark"></span>Grape benchmark</a>
<nav><a href="{GRAPE_URL}">Grape</a><a href="{REPO}">Source</a><button class="theme" type="button">Dark</button></nav></header>
<div class="horizon"></div>
<div class="layout">
<aside class="side"><button class="menu" aria-expanded="false">Menu</button><div class="side-in">{sidebar}</div></aside>
<main>{body}</main>
</div>
<script>
const theme = document.querySelector(".theme");
const paint = () => theme.textContent = document.documentElement.dataset.theme === "dark" ? "Light" : "Dark";
theme.onclick = () => {{
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {{ localStorage.setItem("grape-bench-theme", next); }} catch (e) {{}}
  paint();
}};
paint();
document.querySelector(".menu").onclick = (e) => {{
  const open = e.target.getAttribute("aria-expanded") !== "true";
  e.target.setAttribute("aria-expanded", open); e.target.textContent = open ? "Close" : "Menu";
  document.querySelector(".side-in").classList.toggle("open", open);
}};
</script>
</body></html>
"""


def suites_nav(order, suites, current, depth):
    up = "../" * depth
    items = "".join(
        f'<li><a href="{up}suite/{slug(SUITES.get(k, (k[0],))[0])}.html"{" class=active" if k == current else ""}>'
        f'{esc(SUITES.get(k, (k[0],))[0])}</a></li>' for k in order)
    return (f'<span class="label">Report</span><ul><li><a href="{up}index.html"{" class=active" if current is None else ""}>Overview</a></li>'
            f'<li><a href="{up}index.html#method">How it works</a></li></ul>'
            f'<span class="label">Suites</span><ul>{items}</ul>')


def cards(key, runs):
    """Grape and Vector RAG side by side: mean and range over the runs."""
    modes = [m for m in MODES if any(stats(r, m) for r in runs)]
    per = {m: [s for s in (stats(r, m) for r in runs) if s] for m in modes}
    n_ans = per[modes[0]][0]["answerable"]
    top = lambda f: max(statistics.mean(f(s) for s in per[m]) for m in modes)
    cost_top, tok_top, call_top, sec_top = top(lambda s: s["cost"]), top(lambda s: s["tokens_in"]), top(lambda s: s["calls"]), top(lambda s: s["seconds"])
    setup = {m: next((s["setup"] for s in per[m] if s["setup"] is not None), None) for m in modes}
    setup_top = max((v for v in setup.values() if v), default=0)
    out = []
    for m in modes:
        s, label = per[m], MODES[m][0]
        acc = [x["answered"] for x in s]
        mean = lambda f: statistics.mean(f(x) for x in s)
        out.append(f"""<div class="mode {m}">
<span class="label">{label}</span>
<p class="big">{spread(acc, fmt_n)}<span> / {n_ans}</span></p>
{bar(statistics.mean(acc), n_ans, m)}
<dl>
<dt>Off-topic refused</dt><dd>{spread([x["refused"] for x in s], fmt_n)} / {s[0]["off_topic"]}</dd>
<dt>Input tokens per question</dt><dd>{spread([x["tokens_in"] for x in s], lambda v: f"{v:,.0f}")}{bar(mean(lambda x: x["tokens_in"]), tok_top, m)}</dd>
<dt>LLM calls per question</dt><dd>{spread([x["calls"] for x in s], lambda v: f"{v:.2f}")}{bar(mean(lambda x: x["calls"]), call_top, m)}</dd>
<dt>Cost per question</dt><dd>{spread([x["cost"] for x in s], lambda v: f"${v:.5f}")}{bar(mean(lambda x: x["cost"]), cost_top, m)}</dd>
<dt>Time per question</dt><dd>{spread([x["seconds"] for x in s], lambda v: f"{v:.1f} s")}{bar(mean(lambda x: x["seconds"]), sec_top, m)}</dd>
<dt>{"Index time" if m == "grape" else "Embedding time"}</dt><dd>{seconds(setup[m])}{bar(setup[m] or 0, setup_top, m)}</dd>
</dl></div>""")
    return f'<div class="modes">{"".join(out)}</div>'


def expect_text(e):
    if e == NOT_FOUND:
        return "Off-topic: must say it is not in the documents."
    return " and ".join(" or ".join(x) if isinstance(x, list) else x for x in e)


def suite_data(runs):
    """Every question with each run's answers, for the question viewer."""
    modes = [m for m in MODES if any(m in r for r in runs[0]["rows"])]
    qs = []
    for i, r in enumerate(runs[0]["rows"]):
        entry = {"n": i + 1, "q": r["q"], "type": r["type"], "expect": expect_text(r["expect"]), "runs": []}
        for run in runs:
            row = next((x for x in run["rows"] if x["q"] == r["q"]), None)
            if not row:
                continue
            per = {}
            for m in modes:
                if m not in row:
                    continue
                x = row[m]
                searched = x.get("searched") or []
                rounds = searched if searched and isinstance(searched[0], list) else [searched] if searched else []
                per[m] = {"ok": bool(x["correct"]), "judged": bool(x.get("judged")), "answer": x["answer"],
                          "in": x["in"], "out": x["out"], "cost": x["cost"], "s": x["seconds"], "calls": x["calls"],
                          "route": x.get("route"), "coverage": x.get("coverage"), "searched": [" | ".join(q) for q in rounds]}
            entry["runs"].append({"name": run["meta"].get("date") or run["name"], "file": run["file"], **per})
        qs.append(entry)
    return qs


def mark(entry, m):
    oks = [r[m]["ok"] for r in entry["runs"] if m in r]
    k = sum(oks)
    return "ok" if k == len(oks) else "miss" if k == 0 else "part"


def suite_page(key, runs, order, suites):
    title, what = SUITES.get(key, (f"{key[0]} / {key[1]}", ""))
    data = suite_data(runs)
    last = runs[-1]
    meta = last["meta"]
    facts = [f"{len(runs)} run{'s' if len(runs) > 1 else ''}", f"model {last['model']}, thinking {'on' if meta.get('thinking') else 'off'}",
             f"Vector RAG embedder {last['embed']}"]
    if meta.get("corpus_bytes"):
        facts.insert(1, f"{meta['corpus_files']} files, {meta['corpus_bytes'] / 1e6:.1f} MB")
    strip = "".join(
        f'<li><a href="#q-{e["n"]}" data-n="{e["n"]}" data-diff="{int(mark(e, "grape") != mark(e, "rag"))}" '
        f'data-miss="{int(mark(e, "grape") != "ok" or mark(e, "rag") != "ok")}" title="{esc(e["q"])}">'
        f'<span>{e["n"]:02d}</span><i class="{mark(e, "grape")}"></i><i class="{mark(e, "rag")}"></i></a></li>'
        for e in data)
    sidebar = suites_nav(order, suites, key, 1)
    raw = " ".join(f'<a href="../results/{r["file"]}">{esc(r["meta"].get("date") or r["name"])}</a>' for r in runs)
    body = f"""<p class="crumbs">Suites / {esc(title)}</p>
<h1>{esc(title)}</h1><p class="lede">{esc(what)}</p>
<p class="facts">{esc(" / ".join(facts))}. Raw results: {raw}</p>
{cards(key, runs)}
<section class="viewer">
<div class="strip-head"><span class="label">Questions</span>
<div class="filters"><button data-f="all" aria-pressed="true">All</button><button data-f="diff" aria-pressed="false">Differences</button><button data-f="miss" aria-pressed="false">Misses</button></div>
<p class="legend"><i class="ok"></i>right every run <i class="part"></i>some runs <i class="miss"></i>never. Left half Grape, right half Vector RAG.</p></div>
<ol class="strip">{strip}</ol>
<div class="vbody" aria-live="polite">
<div class="vhead"><span class="label" id="vpos"></span>
<div class="runs" id="vruns"></div></div>
<h2 id="vq"></h2><p class="expect" id="vexp"></p>
<div class="answers" id="vans"></div>
<div class="pager"><button id="prev">Previous</button><span class="keys">j / k or arrow keys</span><button id="next">Next</button></div>
</div>
</section>
<script type="application/json" id="data">{json.dumps(data).replace("</", "<\\/")}</script>
<script>
const Q = JSON.parse(document.getElementById("data").textContent);
const NAMES = {{grape: "Grape", rag: "Vector RAG"}};
let cur = 0, run = 0, filter = "all";
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {{ const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }};
const keep = (a) => filter === "all" || a.dataset[filter] === "1";
const shown = () => [...document.querySelectorAll(".strip a")].filter(keep).map((a) => +a.dataset.n - 1);
function show(i) {{
  cur = Math.max(0, Math.min(Q.length - 1, i));
  const q = Q[cur]; run = Math.min(run, q.runs.length - 1);
  $("vpos").textContent = "Question " + q.n + " of " + Q.length + ", " + q.type;
  $("vq").textContent = q.q; $("vexp").textContent = "Expected: " + q.expect;
  $("vruns").replaceChildren(...q.runs.map((r, k) => {{ const b = el("button", "", "Run " + (k + 1)); b.setAttribute("aria-pressed", k === run); b.title = r.name; b.onclick = () => {{ run = k; show(cur); }}; return b; }}));
  const r = q.runs[run];
  $("vans").replaceChildren(...Object.keys(NAMES).filter((m) => r[m]).map((m) => {{
    const x = r[m], card = el("div", "ans " + m + (x.ok ? " ok" : " miss"));
    const h = el("div", "ans-head"); h.append(el("span", "label", NAMES[m]), el("b", "verdict", (x.ok ? "Right" : "Wrong") + (x.judged ? " (judge)" : "")));
    const stats = el("dl", "ans-stats");
    [["Input tokens", x.in.toLocaleString("en")], ["LLM calls", x.calls], ["Cost", "$" + x.cost.toFixed(5)], ["Time", x.s.toFixed(1) + " s"]]
      .concat(x.route ? [["Route", x.route]] : []).concat(x.coverage != null ? [["Coverage", x.coverage]] : [])
      .forEach(([k, v]) => {{ const d = el("div"); d.append(el("dt", "", k), el("dd", "", String(v))); stats.append(d); }});
    card.append(h, el("p", "ans-text", x.answer), stats);
    if (x.searched.length) {{ const ul = el("ul", "searched"); x.searched.forEach((s) => ul.append(el("li", "", "Searched: " + s))); card.append(ul); }}
    return card;
  }}));
  document.querySelectorAll(".strip a").forEach((a) => {{ a.classList.toggle("active", +a.dataset.n === q.n); a.toggleAttribute("aria-current", +a.dataset.n === q.n); }});
  history.replaceState(null, "", "#q-" + q.n);
}}
function step(d) {{ const list = shown(); const at = list.indexOf(cur); show(list[Math.max(0, Math.min(list.length - 1, (at < 0 ? 0 : at) + d))] ?? cur); }}
document.querySelectorAll(".strip a").forEach((a) => a.onclick = (e) => {{ e.preventDefault(); show(+a.dataset.n - 1); }});
document.querySelectorAll(".filters button").forEach((b) => b.onclick = () => {{
  filter = b.dataset.f; document.querySelectorAll(".filters button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  document.querySelectorAll(".strip li").forEach((li) => li.hidden = !keep(li.firstChild));
  const list = shown(); if (list.length && !list.includes(cur)) show(list[0]);
}});
$("prev").onclick = () => step(-1); $("next").onclick = () => step(1);
document.addEventListener("keydown", (e) => {{
  if (e.target.closest("input,textarea")) return;
  if (e.key === "j" || e.key === "ArrowDown" || e.key === "ArrowRight") {{ e.preventDefault(); step(1); }}
  if (e.key === "k" || e.key === "ArrowUp" || e.key === "ArrowLeft") {{ e.preventDefault(); step(-1); }}
}});
const m = location.hash.match(/^#q-(\\d+)$/); show(m ? +m[1] - 1 : 0);
</script>"""
    return page(f"{title}: Grape benchmark", sidebar, body, depth=1)


METHOD = """
<section class="method"><h2>How it works</h2>
<p>Both answer the same questions with the same LLM and the same instruction: answer in 1-2 sentences from the
given text, in the language of the question, or reply "Not in the documents." Answers are checked by keyword first:
each expected fact must appear (common Gujarati spellings are listed as alternatives). When no keyword matches, an LLM
judge (the same model) is asked whether the answer still states every expected fact, because a foreign name written in
Gujarati has many spellings. The judge is used the same way for Grape and Vector RAG, and judged answers are marked "(judge)" on
the run pages. Off-topic questions must be refused, checked by keyword only.</p>
<ul>
<li><b>Grape</b> searches a trigram index with the question itself (BM25 per line, plus title, lead and proximity signals).
For English questions the server then reranks the 20 best candidates with a small cross-encoder (ms-marco-MiniLM-L-6-v2, CPU)
and sends the best 5 passages of up to 600 characters; a clearly off-topic English question is refused with no LLM call. When the match is weak (another language, a vocabulary gap), one small call sees
only the file titles and rewrites the question in the documents' language, or says it is off-topic. The answer call may ask
for one more search in other words.</li>
<li><b>Vector RAG</b> splits documents into 1000-character chunks, embeds them locally (fastembed, CPU), and sends the 5
nearest chunks. English sets use bge-small-en-v1.5; Gujarati sets use multilingual-e5-large.</li>
</ul>
<p><b>Cost</b> is the Claude API list price that the claude CLI reports per call. <b>Index / embedding time</b> is measured once
on an 8-core laptop CPU; the vectors are cached for later runs. LLM answers vary between runs, so each suite is run several
times and the page shows the mean with the range. With 23-29 questions a suite, one question is 3-4 points: read small gaps as
ties.</p>
<p><b>Limits.</b> Keyword scoring can miss a correct answer written another way; every answer is on the suite pages so you
can check. The question sets are small and written by us from the articles. Grape's ranking was tuned on the English and
Gujarati-question sets; the Gujarati-documents and large sets were only run after tuning.</p>
<p>Documents: Wikipedia articles, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>
(corpus/ in the repository). Run it yourself: see the <a href="REPO">README</a>.</p>
</section>
""".replace('"REPO"', f'"{REPO}"')

CSS = """
@font-face{font-family:"Inter";src:url("inter-ascii.woff2") format("woff2");font-weight:400 700;font-display:swap;unicode-range:U+0020-007E}
:root{--ink:#1d1d1f;--grey:#6e6e73;--line:#e8e8ed;--line-soft:#f0f0f3;--soft:#f5f5f7;--bg:#fff;--ok:#61bb46;--part:#fdb827;--miss:#e03a3e;--indigo:#5b52b0;
--seven:linear-gradient(90deg,#61bb46 0 14.29%,#fdb827 0 28.57%,#f5821f 0 42.86%,#e03a3e 0 57.14%,#963d97 0 71.43%,#5b52b0 0 85.71%,#009ddc 0);
--seven-down:linear-gradient(#61bb46 0 14.29%,#fdb827 0 28.57%,#f5821f 0 42.86%,#e03a3e 0 57.14%,#963d97 0 71.43%,#5b52b0 0 85.71%,#009ddc 0);
--mono:ui-monospace,"SF Mono",Menlo,"DejaVu Sans Mono",monospace;color-scheme:light}
:root[data-theme=dark]{--ink:#f5f5f7;--grey:#a1a1a6;--line:#2c2c2e;--line-soft:#1f1f21;--soft:#161618;--bg:#000;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Inter","Noto Sans Gujarati",system-ui,sans-serif}
a{color:inherit}
.top{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:24px;height:56px;padding:0 24px;background:var(--bg);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;letter-spacing:-.02em;text-decoration:none}
.mark{width:18px;height:18px;background:var(--seven-down)}
.top nav{margin-left:auto;display:flex;align-items:center;gap:20px;font-size:14px}.top .theme{border:0;padding:0;background:none;color:var(--grey);font-size:14px}.top .theme:hover{color:var(--ink)}.top nav a{color:var(--grey);text-decoration:none}.top nav a:hover{color:var(--ink)}
.horizon{height:3px;background:var(--seven);opacity:.7}
.layout{display:grid;grid-template-columns:240px minmax(0,1fr);gap:48px;padding:0 32px}
.side{position:sticky;top:59px;align-self:start;max-height:calc(100vh - 59px);overflow-y:auto;padding:32px 0}
.side ul,.side ol{list-style:none;margin:8px 0 24px;padding:0}
.side a{display:block;padding:5px 10px;border-left:2px solid transparent;color:var(--grey);font-size:14px;text-decoration:none}
.side a:hover{color:var(--ink)}.side a.active{border-image:var(--seven-down) 1;background:var(--soft);color:var(--ink);font-weight:500}
.label{display:block;font:12px/1.5 var(--mono);letter-spacing:.02em;color:var(--grey)}
.label::before{content:"[ "}.label::after{content:" ]"}
.menu{display:none}
.legend i{display:inline-block;width:8px;height:8px;margin:0 4px 0 10px}
.strip-head{display:flex;flex-wrap:wrap;align-items:center;gap:12px 20px;margin-bottom:12px}.strip-head .legend{margin:0}
.strip{list-style:none;margin:0 0 24px;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(46px,1fr));gap:6px}
.strip a{position:relative;display:grid;place-items:center;height:44px;border:1px solid var(--line);font:12px var(--mono);color:var(--grey);text-decoration:none;overflow:hidden}
.strip a:hover{color:var(--ink);border-color:var(--grey)}.strip a.active{border-color:var(--ink);color:var(--ink);font-weight:600;box-shadow:inset 0 0 0 1px var(--ink)}
.strip i{position:absolute;bottom:0;width:50%;height:5px}.strip i:first-of-type{left:0}.strip i:last-of-type{right:0}
i.ok{background:var(--ok)}i.part{background:var(--part)}i.miss{background:var(--miss)}
.legend{font-size:12px;color:var(--grey)}
.filters{display:flex;margin:8px 0 0}
button{font:inherit;font-size:13px;padding:6px 12px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
button+button{border-left:0}button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
main{min-width:0;padding:32px 0 96px}
.crumbs{font:12px var(--mono);color:var(--grey);margin:0}
h1{font-size:clamp(32px,4.5vw,48px);line-height:1.08;letter-spacing:-.035em;margin:8px 0 12px}
h2{font-size:26px;letter-spacing:-.02em;margin:0 0 8px;line-height:1.25}
.lede{font-size:18px;color:var(--grey);max-width:70ch;margin:0 0 12px}.facts{color:var(--grey);font-size:14px}
.modes{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1px;margin:24px 0 40px;background:var(--line);border:1px solid var(--line)}
.mode{background:var(--bg);padding:24px}
.big{font-size:52px;font-weight:600;letter-spacing:-.045em;margin:8px 0 0;line-height:1.05}.big span{font-size:22px;color:var(--grey);letter-spacing:0}.big small{font-size:13px;font-weight:400;color:var(--grey);letter-spacing:0}
.bar{display:block;height:5px;background:var(--soft);margin:6px 0 2px}.bar i{display:block;height:100%;background:var(--grey)}.bar.grape i{background:var(--seven)}
dl{margin:16px 0 0;font-size:14px}dt{color:var(--grey);margin-top:10px}dd{margin:0;font-weight:600;font-variant-numeric:tabular-nums}dd small{font-weight:400;color:var(--grey)}
.viewer{border:1px solid var(--line);padding:28px}.vbody{padding-top:24px;border-top:1px solid var(--line-soft)}
a:focus-visible,button:focus-visible{outline:2px solid var(--indigo);outline-offset:2px}
.vhead{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:12px;margin-bottom:16px}
.runs{display:flex}
.expect{color:var(--grey);font-size:15px;margin:0 0 20px}
.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.ans{display:flex;flex-direction:column;border:1px solid var(--line)}
.ans-head{display:flex;justify-content:space-between;align-items:baseline;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--soft)}
.ans.grape .ans-head{background:var(--seven) top/100% 2px no-repeat,var(--soft)}
.verdict{font-size:13px}.ans.ok .verdict{color:var(--ok)}.ans.miss .verdict{color:var(--miss)}
.ans-text{margin:0;padding:16px;white-space:pre-wrap;flex:1}
.ans-stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:10px 16px;margin:0;border-top:1px solid var(--line-soft);padding:12px 16px}
.ans-stats dt{margin:0;font-size:12px}.ans-stats dd{font-size:14px}
.searched{margin:0;padding:10px 16px;list-style:none;border-top:1px solid var(--line-soft);font:12px var(--mono);color:var(--grey)}
.pager{display:flex;justify-content:space-between;align-items:center;margin-top:24px}.pager button{border-left:1px solid var(--line)}
.keys{font:12px var(--mono);color:var(--grey)}
.suites{display:grid;gap:16px;margin:24px 0 48px}
.suite-card{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:24px;padding:24px;border:1px solid var(--line);text-decoration:none}
.suite-card:hover{background:var(--soft)}.suite-card h2{font-size:20px}.suite-card p{margin:0;color:var(--grey);font-size:14px}
.duo{display:grid;gap:10px;align-content:center}.duo div{display:grid;grid-template-columns:90px 1fr 70px;gap:10px;align-items:center;font-size:13px}.duo b{text-align:right;font-variant-numeric:tabular-nums}.duo .bar{margin:0}
.method{max-width:72ch;padding-top:24px;border-top:1px solid var(--line-soft)}.method li{margin-bottom:8px}
@media (max-width:900px){.layout{grid-template-columns:1fr;gap:0}.side{position:static;max-height:none;padding:16px 0 0}
.menu{display:block;width:100%;text-align:left}.side-in{display:none;margin-top:12px}.side-in.open{display:block}.suite-card{grid-template-columns:1fr}.viewer{padding:16px}}
"""


def overview(order, suites):
    rows = []
    for k in order:
        runs = suites[k]
        title, what = SUITES.get(k, (k[0], ""))
        per = {m: [s for s in (stats(r, m) for r in runs) if s] for m in MODES}
        n = per["grape"][0]["answerable"] if per.get("grape") else 0
        duo = ""
        for m in MODES:
            if per.get(m):
                acc = statistics.mean(x["answered"] for x in per[m])
                duo += f'<div><span>{MODES[m][0]}</span>{bar(acc, n, m)}<b>{fmt_n(acc)} / {n}</b></div>'
        rows.append(f'<a class="suite-card" href="suite/{slug(title)}.html"><div><h2>{esc(title)}</h2><p>{esc(what)}</p></div>'
                    f'<div class="duo">{duo}</div></a>')
    return f'<div class="suites">{"".join(rows)}</div>'


def build():
    runs = load()
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "suite").mkdir(parents=True)
    (SITE / "results").mkdir()
    (SITE / "style.css").write_text(CSS)
    for asset in ("inter-ascii.woff2", "favicon.svg", "icon-light-192.png", "icon-dark-192.png"):
        shutil.copy(ROOT / "assets" / asset, SITE)
    suites = {}
    for r in runs:
        suites.setdefault(r["suite"], []).append(r)
        shutil.copy(RESULTS / r["file"], SITE / "results")
    order = [k for k in SUITES if k in suites] + [k for k in suites if k not in SUITES]
    for k in order:
        (SITE / "suite" / f"{slug(SUITES.get(k, (k[0],))[0])}.html").write_text(suite_page(k, suites[k], order, suites))
    latest = max((r["meta"].get("date") or "" for r in runs), default="")
    intro = f"""<p class="crumbs">Report / Overview</p><h1>Grape vs Vector RAG</h1>
<p class="lede">Same documents, same questions, same LLM, two ways to find the passages it answers from. Open a suite to
step through every question and read both answers side by side.{f" Last run {esc(latest)}." if latest else ""}</p>"""
    body = intro + overview(order, suites) + METHOD.replace('<section class="method">', '<section class="method" id="method">')
    (SITE / "index.html").write_text(page("Grape benchmark", suites_nav(order, suites, None, 0), body))
    data = {"suites": [{"corpus": k[0], "questions": k[1], "title": SUITES.get(k, (None,))[0],
                        "runs": [{"name": r["name"], **{m: stats(r, m) for m in MODES if stats(r, m)}} for r in suites[k]]}
                       for k in order]}
    (SITE / "data.json").write_text(json.dumps(data, indent=2))
    return SITE / "index.html"


if __name__ == "__main__":
    print(build())
