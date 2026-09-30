"""Download a set of Wikipedia articles as plain text into corpus/<set>/ (one paragraph per line).

Usage: uv run tools/fetch_corpus.py [space|mixed|fresh|gujarati|large]
  gujarati: Gujarati Wikipedia articles about Gujarat.
  large:    300 long English articles (~20 MB) from one Wikipedia dump shard, for the large-context test.
"""

import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

SETS = {
    "space": [
        "Apollo 11", "Apollo 13", "Voyager 1", "Voyager 2", "Hubble Space Telescope",
        "James Webb Space Telescope", "International Space Station", "Curiosity (rover)",
        "Perseverance (rover)", "Sputnik 1", "Vostok 1", "Space Shuttle", "Cassini–Huygens",
        "New Horizons", "Falcon 9",
    ],
    "mixed": [
        "Coffee", "Honey bee", "Chess", "Great Wall of China", "Photosynthesis", "Bitcoin",
        "Mount Everest", "Penicillin", "The Beatles", "Leonardo da Vinci", "Tsunami", "Kimchi",
    ],
    "fresh": [
        "Octopus", "Eiffel Tower", "Amazon River", "Marie Curie", "Olympic Games", "Vaccine",
        "Sahara", "Silk Road", "Tea", "Printing press", "Black hole", "Samurai",
    ],
    # Searched on gu.wikipedia.org; the top hit is taken.
    "gujarati": [
        "ગુજરાત", "અમદાવાદ", "મહાત્મા ગાંધી", "વલ્લભભાઈ પટેલ", "સોમનાથ મંદિર", "કચ્છનું રણ",
        "નર્મદા નદી", "ગીર રાષ્ટ્રીય ઉદ્યાન", "ગરબા", "ઉત્તરાયણ", "સાબરમતી આશ્રમ", "ગિરનાર",
    ],
    "large": [],
}
LANG = {"gujarati": "gu"}
# The API allows one full article per request and rate-limits hard; one dump shard is much faster.
SHARD = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00000-of-00041.parquet"
LARGE, LONG = 300, (40_000, 120_000)

name_set = sys.argv[1] if len(sys.argv) > 1 else "space"
ARTICLES = SETS[name_set]
API = f"https://{LANG.get(name_set, 'en')}.wikipedia.org/w/api.php?"
ROOT = pathlib.Path(__file__).resolve().parents[1]
out = ROOT / "corpus" / name_set
out.mkdir(parents=True, exist_ok=True)



def get(req):
    for wait in (5, 15, 30, 60, 120):
        try:
            return json.load(urllib.request.urlopen(req))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(wait)
    raise SystemExit("Wikipedia keeps rate-limiting, try again later")


def api(**params):
    q = urllib.parse.urlencode({"action": "query", "format": "json", "formatversion": 2, **params})
    return get(urllib.request.Request(API + q, headers={"User-Agent": "grape-bench/0.1 (benchmark corpus fetch)"}))


def safe(title):
    return "".join(c for c in title.replace(" ", "_") if c not in '()/:"?*') + ".txt"


if name_set == "large":
    import pyarrow.parquet as pq

    shard = ROOT / "cache" / "wiki-en-shard0.parquet"
    if not shard.exists():
        shard.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SHARD, shard)
    table = pq.read_table(shard, columns=["title", "text"]).to_pylist()
    picked = [a for a in table if LONG[0] <= len(a["text"]) <= LONG[1]][:LARGE]
    for a in picked:
        (out / safe(a["title"])).write_text(a["text"])
    print(f"{len(picked)} articles, {sum(len(a['text']) for a in picked):,} chars")
    sys.exit()
elif name_set in LANG:
    # Local-language titles vary; take the best search hit for each topic.
    ARTICLES = [api(list="search", srsearch=t, srlimit=1)["query"]["search"][0]["title"] for t in ARTICLES]

for title in ARTICLES:
    name = safe(title)
    if (out / name).exists():
        continue
    time.sleep(1.5)
    text = api(prop="extracts", explaintext=1, redirects=1, titles=title)["query"]["pages"][0].get("extract", "")
    if not text:
        continue
    (out / name).write_text(text)
    print(f"{name}: {len(text):,} chars")
