# Corpus

Plain-text copies of Wikipedia articles, fetched by `tools/fetch_corpus.py`:

- `fresh/`, `space/`, `mixed/`: English Wikipedia, via the MediaWiki API.
- `gujarati/`: Gujarati Wikipedia (gu.wikipedia.org), via the MediaWiki API.
- `squad/`: the Wikipedia paragraphs of the SQuAD 2.0 dev set (CC BY-SA 4.0), for `tools/recall.py`.
- `large/`: 300 long articles from the `wikimedia/wikipedia` dataset on Hugging Face (English, 20231101 dump, shard 0).

Wikipedia text is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each file is named
after its article; the article history on Wikipedia lists its authors. The text is unchanged apart from being converted to
plain text. It is kept here so the benchmark can be re-run on exactly the same documents.
