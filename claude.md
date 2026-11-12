# Patent Claim Search

## Goal

Build a search engine over a folder of patent JSON files.

* A user types a plain English question and gets back the patent **claims** that best match it.
* The user can optionally add keywords. When keywords are given, only claims that literally contain them are returned. This part must be exact and repeatable: same input, same output, every time.

Keep the build small and simple. No database server. The frontend is one plain HTML page.

## Input data

Files live in `data 2/patent_data_small/patents_*.json` (set in `config.RAW_DIR`, override with `PATENT_SEARCH_RAW_DIR`). Use all files, not a sample. Each file is a JSON list of patent records:

| Field | Type | How we use it |
|---|---|---|
| `doc_number` | string | Patent id. Primary key. |
| `title` | string | Embedded as context, shown in results |
| `abstract` | string | Embedded as the patent level text, shown on request |
| `claims` | list of strings | The search unit. One item per claim, in order |
| `detailed_description` | list of strings | Stored, not embedded |
| `classification` | string | Stored as is (example `B60C1112FI`), shown in results |
| `filename` | string | Parse the kind code (`A1`, `A9`) and the 8 digit publication date at the end (stored as `YYYY-MM-DD`) |
| `bibtex` | string | Ignore. It only repeats title, number, year and filename |

Full dataset: 64 files, 640 patents, 10,578 claim items. (The spec was first written on a 4 file, 40 patent sample; the rules below are checked on the full set.)

## Known data problems (must handle)

These were measured on the sample. The code must handle them in general, not just for these files.

1. **Empty values.** No field is JSON null in the sample, but future files may have real nulls. Treat all of these as empty: missing key, `null`, `""`, whitespace only, `[]`, and a list whose items are all blank.

2. **Empty descriptions.** 119 of 640 patents have no description at all (stored as `[""]`). About half of all description items (1,027 of 2,076) are blank spacer strings between paragraphs. Drop blank items. Leave an empty description empty and set `has_description = False`. Never fill missing text with invented text.

3. **Cancelled claims.** 167 claim items (1,560 claim numbers) are placeholders with no content, for example `"17 . (canceled)"` or a range. Ranges appear as `"6 - 9 . (canceled)"`, `"1 .- 15 . (canceled)"`, `"1 - 15 : (canceled)"` and `"1 - 10 (canceled)"`. Do not index them. They still use up claim numbers (the range example uses up 6, 7, 8 and 9).

4. **Claims with missing openings.** 3,200 of 10,578 claim items lost their claim number and their opening words. Example: the real claim is `"1 . A bassinet comprising: a base; a column assembly ..."` but the item only holds `"a base; a column assembly ..."`. This happens when the claim had nested parts in the source XML. Each list item is still exactly one claim, in the right order, so the number can be recovered from position:

   ```
   n = 0
   for item in claims:
       if item starts with a number (single "17 ." or range "6 - 9 ."):
           a, b = the numbers (b = a when there is no range)
           if a != n + 1: log a warning
           n = b
       else:
           n = n + 1        # this item is claim n, opening missing
   ```

   A number is `N .` with the period not followed by a digit, so a recovered item starting `"0.5 phr of ..."` is not read as claim 0. Two patents write the number as `N :` (`20240075771`) or `N )` (`20240116312`); these are accepted the same way (23 items), again only when no digit follows.

   **Source gaps.** On the full set 4 patents have a numbering gap or a claim out of order (`20240294037`, `20240391281`, `20250010663`, `20250065669`). When a run of unnumbered items is followed by a numbered claim that disagrees with forward counting, number the run backward from that claim (`8, ?, 11` makes the gap claim 10), as long as those numbers are free. Claim references in the source confirm this every time it can be checked. Log a warning. Claim ids must stay unique: a number that would clash gets the next free number above the highest, with a warning.

   Mark recovered claims with `opening_missing = True`. For these we cannot know whether they depend on another claim, so `parent_number` and `is_independent` are `None`.

5. **Spacing noise in claim text.** Text looks like `"claim 1 , wherein"` and `"1 . A spoke star"`. Fix spacing before commas and periods. Reference numbers in brackets like `"( 1 )"` or `"( 2a )"` appear in some claims: keep them in the display text, remove them from the embedding text. A reference bracket always has a space after `(` and holds only numbers with an optional letter or prime (`( 1 , 1 a )`, `( 56 A, 56 B; 57 A)`); unspaced brackets such as `formula (1)`, `(2018)` or `(i)` are real text and stay. Decimals written `0 . 75` become `0.75`.

6. **Duplicates.** Two different patents can share a title, so always key on `doc_number`. If the same `doc_number` appears more than once (for example an `A1` and a later `A9` correction), keep the record with the later publication date.

## Parsing claims

For each real (not cancelled) claim, produce:

* `claim_id`: `"{doc_number}:{claim_number}"`
* `doc_number`, `claim_number` (int)
* `text`: cleaned display text without the leading number
* `embed_text`: see below
* `parent_number`: first `claim N` found in the first 200 characters of the text (for `claims 1 to 3` take the first number). `None` if not found.
* `is_independent`: `True` if numbered and no parent; `False` if it has a parent; `None` if the opening is missing
* `root_number`: follow parent links upward until reaching a claim with no parent, or a parent that is missing or cancelled. Guard against loops. A claim with no parent is its own root. Five claims in the source refer to themselves (`20240149618` claim 7, `20240190172` claim 8, `20240278602` claim 32, `20240343066` claim 20, `20250058586` claim 20): keep the self reference as `parent_number`, stop there as the root, and log a warning.
* `opening_missing` (bool)

## What gets embedded

Two kinds of vectors, same model, all normalized.

**Claim vector** (one per real claim). The text is:

```
Patent: {title}
Main claim: {text of the root claim}
Claim: {this claim's text}
```

Include the `Main claim` line only when this claim depends on another claim (it has a parent and its root is a different claim, so a claim that refers to itself gets no main claim line). Leave out the `Patent` line when the title is empty.

The main claim line matters because a dependent claim like "The spoke star of claim 2, wherein the fastening means is connectable to the loop" means very little on its own. Remove bracketed reference numbers from both claim texts here.

**Patent vector** (one per patent): `{title}. {abstract}`. Use the title alone if the abstract is empty, and the abstract alone if the title is empty. If both are empty, store no vector and use a patent score of 0 for that patent.

**Not embedded:** `detailed_description` (long, missing for 119 of 640, and results are claims anyway), `classification`, `bibtex`, ids.

## Embedding model

Default: `nomic-ai/nomic-embed-text-v1.5` through `sentence-transformers` with `trust_remote_code=True`.

* Free, runs locally on a laptop CPU, 768 numbers per vector.
* Accepts long text (8,192 tokens), so a long claim plus its main claim is never cut off.
* It needs a prefix: `search_document: ` on everything we store, `search_query: ` on the user question.
* Pin the model revision in config so results stay the same over time.

Wrap the model in a small class with `embed_documents(texts)` and `embed_query(text)` so it can be swapped later (for example `BAAI/bge-base-en-v1.5`, which cuts off after 512 tokens, or OpenAI `text-embedding-3-small`, which is paid). Save the model name and revision in the index metadata. If the search config uses a different model (name, revision or code revision) than the saved index, stop with a clear message telling the user to rebuild (`IndexNotReady` in `search.py`). The model is only loaded when a query is given; keyword only searches never load it.

## Storage

Small data, keep everything in files and load it into memory at search time.

* `data/index/search.db` (SQLite)
  * `patents(doc_number PK, title, abstract, description, has_description, classification, kind_code, pub_date, source_file, vec_row, cpc_subclass, cpc_group, cpc_subgroup, cpc_code)`
  * `claims(claim_id PK, doc_number, claim_number, text, embed_text, parent_number, root_number, is_independent, opening_missing, vec_row)`
* `data/index/claim_vectors.npy` and `data/index/patent_vectors.npy`, rows matching `vec_row`
* `data/index/meta.json`: model name, revision, code revision, vector size, build time, counts
* `data/index/patent_map.json`: the map layout cache (see `GET /map`), written on first use, not by the build

The build writes every file under a `.tmp` name first and renames them into place at the end (`meta.json` last), so a failed build never leaves a half written index.

`description` is the non blank paragraphs joined with a blank line, or empty.

Keeping all vectors in memory is fine up to a few hundred thousand claims. Keep the vector lookup behind one function so it can be moved to FAISS or Qdrant later without touching the rest. In `search.py` this is the `VectorStore` class; nothing else touches the vectors. The index (SQLite rows, vectors, meta) is loaded once per process.

## Classification codes

`classification` is a CPC symbol stored without separators: `B60C1112FI` is CPC **B60C 11/12** (subclass `B60C`, main group `11`, subgroup `12`; the trailing `FI` is not part of the symbol). The digits alone are ambiguous (`1112` could be `1/112` or `11/12`), so a text prefix filter would be wrong: `B60C1` as a prefix also catches groups 11, 13, 15, 17 and 19. `cpc.py` splits codes properly.

* **Reference data.** `MAIN_GROUPS` lists the real main groups of B60B, B60C, B60D, B60F, B60G, B60J and B60R, taken from the USPTO CPC scheme pages (`https://www.uspto.gov/web/patents/classification/cpc/html/cpc-<SUBCLASS>.html`), without the indexing codes (2000 series). `SUBCLASS_TITLES` and `GROUP_TITLES` hold the scheme titles (cross references in brackets removed). A group whose title could not be read is left out rather than guessed. There are no subgroup titles.
* **Split rule** (`split_code`). Take the longest real main group that leaves a subgroup of at least 2 digits: `B60B1110` is `11/10` ("Emergency wheels"), `B60D1167` is `1/167` (B60D has no group 11), `B60C15024` is `15/024`. In a subclass not in `MAIN_GROUPS`, a code is split only when one reading is possible (3 digits, `A01B100` is `1/00`); otherwise only the subclass is kept and a warning is logged. All 145 distinct codes in the data split, with no warnings, and every split is a real symbol on the USPTO pages.
* **Queries** (`parse_query`). Accepts `B60C`, `B60C 1`, `B60C 11/12`, `B60C11/12` and the stored form `B60C1112` (split by the same rule). `/00` means the whole main group. An unknown main group in a known subclass, a one digit subgroup, or text that is not a CPC symbol is a clear `ValueError`.
* **Matching** (`matches(query, code)`). Whole parts are compared, never text prefixes: `B60C 1` matches only group 1. A subgroup also covers the subgroups under it: `15/02` covers `15/024`.
* **Stored columns.** `write_cpc_columns(db_path)` fills `cpc_subclass`, `cpc_group` (int), `cpc_subgroup` (digits after the slash, `00` for the main group itself) and `cpc_code` (`B60C 11/12`) in the patents table. The build calls it at the end; `python -m patent_search build --cpc-only` runs it on an existing `search.db` (adding the columns if missing) without re-embedding. A patent without a classification gets nulls.
* The result cards still show `classification` exactly as stored.

## Search

```python
search(query: str | None = None,
       keywords: str | list[str] | None = None,
       top_n: int = 10,
       classification: str | list[str] | None = None,
       date_from: str | None = None,
       date_to: str | None = None,
       independent_only: bool = False,
       group_by: str = "claim",
       title_abstract_keywords: str | list[str] | None = None) -> SearchResponse
```

At least one of `query`, `keywords` or a filter is required, otherwise raise a clear error. A filter alone is a valid search. A blank query counts as none. `top_n` must be a whole number from 1 to `TOP_N_MAX` (100). `group_by` must be `"claim"` or `"patent"`.

**Step 1: keyword filter and field filters (optional, exact). Runs first.** A claim passes only if it passes the keywords and every filter given (AND).

**Filters** (`make_filters(classification, date_from, date_to, independent_only, title_abstract_keywords)` in `search.py`, which checks the input and raises `ValueError` with a clear message):

* `classification`: CPC codes parsed with `cpc.parse_query` and compared with `cpc.matches` (whole parts, see Classification codes), so `B60C 1` matches main group 1 only. Several codes (a list, or one string with commas, `"B60C 11/12, B60D"`) match if any of them matches. Unknown main groups (`B60C 2`), one digit subgroups and text that is not a code are errors.
* `date_from`, `date_to`: inclusive, on `pub_date`. Each accepts `2024`, `2024-06` or `2024-06-15`; a partial date covers the whole year or month (`date_to = 2024-02` ends on 2024-02-29). Other formats, dates that do not exist (`2024-02-30`) and `date_from` after `date_to` are errors. A patent without a publication date is left out whenever a date filter is set.
* `title_abstract_keywords`: keeps the claims of patents whose title or abstract contains every keyword. Same parsing and matching rules as the claim keywords below (whole word, case insensitive, `"phrases"`, `*` for word starts, hyphen equals space, no stemming). Each keyword may be in either field, but is checked in the title and the abstract separately, so a phrase never runs from the end of the title into the abstract. A patent with neither field never matches. It is separate from `keywords`, which still match only the claim text; both can be given (AND). On the full data `sipe` matches 24 patents (407 claims) and `bassinet` 1 (46 claims), the same as a direct scan of the database.
* `independent_only`: keeps only claims with `is_independent = True` (120 in the data). Claims whose opening is missing have an unknown type (`is_independent = None`), so they are left out too, even though some of them are main claims.

**Keywords:**

* Every keyword must appear in the claim (AND across keywords).
* Match against the claim's own display text only, not the title and not the main claim (use `title_abstract_keywords` for the title and abstract). A dependent claim matches only on words it actually contains.
* Case insensitive, whole word.
* A phrase in double quotes matches that exact phrase.
* A trailing `*` matches word starts: `spoke*` matches spoke, spokes, spoked.
* A hyphen between words matches a space and the other way round: `fiber reinforced` matches `fiber-reinforced`.
* No stemming, no spelling variants (fibre vs fiber), no synonyms. Exact means exact.
* Highlights (`keywords.highlight_spans`): every claim in a response, claim hits and the claims inside patent hits, carries `highlights`, the `[start, end)` spans in its `text` where a keyword matched, by the same compiled patterns as the filter. Spans are sorted, overlapping or touching spans are merged, and offsets are Python string indices (Unicode code points). Without keywords the list is empty. Only the claim text is highlighted, never the title or main claim, because only the claim text is matched.
* Parsing (`keywords.py`, used by both the CLI and the API): in a string, spaces separate keywords and text in double quotes stays one phrase; an unclosed quote runs to the end. In a list, each item is one keyword or phrase (quotes are stripped). Blank keywords and a bare `*` are ignored. Unicode hyphens (‐ ‑) count as hyphens; an en dash does not.

**Step 2a: query given.** Embed the query with the `search_query: ` prefix. For each claim that passed step 1:

```
score = W_CLAIM * claim_similarity + W_PATENT * patent_similarity
```

Defaults `W_CLAIM = 0.8`, `W_PATENT = 0.2`, both in config. Scores are rounded to 6 decimals before sorting, so tiny floating point differences cannot reorder results. Sort by score high to low, then `doc_number`, then `claim_number` so ties always come out in the same order. Return the top `top_n`.

**Step 2b: no query (keywords and/or filters only).** Return matching claims sorted by `doc_number` then `claim_number`, first `top_n`, with no score.

**Grouping.** `group_by = "claim"` (default) returns claims. `group_by = "patent"` runs the same steps and scores, then groups the ranked claims by `doc_number` in ranked order: a patent ranks by its best claim (its score is that claim's score; ties follow the claim tie order, so `doc_number`). With no query, patents come in `doc_number` order with no score and their claims in claim number order. `top_n` counts patents.

**Response.** `group_by`, `total_matches` (claims left after step 1, or with `group_by = "patent"` the number of patents that have at least one such claim) and a list of hits.

* Claim hits: `doc_number`, `title`, `claim_number`, `text`, `score` (`None` without a query), `is_independent`, `opening_missing`, `pub_date`, `classification` (as stored), `cpc_code` (split, e.g. `B60C 11/12`, or `None`), `source_file`, `highlights`.
* Patent hits: `doc_number`, `title`, `abstract`, `score`, `matching_claims` (its claims that passed step 1), `claims` (its best `config.CLAIMS_PER_PATENT` = 3, each with `claim_number`, `text`, `score`, `is_independent`, `opening_missing`, `highlights`), `pub_date`, `classification`, `cpc_code`, `source_file`.

When `opening_missing` is true, show the text as it is. The flag lets the caller say the start of the claim is missing in the source.

## Interfaces

* Python function `search(...)` above.
* CLI through `python -m patent_search` (also `serve` to run the web page on port 8000; `--host` and `--port` override `config.HOST` and `config.PORT`):
  * `build`: read the raw dir (`config.RAW_DIR`), clean, embed, write `data/index` (`--cpc-only` only rewrites the CPC columns), print counts (patents, claims indexed, patents without description, claims with missing openings, cancelled claim numbers)
  * `search "<question>"` with an optional repeatable keyword option and an optional top option: `search "<question>" -k KEYWORD -n TOP --json`, plus `--by claim|patent`, `-c/--classification CODE` (repeatable, commas allowed, any code matches), `--from DATE`, `--to DATE`, `--main-only` (independent claims only) and `-t/--title-abstract KEYWORD` (repeatable, split by the same keyword parser; every keyword must be in the title or the abstract). The question is optional when `-k` or a filter is given. Each `-k` value is split by the same keyword parser as the API (`-k 'sipe "tread block"'` gives two keywords). `--json` prints the response as JSON.
* HTTP with FastAPI: `POST /search` with body `{"query": ..., "keywords": ..., "top_n": 10, "classification": ..., "date_from": ..., "date_to": ..., "independent_only": false, "group_by": "claim", "title_abstract_keywords": ...}` (all optional; `title_abstract_keywords` is a string or a list of strings, like `keywords`) returning the response above. `classification` is a string (commas allowed) or a list of strings; the dates are strings, or a number for a year. `keywords` may be a list of strings or one string. A string is split by the same function the CLI uses (spaces separate keywords, text in double quotes stays one phrase), so there is only one keyword parser. Errors come back as `{"error": "<message>"}`: `400` for bad input (no query, keyword or filter, `top_n` out of range, a bad code, date or `group_by`, wrong types, a body that is not JSON) and `503` when the index is missing or was built with a different model. The index loads on the first request, and the model on the first request with a query.
* `GET /classifications`: the dropdown data, built from the data in the index: `{"subclasses": [{code, count, title?, groups: [{code, count, title?, subgroups: [{code, count}]}]}]}`. `code` is a valid `classification` query, `count` is the number of patents it matches with the same `cpc.matches` as the filter (so `15/02` counts `15/024` too), and `title` is there only when the scheme title is known. Subgroups are in scheme order (`02` < `024` < `10`).
* `GET /map`: the patent map. `{"points": [{doc_number, x, y, title, pub_date, cpc_code, subclass, group}], "classifications": <same tree as /classifications>}`, one point per patent with a vector (all 640), `x` and `y` in 0 to 1. The layout (`landscape.py`) is t-SNE on the patent vectors with cosine distance, `init="pca"`, `MAP_SEED` (42) and `MAP_PERPLEXITY` (30) from config, so the same index always gives the same map. It takes about 2 seconds, so it is cached in `data/index/patent_map.json`, keyed by the index build time, model, seed and perplexity; a rebuild or a changed setting lays it out again.
* `POST /map/matches`: the `/search` body without `top_n` and `group_by`, returning `{"total_matches", "matches": [{doc_number, score, matching_claims, best_claim}]}` for **every** patent with a claim that passes (not cut at `TOP_N_MAX`), ranked as `group_by = "patent"` ranks them (`Index.patent_matches`, which shares steps 1 and 2 with `search`). Same errors as `/search`.
* Web page at `GET /` (see Frontend).

## Frontend

Keep it minimal. One file, `src/patent_search/static/index.html`, served by FastAPI at `GET /`. Plain HTML, CSS and JavaScript in that one file. No framework, no build step, nothing loaded from the internet.

**Page contents**

* A text box for the question.
* A "Show: Claims | Patents | Map" toggle under the search bar. Claims and Patents set `group_by` (`claim` or `patent`) and rerun the current search if one has run. Map switches to the map view (below) and runs at once, even with nothing typed.
* A collapsible "Exact filters" panel with:
  * a "Claim keywords" tag box (see Behavior), marked optional. Placeholder: `Type a keyword and press Enter, e.g. sipe` (`Add another` once it has a tag)
  * a "Title or abstract keywords" tag box (`title_abstract_keywords`), marked optional. Placeholder: `Type a keyword and press Enter, e.g. bassinet`
  * a number box labelled "Number of results" (claims or patents), default 10
  * three linked classification dropdowns, Subclass, Main group and Subgroup, filled from `GET /classifications`. Each option shows the code, the title when known and the patent count (`11 — Tyre tread bands; … (101)`). Each later dropdown is disabled until the one before it is chosen; changing one resets the ones after it. The most specific choice is sent as one `classification` code (`B60C`, `B60C 11` or `B60C 11/12`). Subgroups listed are the ones in the data (not `/00`, which is the group itself)
  * "Published from" and "Published to" text boxes (`2024`, `2024-06` or `2024-06-15`, inclusive; the server checks them). Each has a calendar button inside it that opens the browser's own date picker (`showPicker()` on a hidden `<input type="date">`); the chosen day replaces the box's text. The picker opens on what the box already says (a year or month at its first day for "from", its last day for "to"). Typing stays possible because a picker can only give a full day, while the filter also takes a year or a month
  * a "Clear filters" link
* A Search button. Pressing Enter in any text box also searches.

**Behavior**

* Sends the question, the claim keyword tags as a list (no tags sends no keywords), the number, `classification`, `date_from`, `date_to` and `group_by` to `POST /search`. The page has no main claims only option; `independent_only` is available in the API and the CLI (`--main-only`).
* Filters alone can search. If the question, keywords and every filter are empty, show "Type a question or a keyword, or choose a filter" and do not call the API.
* Sends the title or abstract tags as a list in `title_abstract_keywords` (no tags sends nothing). A question with title or abstract keywords is filtered, so the count line says `{n} matching claims`.
* The panel's badge counts the claim keyword tags and the title or abstract keyword tags plus the other filters set (classification, each date): "1 filter", "4 filters".
* The example chips clear the filters, then set the question; the Show toggle keeps its setting.
* While waiting, disable the button and show "Searching...".
* Above the results show `{total_matches} matching claims` (or `matching patents`), and `showing top {n}` when there are more than shown. When there is a question but no keyword and no filter, nothing is filtered out and every claim is ranked, so the line says `Top {n} claims by meaning` (or `patents`) instead, without the total, so it does not look as if every claim matched. There is no score cutoff: on the graded evaluation claims, nomic's scores for relevant and not relevant claims overlap and differ from question to question, so no fixed value separates them.
* Each claim result is a simple card with:
  * the patent title
  * patent number and claim number
  * the score as an orange `{score} match` pill at the top right, after the claim label, as a whole percentage (`79% match` for 0.79; hidden when there is no question)
  * a "main claim" or "dependent claim" label (no label when unknown)
  * the claim text
  * when `opening_missing` is true, a small grey note: "The start of this claim is missing in the source data."
* Each patent result is a card (patent mode only):
  * top left: the patent number badge and the publication date (on a phone the date wraps under the number); top right: an orange `{score} match` pill with the score as a whole percentage (`79% match`) (the best claim's score; hidden when there is no question)
  * the title. All-caps titles get a smaller, letter-spaced style; the text itself is never changed
  * the readable code (`cpc_code`, e.g. `B60C 11/12`, falling back to the stored code) as a small monospace chip, followed by its main group title when known (from `GET /classifications`; the page waits for it before showing patent cards)
  * the abstract clamped to 3 lines with "Show full abstract"
  * a small uppercase heading "Best matching claims" ("Matching claims" without a question) with `{shown} of {matching_claims}` on the right
  * each shown claim (up to 3) in a soft panel: bold "Claim N", a quiet "Main" or "Dependent" tag (none when unknown), a dashed "Start missing" tag when `opening_missing` (its tooltip is the full note "The start of this claim is missing in the source data."), its score on the right (when there is a question), and the text clamped to 3 lines with "Show full claim". The best claim's panel has a thin orange border (only when there is a question)
  * no score repeated at the bottom
* Claim keyword and title or abstract keyword boxes are tag inputs: Enter (or a comma) turns the typed text into a tag with an × to remove it; Backspace in an empty box removes the last tag; Enter in an empty box searches. Each tag is one keyword or one exact phrase (words typed together, quotes not needed), and the tags are sent as a list, so the server's list rules apply. Text still in the box when a search runs becomes a tag first. A repeated tag (ignoring case) is not added twice, and a blank or bare `*` is ignored.
* Keyword matches are marked with a soft yellow highlight (`<mark>`, a token for light and one for dark) in claim cards and in the claims of patent cards, using the `highlights` spans from the server; the page does not re-implement the keyword rules. Spans are code point offsets, so the page slices an array of code points (`Array.from(text)`), not the UTF-16 string, and builds the text from text nodes (no HTML from the data).
* A "See more" button under the results runs the same search again with 10 more (`top_n` + 10, up to `TOP_N_MAX`). Ranking is deterministic, so the results already shown keep their places. It shows only while the last search filled its `top_n` and there are more matches; at 100 it is replaced by a note that 100 is the most one search returns and keywords or filters narrow it down. It does not change the "Number of results" box, so a new search, an example chip or the Show toggle starts again from the box's number.
* No results: show "No claims matched." or "No patents matched."
* API errors: show the error message in red above the results.
* If `/classifications` cannot be loaded, the subclass dropdown says "Classifications unavailable" and the other searches still work.

**Map view** (Show: Map). Works with no input, which shows every patent. The line above the map says `{n} patents`, `Top {shown} of {total} patents highlighted` (with a question) or `{n} of {total} patents match` (keywords or filters only). One card:

* *Patent landscape*: a scatter of every patent from `GET /map`, colored by subclass with three hues (B60C blue, B60B orange, B60D green; the rest grey "Other"), each dot with a 2px ring in the card color. The legend shows each subclass with the first part of its scheme title in normal case ("Vehicle tyres") and its count. Main groups with 12 or more patents are labelled at their median position; ranks are placed first and a label that would overlap one already placed is skipped. With a search, `POST /map/matches` decides what is lit: with a question, the top N by score (N = the "Number of results" number) keep their color and grow, and the top 5 get rank numbers; with keywords or filters only, every match is lit. Everything else turns grey ("Outside the top N" or "Not matching" in the legend). Hovering picks the nearest point within 24px and shows the title, code and main group title, number and date, and with a search the score, rank, matching claims and best claim; a click pins it.
* While a new search loads, the map stays in place at reduced opacity. It redraws on resize. Errors show in the results area as for the other views.

**Look**

* Clean landing style: header with an orange "P" logo and "Patent Claim Search", a large two-line headline ("Patent claims," then "found by meaning." in orange `#e8622a`), the subtitle "Ask in plain English and get the patent claims that match. Add exact keywords when wording matters.", and a soft orange glow behind the hero.
* Large rounded search bar with a search icon and an orange Search button inside it.
* Pill chips under a "Try an example" label that run four of the example searches from "Done when": Carbon fiber spokes, Bassinet arm, Tire sensor setup, Wall-climbing robots (the data has no categories, so no category chips). The `sipe` and heavy truck searches have no chip; they are run by typing them.
* The Show toggle is a small segmented control labelled "Show" under the search bar.
* The filters live in the collapsible "Exact filters" panel (claim keywords on the first row, title or abstract keywords on the second, the three classification dropdowns on the next, then the dates, then the "Number of results" number with the "Clear filters" link), which shows a count badge when any filter is set. On a phone the fields stack.
* Patent cards are top-aligned in the grid and keep their own height (not stretched to the tallest card in the row).
* Result cards in a two column grid (one column on a phone): monospace patent number badge (`US 2024/0116310`), the main or dependent claim label in orange and the score pill at the top right, a bold title, then a line with claim number, classification and publication date (`Claim 22 · B60B3300FI · Published 2024-03-21`; the classification is shown exactly as stored, not reformatted), then the claim text clamped to 5 lines with "Show full claim".
* Show only fields the data has (no assignee or inventor). System font, nothing loaded from the internet.
* Follows the system setting with `prefers-color-scheme`. Light and dark are equally finished; neither is the main one.

## Evaluation

A model comparison runner, so the embedding model can be chosen on evidence. It saves rankings only; there is no hand-made ground truth yet. Grading (labels per question) is added in a later step (B3) and fed to the metric functions below.

* **Questions:** `tests/eval/questions.json`, 40 questions `q01`–`q40`, text only, no relevant claims listed. `q11`–`q40` were each written from one patent's abstract, asking about one point in plain words rather than copying its wording.
* **Candidates:** `bm25` plus every model in `config.EVAL_MODELS`:

  | Name | Model | Query prefix | Document prefix | Reads up to |
  |---|---|---|---|---|
  | `nomic` | `nomic-ai/nomic-embed-text-v1.5` (production) | `search_query: ` | `search_document: ` | 8,192 tokens |
  | `bge-small` | `BAAI/bge-small-en-v1.5` | `Represent this sentence for searching relevant passages: ` | none | 512 |
  | `bge-base` | `BAAI/bge-base-en-v1.5` | same as bge-small | none | 512 |
  | `patentsberta` | `AI-Growth-Lab/PatentSBERTa` | none | none | 512 |
  | `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | none | none | 256 |
  | `qwen3-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | `Instruct: Given a patent search question, retrieve patent claims that answer it` + line break + `Query:` (no space after `Query:`, Qwen's own format) | none | 8,192 (capped) |

  Each has a pinned revision in config. An entry may also set `batch_size` (default `EVAL_BATCH_SIZE`, 32) and `max_seq_length` (default: the model's own limit). `qwen3-0.6b` is 0.6B parameters, 1,024 numbers per vector: batches of 16 overflow 16 GB of memory on mps, so it uses 4. Its own limit is 131,072 tokens; it is capped at nomic's 8,192 so the two long-context models read the same, and nothing is cut either way (the longest stored text is 1,151 Qwen tokens; bge and PatentSBERTa cut 152 texts at 512). It loads through `SentenceTransformer` with the model's own modules (Transformer, last-token Pooling, Normalize), not mean pooling; `Embedder.pooling` reads the mode from the loaded Pooling module and the report records it (`lasttoken` for Qwen, `mean` for MiniLM). Last-token pooling finds the last real token from the attention mask, so padding in a batch does not change a vector (checked: cosine 1.0). `nomic` reuses `data/index`; the others embed the same claim and patent texts the build stored (`claims.embed_text`, and `{title}. {abstract}`).
* **Same search for everyone.** Every candidate goes through the production `Index.search` with the same weights (`W_CLAIM`, `W_PATENT`), filter and tie order; only the vectors change (`Index.with_store`). `bm25` is `BM25Store` in `evaluate.py`, shaped like `VectorStore`: Okapi BM25 (`BM25_K1 = 1.5`, `BM25_B = 0.75`) over lowercase word tokens of the same texts, with each query's scores divided by the best document's score so they lie in 0 to 1 like a cosine similarity.
* **Cache.** Vectors for non-nomic models are cached in `data/eval_cache/<name>/` (git ignored), keyed by model, revision, code revision, document prefix, `max_seq_length` (only when the entry sets one, so older caches stay valid) and a SHA-256 of the claim and patent texts. Any change re-embeds.
* **CLI:** `python -m patent_search eval [--models NAME ...] [--device DEVICE] [-n TOP]`. Defaults: bm25 plus every model in `EVAL_MODELS`, the device is `mps` when available (else `cpu`), top 10. It prints the top 3 per question per candidate.
* **Report:** `data/eval_results/run_<UTC time>.json` with the index meta, settings, questions, each candidate's model, revision, prefixes, batch size, max length, pooling, where its vectors came from (`data/index`, `cache` or `embedded`) and run time, and for every candidate and question the top 10 `claim_id`s with scores.
* **Metrics** (`evaluate.py`), each taking the ranked claim ids and a `{claim_id: grade}` dict. Grades are 0 (not relevant), 1 (partly), 2 (fully); claims without a grade count as 0. Relevant means grade ≥ `EVAL_RELEVANT_GRADE` (1).
  * `hit@5`: 1 if any relevant claim is in the top 5.
  * `p@5`: relevant claims in the top 5, divided by 5.
  * `recall@10`: share of all relevant claims found in the top 10 (`None` if the question has none).
  * `mrr`: 1 / rank of the first relevant claim (0 if none).
  * `ndcg@10`: graded gain `2^grade - 1` (`None` if no claim has a positive grade).
  * `first_full_rank`: rank of the first claim with grade ≥ `EVAL_FULL_GRADE` (2), `None` if none is in the list.

### Pooled blind grading

The only source of relevance labels (`grading.py`, `python -m patent_search grade prepare | score`).

1. **Pool.** For each question, the union of every candidate's top 10 from the latest eval report (`data/eval_results/run_*.json`, or `--report`).
2. **Blind packets.** `grade prepare` writes `data/eval_judgments/packets/<qid>.json`: the question, the grading instructions, and for each pooled claim that has no grade yet its `claim_id`, patent title, main claim text (dependent claims only: the text of its root claim), claim text and `opening_missing`. The claims are in a fixed random order (seeded from the question id). No model names, scores or ranks.
3. **Reuse.** Grades live in `data/eval_judgments/grades/<qid>.json` and are never changed once written. Only pooled claims without a grade go into a packet, so a new model only needs its new claims graded. A question with nothing left to grade gets no packet.
4. **Judges.** One blind subagent per packet. It may read only its packet and write only `data/eval_judgments/incoming/<qid>.json`; it never sees which model returned a claim or any eval result. Scale: **2** = the claim itself states what is asked, **1** = right topic and a closely related detail, **0** = not relevant. It judges the claim's own text (the main claim is context only) and gives one short reason per claim.
5. **Score.** `grade score` checks each incoming file (every packet claim graded exactly once, grade 0, 1 or 2, a reason, nothing extra), merges it into `grades/`, writes all pooled grades to `data/eval_judgments/pooled_grades.json`, scores every candidate with the metric functions above (claims never graded count as 0, with a warning) and prints the comparison table and the rank of the first fully relevant claim per question. The result is saved to `data/eval_results/scores_<UTC time>.json`.

**Results** (`scores_20261111T064218Z.json`, from `run_20261111T063744Z.json`, all 7 candidates, 40 questions, 1,092 graded claims):

| Model | Hit@5 | P@5 | R@10 | MRR | nDCG@10 | Full@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.825 | 0.645 | 0.460 | 0.739 | 0.544 | 25/40 |
| nomic (production) | 0.925 | 0.770 | 0.542 | 0.857 | 0.683 | 34/40 |
| bge-small | 0.875 | 0.710 | 0.473 | 0.818 | 0.572 | 26/40 |
| bge-base | 0.925 | 0.750 | 0.510 | 0.879 | 0.585 | 28/40 |
| patentsberta | 0.925 | 0.775 | 0.522 | 0.833 | 0.641 | 30/40 |
| minilm | 0.875 | 0.700 | 0.463 | 0.797 | 0.572 | 27/40 |
| qwen3-0.6b | 0.975 | 0.805 | 0.603 | 0.898 | 0.720 | 35/40 |

`Full@10` counts the questions with a fully relevant claim in the top 10. History: the first round graded 285 pooled claims for six candidates on `q01`–`q10`; adding `qwen3-0.6b` (embedding took 41 minutes on mps at batch size 4, about 6.4 GB) added 14 claims. Then 30 questions (`q11`–`q40`) were added, each written from one patent's abstract across 25 CPC main groups (B60F, B60G, B60J and B60R have one patent each and are left out), avoiding the topics of `q01`–`q10`. Their 793 pooled claims were graded by 30 blind judges with the same prompt (one judge missed a claim, the check caught it, and the same judge graded it). The 299 earlier grades did not change, and every candidate's rankings for `q01`–`q10` are identical to the previous run, so the old questions score exactly as before.

What 40 questions show:

* nomic now clearly beats bm25 (nDCG 0.683 vs 0.544, Full@10 34 vs 25), and leads the other small models on nDCG and Full@10; on 10 questions it looked no better than bge-base or PatentSBERTa.
* Qwen is first on every metric, but the margin over nomic is small. Per question (paired, 10,000 bootstrap resamples of the 40 questions): nDCG +0.037 (95% interval -0.031 to +0.109; Qwen better on 21 questions, worse on 18), MRR +0.041 (-0.044 to +0.133), P@5 +0.035 (-0.045 to +0.125), recall +0.061 (+0.002 to +0.128). Only recall is above zero, and only just. On the 30 new questions alone the two are nearly level (nDCG 0.735 vs 0.722, MRR 0.892 vs 0.890), so most of Qwen's lead comes from `q01`–`q10`.
* By the rule below (switch only on a clear win), nomic stays: the gain is within noise on 3 of 4 metrics. Switching would not need re-embedding (the Qwen vectors in `data/eval_cache/qwen3-0.6b/` were made from the same texts and could be copied into `data/index` with a new `meta.json`; the build has no step for that yet), but every search with a question would load and run a 0.6B model instead of nomic, and the vectors grow to 1,024 numbers (nomic 768). Qwen's question embedding time has not been measured.
* No model finds a fully relevant claim for `q14`, `q16` and `q32`. Their judges gave no grade 2 at all: no claim in the pool states the asked point in its own text (for `q32`, all 23 pooled claims are tread designs graded 1).

**Limits.**

* Only pooled claims are graded. A relevant claim that no candidate returned in its top 10 is invisible, so recall is recall within the pool, and a model that finds relevant claims nobody else found is not rewarded for them until they are graded.
* 40 questions and one judge per claim. The judges are language models, not patent examiners; their grades are consistent within a packet but not checked by a second judge. Small differences between models are noise.
* A model is only worth switching to if it clearly beats the current one (nomic) on these metrics, not by a hair; switching means a rebuild and a different cost and context length.
* If nomic does not beat bm25, the embedding step is not earning its cost and plain word matching should be considered instead.

## Timing

`python -m patent_search.benchmark [REPEATS]` (one file, `benchmark.py`, cases listed in it) times the production `Index.search` on the evaluation questions, with no filters and with each hybrid filter, REPEATS (default 20) timed runs per question per case after one warm-up. Embedding the question costs the same whatever the filters are, so it is timed on its own and the search timings reuse a vector embedded up front: they cover the keyword and filter step, scoring the claims that passed it, sorting and building the response. Each filtered case is also timed without a question ("filters only": the exact step alone). It prints the table; nothing is saved.

Results on the full data (10,411 claims, 768-number vectors, Apple silicon Mac), measured when there were 10 questions (`q01`–`q10`):

| Case | Claims scored | Median ms | p95 ms | vs no filters | Filters only ms |
|---|---|---|---|---|---|
| No filters (semantic only) | 10,411 | 8.20 | 9.21 | 1.00x | - |
| No filters, grouped by patent | 10,411 | 10.39 | 12.68 | 1.27x | - |
| Classification `B60B` | 4,782 | 3.95 | 4.60 | 0.48x | 0.89 |
| Classification `B60C 11` | 1,658 | 1.56 | 1.75 | 0.19x | 0.62 |
| Claim keyword `tire*` | 3,383 | 32.95 | 37.17 | 4.02x | 30.30 |
| Claim keyword `sipe` | 222 | 43.54 | 45.48 | 5.31x | 43.19 |
| Title or abstract keyword `wheel*` | 4,495 | 7.20 | 7.59 | 0.88x | 4.37 |
| Published 2024-06 to 2024-12 | 4,884 | 3.80 | 4.20 | 0.46x | 0.80 |
| Main claims only | 120 | 0.48 | 0.50 | 0.06x | 0.41 |
| All: `B60B` + `spoke*` + `wheel*` + 2024 | 284 | 11.86 | 12.61 | 1.45x | 11.44 |

Embedding one question: median 13.3 ms (p95 24.3 ms). Once per process: loading the index 0.5 s, loading the model 8 s.

**What it shows.**

* The exact step runs before scoring, so a filter that removes claims also removes their scoring. Filters decided per patent (classification, dates) or by a stored flag (main claims only) are cheap, so with them a search is faster than without: `B60B` halves the time, `B60C 11` takes a fifth, main claims only takes 0.5 ms.
* Claim keywords are the slow part, 4 to 5 times slower than no filter. The keyword filter is a regular expression run over every claim text (3.1 million characters) on every search, about 30 to 43 ms whatever the question. A rare word is the slowest (`sipe`, 43 ms for 222 matches) because the search only stops early in texts that contain the word. Title or abstract keywords scan 640 short texts, about 4 ms.
* Without filters, the vector work is small: copying the selected rows and multiplying takes 1.7 ms (multiplying the whole matrix without the copy, 0.5 ms). Most of the 8 ms is Python looping over all 10,411 claims (building the list that passed, rounding every score, sorting all of them).
* With a question, embedding it (13 ms) costs more than the search itself unless claim keywords are set.

**Already used to keep it fast.** Filters run first and scoring runs only on what passed. Patent filters are checked once per patent (640), not per claim. CPC codes are split once when the index loads. Vectors are normalized when stored, so similarity is one matrix product. The model loads only when a question is given; keyword and filter searches never load it.

**What would make it faster (not built).**

* An inverted index for keywords (word to claim ids, for example SQLite FTS5, Tantivy or Lucene). Intersect the id lists, then run the exact regular expression only on those candidates, so the phrase, `*` and hyphen rules stay exact while the cost follows the number of matches instead of the size of the collection.
* Filters and scores as numpy arrays: a boolean mask per classification part and date, the scores computed for all claims at once without copying rows, and `np.argpartition` to take the top `top_n` instead of sorting every claim.
* At 10 million patents (well over 100 million claims) exact scoring of every claim is too slow, so an approximate nearest neighbour index (HNSW or IVF-PQ in FAISS or Qdrant) is needed. A filter then decides the plan: when it is selective (a narrow CPC group, a rare keyword), score its candidates exactly, which is what makes hybrid search faster than plain semantic search; when it is broad, search the ANN index with the filter applied inside it (filtered HNSW, Qdrant payload filters) instead of filtering afterwards, which can leave fewer than `top_n` results.

## Project layout

```
patent_search/
  CLAUDE.md
  pyproject.toml
  data 2/patent_data_small/ input JSON files (all 64)
  data/index/               built files, git ignored
  data/eval_cache/          cached vectors for model comparison, git ignored
  data/eval_results/        model comparison reports and scores
  data/eval_judgments/      blind packets, grades and pooled_grades.json
  src/patent_search/
    config.py               paths, model name and revision, weights, top_n, eval models
    load.py                 read JSON, empty value handling, cleaning, duplicates
    claims.py               claim number recovery, cancelled claims, parents, roots
    embedder.py             model wrapper
    build_index.py
    keywords.py             keyword parsing and matching
    cpc.py                  CPC code splitting, query parsing and matching, scheme titles
    search.py
    landscape.py            patent map layout (t-SNE of the patent vectors, cached next to the index)
    evaluate.py             model comparison: BM25 baseline, cached vectors, metrics
    grading.py              pooled blind grading and scoring
    benchmark.py            timing with and without the hybrid filters (python -m patent_search.benchmark)
    api.py
    __main__.py             CLI
    static/index.html       the web page
  tests/
    eval/questions.json     the evaluation questions
```

Stack: Python 3.11, `sentence-transformers`, `numpy`, `scikit-learn` (t-SNE for the map; installed with sentence-transformers), `fastapi`, `uvicorn`, `pytest`. Nothing else unless needed. Pin `transformers>=4.51,<5`: Qwen3-Embedding needs 4.51 or newer, and the nomic model's remote code breaks on transformers 5. `sentence-transformers` must be 2.7 or newer for Qwen (installed: 5.7.0, with transformers 4.57.6).

## Rules

* Never fill missing text with made up content.
* Keep display text and embedding text separate.
* All tunable values live in `config.py`.
* When a feature is added later, update this file so it always describes the project as it is.

## Done when

`python -m patent_search build` on the full data prints: 640 patents, 10,411 claims indexed, 119 patents without description, 3,200 claims with missing openings. (1,560 claim numbers are cancelled.)

These give sensible top results:

* `"wheel spoke made of carbon fiber"` (expect the V shaped wheel spoke patent near the top)
* `"bassinet arm that rotates the infant tub"`
* `"setting up a tire pressure sensor from a phone"`
* `"robot that drives up walls using magnetic wheels"`
* keywords only: `sipe`
* query `"tire for heavy trucks"` with keyword `tread`

The web page at `http://localhost:8000/` runs all of the searches above and shows the same results as the CLI.
