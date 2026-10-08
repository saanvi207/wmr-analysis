"""Step 4: turn cleaned blocks into paragraphs and sentences, with chapter tags.

Outputs:
  data/processed/paragraphs.parquet
  data/processed/sentences.parquet    the main corpus used by every analysis
  outputs/tables/corpus_summary.csv   pages, paragraphs, sentences, words per edition

Every sentence row has:
  edition, chapter_no, chapter_title, chapter_type, author, page,
  para_id, sent_id, text, is_heading, in_references, n_words

Reference lists and headings are kept but flagged, so you can filter them:
  df[~df.is_heading & ~df.in_references]

Usage:
  python src/build_corpus.py [editions...]
"""
import re
import sys

import pandas as pd
import spacy

from common import (CLEAN_DIR, PROCESSED, TABLES, ensure_dirs, load_chapters,
                    load_editions)

REFERENCES = re.compile(r"^\s*(references|bibliography|works cited|endnotes|notes)\s*$", re.I)
AFTER_REFS = re.compile(r"^\s*(appendix|annex)\b", re.I)
SENTENCE_END = re.compile(r"[.!?:;][\"'”’)\]]*\s*$")
WORD = re.compile(r"[A-Za-z]+")
COMPOUND = re.compile(r"[A-Za-z]+-[A-Za-z]+")


# ---------------------------------------------------------------- text repair
def build_vocab(texts):
    """Every word that appears unbroken somewhere in the corpus."""
    vocab = set()
    for t in texts:
        t = re.sub(r"[A-Za-z]+-\s*\n\s*[A-Za-z]+", " ", t)  # skip split fragments
        vocab.update(w.lower() for w in WORD.findall(t))
        vocab.update(c.lower() for c in COMPOUND.findall(t))  # e.g. 'gender-based'
    return vocab


def join_lines(text, vocab):
    """Join a block's lines, repairing words hyphenated at line ends.

    'migra-' + 'tion' becomes 'migration' (the joined word appears elsewhere
    in the corpus, or 'migra' is not a word on its own). 'gender-' + 'based'
    stays 'gender-based' because both halves are words and 'genderbased'
    never appears.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    out = lines[0]
    for nxt in lines[1:]:
        m = re.search(r"([A-Za-z]+)-$", out)
        first = WORD.match(nxt)
        if m and first:
            left, right = m.group(1).lower(), first.group(0).lower()
            real_compound = f"{left}-{right}" in vocab or (
                (left + right) not in vocab and left in vocab and right in vocab)
            out = out + nxt if real_compound else out[:-1] + nxt
        else:
            out = out + " " + nxt
    return re.sub(r"\s+", " ", out).strip()


# ------------------------------------------------------------- chapter mapping
def chapter_lookup(chapters, edition):
    """Function mapping a page number to its chapter row for one edition."""
    ch = chapters[chapters["edition"] == edition].sort_values("start_page")
    if ch.empty:
        print(f"  WARNING: no chapters listed for {edition} in config/chapters.csv; "
              "all text will be tagged as chapter 0.")
    starts = ch["start_page"].tolist()
    rows = ch.to_dict("records")
    front = {"chapter_no": 0, "chapter_title": "Front matter",
             "chapter_type": "front_matter", "author": "IOM"}

    def lookup(page):
        idx = sum(s <= page for s in starts) - 1
        return rows[idx] if idx >= 0 else front
    return lookup


# ------------------------------------------------------------------ paragraphs
def build_paragraphs(blocks, edition, chapter_of, vocab):
    paras = []
    current = None
    in_refs = False
    last_chapter = None

    for b in blocks.itertuples():
        text = join_lines(b.text, vocab)
        if not text:
            continue
        ch = chapter_of(b.page)
        if ch["chapter_no"] != last_chapter:
            in_refs, last_chapter, current = False, ch["chapter_no"], None

        if b.block_type == "heading":
            if REFERENCES.match(text):
                in_refs = True
            elif AFTER_REFS.match(text):
                in_refs = False
            current = None
            paras.append({"chapter": ch, "text": text, "pages": [(0, b.page)],
                          "is_heading": True, "in_references": in_refs})
            continue

        # Continue the previous paragraph if it stopped mid-sentence
        # (text flowing across a column or page break).
        if current is not None and not SENTENCE_END.search(current["text"]):
            current["pages"].append((len(current["text"]) + 1, b.page))
            current["text"] += " " + text
        else:
            current = {"chapter": ch, "text": text, "pages": [(0, b.page)],
                       "is_heading": False, "in_references": in_refs}
            paras.append(current)

    rows = []
    for i, p in enumerate(paras):
        ch = p["chapter"]
        rows.append({
            "edition": edition,
            "chapter_no": ch["chapter_no"], "chapter_title": ch["chapter_title"],
            "chapter_type": ch["chapter_type"], "author": ch["author"],
            "page": p["pages"][0][1], "page_offsets": p["pages"],
            "para_id": f"{edition}_{i:05d}", "text": p["text"],
            "is_heading": p["is_heading"], "in_references": p["in_references"],
        })
    return pd.DataFrame(rows)


def page_at(offsets, char):
    page = offsets[0][1]
    for start, pg in offsets:
        if start <= char:
            page = pg
    return page


# ------------------------------------------------------------------- sentences
def load_nlp():
    # Only the fast sentence splitter is needed here; full parsing comes later.
    nlp = spacy.load("en_core_web_sm",
                     exclude=["parser", "ner", "lemmatizer", "attribute_ruler", "tagger"])
    nlp.enable_pipe("senter")
    nlp.max_length = 2_000_000
    return nlp


def split_sentences(paras, nlp):
    meta = ["edition", "chapter_no", "chapter_title", "chapter_type", "author",
            "para_id", "is_heading", "in_references"]
    rows = []
    texts = paras["text"].tolist()
    for p, doc in zip(paras.itertuples(), nlp.pipe(texts, batch_size=64)):
        sents = [doc[:]] if p.is_heading else list(doc.sents)
        for j, s in enumerate(sents):
            text = s.text.strip()
            if not text:
                continue
            row = {m: getattr(p, m) for m in meta}
            row.update({"page": page_at(p.page_offsets, s.start_char),
                        "sent_id": f"{p.para_id}_{j:03d}", "text": text,
                        "n_words": len(WORD.findall(text))})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    chapters = load_chapters()
    editions = load_editions(sys.argv[1:])
    blocks = {}
    for ed in editions["edition"].astype(int):
        path = CLEAN_DIR / f"{ed}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            blocks[ed] = df[df["block_type"] != "dropped"]
        else:
            print(f"  [skip] {ed}: run clean_blocks.py first")
    if not blocks:
        return

    print("Building vocabulary for hyphen repair...")
    vocab = build_vocab(pd.concat(blocks.values())["text"])
    nlp = load_nlp()

    all_paras, all_sents, summary = [], [], []
    for ed, df in blocks.items():
        paras = build_paragraphs(df, ed, chapter_lookup(chapters, ed), vocab)
        sents = split_sentences(paras, nlp)
        prose = sents[~sents["is_heading"] & ~sents["in_references"]]
        summary.append({"edition": ed, "pages": int(df["page"].max()),
                        "paragraphs": int((~paras["is_heading"]).sum()),
                        "sentences": len(prose), "words": int(prose["n_words"].sum()),
                        "reference_sentences": int(sents["in_references"].sum())})
        print(f"  {ed}: {len(prose):,} sentences, {prose['n_words'].sum():,} words")
        all_paras.append(paras)
        all_sents.append(sents)

    # Merge with any editions built earlier so you can rebuild one at a time.
    for name, new, key in [("paragraphs", all_paras, "para_id"),
                           ("sentences", all_sents, "sent_id")]:
        out = PROCESSED / f"{name}.parquet"
        new = pd.concat(new, ignore_index=True).drop(columns=["page_offsets"], errors="ignore")
        if out.exists():
            old = pd.read_parquet(out)
            new = pd.concat([old[~old["edition"].isin(blocks.keys())], new], ignore_index=True)
        new.sort_values(["edition", key], kind="stable").to_parquet(out, index=False)

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(TABLES / "corpus_summary.csv", index=False)
    print("\nSaved data/processed/sentences.parquet and outputs/tables/corpus_summary.csv")


if __name__ == "__main__":
    main()
