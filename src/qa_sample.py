"""Step 5: hand-check extraction quality on a random sample of pages.

  python src/qa_sample.py              # sample 20 pages (seed 42)
  python src/qa_sample.py --n 30       # sample a different number
  python src/qa_sample.py --report     # after filling in the log: error rates

For each sampled page this saves, in data/qa/:
  <edition>_p<page>.png    an image of the real PDF page
  <edition>_p<page>.txt    what the pipeline extracted, plus what it dropped
and creates qa_log.csv. Compare each image with its .txt file, and put 1 in
a column when you see that problem on the page (0 otherwise).
"""
import argparse

import pandas as pd
import pymupdf

from common import CLEAN_DIR, PROCESSED, QA_DIR, RAW, ensure_dirs, load_editions

PROBLEMS = ["missing_text", "wrong_order", "header_footer_leak", "table_leak",
            "broken_words", "wrong_chapter"]
LOG = QA_DIR / "qa_log.csv"


def sample(n, seed):
    sents = pd.read_parquet(PROCESSED / "sentences.parquet")
    pages = sents[~sents["in_references"]][["edition", "page"]].drop_duplicates()
    picked = pages.sample(n=min(n, len(pages)), random_state=seed).sort_values(["edition", "page"])
    files = load_editions().set_index("edition")["filename"]

    rows = []
    for ed, pg in picked.itertuples(index=False):
        stem = f"{ed}_p{pg:03d}"
        doc = pymupdf.open(RAW / files[ed])
        doc[pg - 1].get_pixmap(dpi=110).save(QA_DIR / f"{stem}.png")

        page_sents = sents[(sents["edition"] == ed) & (sents["page"] == pg)]
        blocks = pd.read_parquet(CLEAN_DIR / f"{ed}.parquet")
        dropped = blocks[(blocks["page"] == pg) & (blocks["block_type"] == "dropped")]
        chapter = page_sents["chapter_title"].iloc[0] if len(page_sents) else "?"

        with open(QA_DIR / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(f"EDITION {ed}  PAGE {pg}  CHAPTER: {chapter}\n")
            f.write("=" * 70 + "\nKEPT (in reading order)\n" + "=" * 70 + "\n")
            for s in page_sents.itertuples():
                tag = "[HEADING] " if s.is_heading else ""
                f.write(f"{tag}{s.text}\n")
            f.write("\n" + "=" * 70 + "\nDROPPED\n" + "=" * 70 + "\n")
            for b in dropped.itertuples():
                f.write(f"[{b.drop_reason}] {b.text[:200].replace(chr(10), ' ')}\n")
        rows.append({"edition": ed, "page": pg, "file": stem,
                     **{p: "" for p in PROBLEMS}, "notes": ""})

    if LOG.exists():
        print(f"{LOG.name} already exists; new sample saved as qa_log_new.csv")
        pd.DataFrame(rows).to_csv(QA_DIR / "qa_log_new.csv", index=False)
    else:
        pd.DataFrame(rows).to_csv(LOG, index=False)
    print(f"Saved {len(rows)} pages to data/qa/. Fill in {LOG.name}, then run with --report.")


def report():
    log = pd.read_csv(LOG)
    filled = log[PROBLEMS].notna().all(axis=1)
    if not filled.all():
        print(f"{(~filled).sum()} pages still have blank columns; checking the rest.")
    log = log[filled]
    if log.empty:
        return
    any_error = (log[PROBLEMS].astype(int).sum(axis=1) > 0)
    print(f"Pages checked: {len(log)}")
    print(f"Pages with at least one problem: {any_error.sum()} "
          f"({100 * any_error.mean():.1f}%)")
    print("\nProblem counts:")
    for p in PROBLEMS:
        print(f"  {p:<20} {int(log[p].sum())}")


def main():
    ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report() if args.report else sample(args.n, args.seed)


if __name__ == "__main__":
    main()
