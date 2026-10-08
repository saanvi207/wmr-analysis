"""Step 1: pull raw text blocks out of every PDF.

For each edition this writes:
  data/interim/blocks/<edition>.parquet  one row per text block, with position,
                                         font size and a table flag
  data/interim/toc/<edition>.csv         the PDF's built-in table of contents
and adds starter rows to config/chapters.csv for you to check and complete.

Usage:
  python src/extract_blocks.py              # all editions in data/raw/
  python src/extract_blocks.py 2003 2024    # only these editions
"""
import sys

import pandas as pd
import pymupdf

from common import (BLOCKS_DIR, CHAPTER_COLUMNS, CHAPTERS_CSV, RAW, TOC_DIR,
                    ensure_dirs, load_chapters, load_editions)


def table_boxes(page):
    """Bounding boxes of tables PyMuPDF detects on the page."""
    try:
        return [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
    except Exception:
        return []


def overlap_share(rect, boxes):
    """Largest share of rect's area covered by any single box."""
    area = rect.get_area()
    if area == 0:
        return 0.0
    return max(((rect & b).get_area() / area for b in boxes), default=0.0)


def block_record(block, page, tables):
    lines, sizes, chars, bold = [], [], 0, 0
    for line in block["lines"]:
        lines.append("".join(s["text"] for s in line["spans"]))
        for s in line["spans"]:
            n = len(s["text"].strip())
            if n == 0:
                continue
            sizes.append((round(s["size"], 1), n))
            chars += n
            if s["flags"] & 16 or "bold" in s["font"].lower():
                bold += n
    if chars == 0:
        return None
    rect = pymupdf.Rect(block["bbox"])
    return {
        "page": page.number + 1,             # 1-indexed, as shown in PDF viewers
        "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
        "page_w": page.rect.width, "page_h": page.rect.height,
        "font_size": sum(s * n for s, n in sizes) / chars,
        "bold_ratio": bold / chars,
        "n_lines": len(lines),
        "text": "\n".join(lines),            # keep line breaks for de-hyphenation
        "in_table": overlap_share(rect, tables) > 0.5,
    }


def extract(edition, path):
    doc = pymupdf.open(path)
    rows, low_text_pages = [], []
    for page in doc:
        tables = table_boxes(page)
        data = page.get_text("dict")
        page_rows = [r for b in data["blocks"] if b.get("type") == 0
                     for r in [block_record(b, page, tables)] if r]
        if sum(len(r["text"]) for r in page_rows) < 50:
            low_text_pages.append(page.number + 1)
        for i, r in enumerate(page_rows):
            r["block_no"] = i
        rows.extend(page_rows)

    df = pd.DataFrame(rows)
    df.insert(0, "edition", edition)
    df.to_parquet(BLOCKS_DIR / f"{edition}.parquet", index=False)

    toc = pd.DataFrame(doc.get_toc(simple=True), columns=["level", "title", "page"])
    toc.to_csv(TOC_DIR / f"{edition}.csv", index=False)

    share_low = len(low_text_pages) / max(len(doc), 1)
    print(f"  {edition}: {len(doc)} pages, {len(df)} blocks, "
          f"{len(toc)} TOC entries, {len(low_text_pages)} near-empty pages")
    if share_low > 0.3:
        print(f"  WARNING: {edition} may be a scanned PDF with no text layer. "
              "Run OCR first (e.g. `ocrmypdf in.pdf out.pdf`).")
    return toc


def add_chapter_template(edition, toc, chapters):
    """Add level-1 TOC entries to config/chapters.csv if the edition is missing."""
    if (chapters["edition"] == edition).any():
        return chapters
    top = toc[toc["level"] == 1]
    if top.empty:
        print(f"  NOTE: {edition} has no built-in TOC. Add its chapters to "
              "config/chapters.csv by hand (title + start page).")
        return chapters
    new = pd.DataFrame({
        "edition": edition,
        "chapter_no": range(1, len(top) + 1),
        "chapter_title": top["title"].str.strip().values,
        "start_page": top["page"].values,
        "chapter_type": "",
        "author": "",
    })
    return pd.concat([chapters, new], ignore_index=True)


def main():
    ensure_dirs()
    editions = load_editions(sys.argv[1:])
    chapters = load_chapters()[CHAPTER_COLUMNS]
    for _, row in editions.iterrows():
        toc = extract(int(row["edition"]), RAW / row["filename"])
        chapters = add_chapter_template(int(row["edition"]), toc, chapters)
    chapters.sort_values(["edition", "start_page"]).to_csv(CHAPTERS_CSV, index=False)
    print(f"\nDone. Now open {CHAPTERS_CSV.name} and check every edition's rows "
          "(see README, step 2).")


if __name__ == "__main__":
    main()
