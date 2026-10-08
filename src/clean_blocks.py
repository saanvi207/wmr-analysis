"""Step 3: decide which blocks are real prose, and put them in reading order.

Nothing is deleted. Every block gets:
  block_type   'body', 'heading' or 'dropped'
  drop_reason  why a block was dropped (table, header_footer, page_number,
               caption, footnote, empty); blank for kept blocks
  order        reading-order position within its page (handles two columns)

Keeping the dropped blocks with a reason lets you audit the cleaning later.

Usage:
  python src/clean_blocks.py [editions...]
"""
import re
import sys

import numpy as np
import pandas as pd

from common import BLOCKS_DIR, CLEAN_DIR, TABLES, ensure_dirs, load_editions

# ---- thresholds you may need to tune after the QA check ---------------------
MARGIN = 0.07           # top/bottom share of the page treated as header/footer zone
REPEAT_PAGES = 3        # same text on this many pages = running header/footer
FOOTNOTE_RATIO = 0.85   # font smaller than this x body size = footnote/source note
HEADING_RATIO = 1.15    # font larger than this x body size = heading
DIGIT_RATIO = 0.40      # blocks where >40% of letters/digits are digits = table
# -----------------------------------------------------------------------------

PAGE_NUM = re.compile(r"^\s*(page\s*)?(\d{1,4}|[ivxlc]{1,6})\s*$", re.I)
CAPTION = re.compile(r"^\s*(figure|fig\.|table|map|box|chart|graph)\s*[\dA-Z.:]", re.I)
SOURCE_NOTE = re.compile(r"^\s*(sources?|notes?)\s*:", re.I)


def norm(text):
    """Normalise text for spotting repeated headers (ignore digits/case/space)."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", text.lower())).strip()


def body_size(df):
    """Most common font size, weighted by characters: the main body text size."""
    weights = df["text"].str.len().groupby(df["font_size"].round(0)).sum()
    return float(weights.idxmax())


def digit_ratio(text):
    alnum = [c for c in text if c.isalnum()]
    return sum(c.isdigit() for c in alnum) / len(alnum) if alnum else 0.0


def classify(df):
    base = body_size(df)
    df["norm"] = df["text"].map(norm)
    pages_per_text = df.groupby("norm")["page"].nunique()
    df["repeats"] = df["norm"].map(pages_per_text)

    in_margin = (df["y0"] < MARGIN * df["page_h"]) | (df["y1"] > (1 - MARGIN) * df["page_h"])
    short = df["text"].str.len() < 150

    reasons = np.select(
        [
            df["text"].str.strip() == "",
            df["text"].str.match(PAGE_NUM),
            in_margin & (short | (df["repeats"] >= REPEAT_PAGES)),
            (df["repeats"] >= REPEAT_PAGES) & short,
            df["in_table"] | (df["text"].map(digit_ratio) > DIGIT_RATIO),
            (df["text"].str.match(CAPTION) & (df["text"].str.len() < 300))
            | df["text"].str.match(SOURCE_NOTE),
            df["font_size"] < FOOTNOTE_RATIO * base,
        ],
        ["empty", "page_number", "header_footer", "header_footer",
         "table", "caption", "footnote"],
        default="",
    )
    df["drop_reason"] = reasons

    is_heading = (df["font_size"] >= HEADING_RATIO * base) | (
        (df["bold_ratio"] > 0.8) & (df["text"].str.len() < 150) & (df["n_lines"] <= 3)
    )
    df["block_type"] = np.where(df["drop_reason"] != "", "dropped",
                                np.where(is_heading, "heading", "body"))
    df["body_font_size"] = base
    return df.drop(columns=["norm", "repeats"])


def reading_order(page):
    """Order blocks on one page, handling two-column layouts.

    Full-width blocks (headings, wide paragraphs) split the page into bands.
    Inside each band the left column is read top to bottom, then the right.
    """
    w = page["page_w"].iloc[0]
    mid = w / 2
    width = page["x1"] - page["x0"]
    full = (width > 0.6 * w) | ((page["x0"] < mid - 0.05 * w) & (page["x1"] > mid + 0.05 * w))
    col = np.where((page["x0"] + page["x1"]) / 2 < mid, 0, 1)

    full_tops = np.sort(page.loc[full, "y0"].values)
    keys = []
    for is_full, y0, c in zip(full, page["y0"], col):
        if is_full:
            k = int(np.searchsorted(full_tops, y0, side="left"))
            keys.append((k, 1, 0, y0))
        else:
            band = int(np.searchsorted(full_tops, y0, side="left"))
            keys.append((band, 0, int(c), y0))
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = np.empty(len(keys), dtype=int)
    ranks[order] = np.arange(len(keys))
    return pd.Series(ranks, index=page.index)


def main():
    ensure_dirs()
    summary = []
    for _, row in load_editions(sys.argv[1:]).iterrows():
        ed = int(row["edition"])
        src = BLOCKS_DIR / f"{ed}.parquet"
        if not src.exists():
            print(f"  [skip] {ed}: run extract_blocks.py first")
            continue
        df = classify(pd.read_parquet(src))
        df["order"] = df.groupby("page", group_keys=False).apply(reading_order)
        df = df.sort_values(["page", "order"]).reset_index(drop=True)
        df.to_parquet(CLEAN_DIR / f"{ed}.parquet", index=False)

        counts = df["block_type"].value_counts().to_dict()
        counts.update({f"drop_{k}": v for k, v in
                       df.loc[df["drop_reason"] != "", "drop_reason"].value_counts().items()})
        summary.append({"edition": ed, "body_font_size": df["body_font_size"].iloc[0], **counts})
        print(f"  {ed}: {counts.get('body', 0)} body, {counts.get('heading', 0)} headings, "
              f"{counts.get('dropped', 0)} dropped")

    if summary:
        pd.DataFrame(summary).fillna(0).to_csv(TABLES / "cleaning_summary.csv", index=False)
        print(f"\nDrop counts per reason saved to outputs/tables/cleaning_summary.csv")


if __name__ == "__main__":
    main()
