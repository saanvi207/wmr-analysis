"""Shared paths and small helpers used by every pipeline step."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "config"
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
BLOCKS_DIR = INTERIM / "blocks"      # step 1 output: raw text blocks per edition
CLEAN_DIR = INTERIM / "clean"        # step 3 output: blocks with keep/drop labels
TOC_DIR = INTERIM / "toc"            # built-in PDF tables of contents
PROCESSED = ROOT / "data" / "processed"
QA_DIR = ROOT / "data" / "qa"
TABLES = ROOT / "outputs" / "tables"

EDITIONS_CSV = CONFIG / "editions.csv"
CHAPTERS_CSV = CONFIG / "chapters.csv"
CHAPTER_COLUMNS = ["edition", "chapter_no", "chapter_title", "start_page",
                   "chapter_type", "author"]


def ensure_dirs():
    for d in [BLOCKS_DIR, CLEAN_DIR, TOC_DIR, PROCESSED, QA_DIR, TABLES]:
        d.mkdir(parents=True, exist_ok=True)


def load_editions(only=None):
    """Return editions whose PDF is present in data/raw/.

    only: optional list of edition years to restrict to.
    """
    df = pd.read_csv(EDITIONS_CSV)
    if only:
        df = df[df["edition"].isin([int(x) for x in only])]
    present = df["filename"].apply(lambda f: (RAW / f).exists())
    for _, row in df[~present].iterrows():
        print(f"  [skip] {row['filename']} not found in data/raw/")
    return df[present].reset_index(drop=True)


def load_chapters():
    if not CHAPTERS_CSV.exists():
        return pd.DataFrame(columns=CHAPTER_COLUMNS)
    df = pd.read_csv(CHAPTERS_CSV)
    df["chapter_type"] = df["chapter_type"].fillna("unspecified")
    df["author"] = df["author"].fillna("unknown")
    return df
