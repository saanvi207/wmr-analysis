# WMR Gender Project: Phase 1 (Corpus Building)

Turns the 13 IOM World Migration Reports (2000 to 2026) into one clean,
sentence-level dataset: `data/processed/sentences.parquet`.

## Folder structure

```
wmr-gender/
├── config/
│   ├── editions.csv        # edition year -> PDF filename
│   └── chapters.csv        # created by step 1, completed by you (step 2)
├── data/
│   ├── raw/                # the 13 original PDFs (never edit these)
│   ├── interim/            # pipeline working files (safe to delete and rebuild)
│   │   ├── blocks/         # step 1: raw text blocks
│   │   ├── toc/            # step 1: each PDF's built-in table of contents
│   │   └── clean/          # step 3: blocks labelled body / heading / dropped
│   ├── processed/          # step 4: paragraphs.parquet, sentences.parquet
│   ├── qa/                 # step 5: sampled pages + your quality-check log
│   └── coding/             # later: your hand-coded spreadsheets
├── notebooks/              # exploration only; real code lives in src/
├── prompts/                # every AI prompt you use, saved as text files
├── outputs/
│   ├── figures/
│   └── tables/             # corpus_summary.csv, cleaning_summary.csv
├── src/
│   ├── common.py           # shared paths and helpers
│   ├── extract_blocks.py   # step 1
│   ├── clean_blocks.py     # step 3
│   ├── build_corpus.py     # step 4
│   └── qa_sample.py        # step 5
├── requirements.txt
└── .gitignore
```

## Setup (once)

```bash
cd wmr-gender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
git init && git add . && git commit -m "Project setup"
```

Download each report from worldmigrationreport.iom.int and save it in
`data/raw/` using the names in `config/editions.csv` (`wmr_2000.pdf`,
`wmr_2003.pdf`, ... `wmr_2026.pdf`).

Tip: test everything on 2 or 3 editions first (for example 2003 2015 2024),
then run the rest.

## Run order

Run all commands from the `wmr-gender/` folder. Every script accepts
edition years to process only those, e.g. `python src/clean_blocks.py 2024`.

### Step 1: Extract text blocks
```bash
python src/extract_blocks.py
```
Pulls every text block with its position, font size and a table flag.
It warns you if a PDF looks scanned (no text layer). If so, OCR it first
with `ocrmypdf input.pdf output.pdf` and replace the file.

### Step 2: Complete `config/chapters.csv` (manual, important)
Step 1 fills this from each PDF's table of contents. Open it in Excel or
Numbers and, for every edition:

1. **Check the chapter rows.** Delete rows that aren't chapters (e.g. "Foreword",
   "Part I") and check `start_page` against the PDF. `start_page` is the
   PDF page number shown in your viewer, not the number printed on the page.
   If a chapter list looks wrong, compare with `data/interim/toc/<edition>.csv`,
   which also shows sub-levels.
2. **Add missing editions by hand.** Older PDFs may have no built-in contents;
   type in the chapter titles and start pages yourself.
3. **Fill `chapter_type`** using one fixed set of labels, for example:
   `overview` (global/regional data chapters), `thematic`, `regional`,
   `introduction`, `appendix`.
4. **Fill `author`** from the chapter's first page (e.g. `IOM` or the names of
   commissioned authors).

Pages before the first chapter are tagged as chapter 0, "Front matter".

### Step 3: Clean and order blocks
```bash
python src/clean_blocks.py
```
Labels each block as `body`, `heading` or `dropped` (with a reason: table,
header_footer, page_number, caption, footnote) and fixes two-column reading
order. Nothing is deleted, so you can audit what was removed.
Drop counts per edition: `outputs/tables/cleaning_summary.csv`.

### Step 4: Build paragraphs and sentences
```bash
python src/build_corpus.py
```
Joins lines, repairs words hyphenated across lines (keeping real compounds like
"gender-based"), joins paragraphs that flow across columns or pages, attaches
chapter tags, marks reference lists, and splits into sentences.

Output: `data/processed/sentences.parquet`. For analysis, usually filter to prose:
```python
import pandas as pd
df = pd.read_parquet("data/processed/sentences.parquet")
prose = df[~df.is_heading & ~df.in_references]
```
Word counts per edition: `outputs/tables/corpus_summary.csv`.

### Step 5: Quality check (for your methodology section)
```bash
python src/qa_sample.py            # samples 20 random pages
```
For each page, open the `.png` (the real page) next to the `.txt` (what the
pipeline kept and dropped). In `data/qa/qa_log.csv`, put `1` in each column
where you see that problem, `0` otherwise. Then:
```bash
python src/qa_sample.py --report   # error rate to report in your write-up
```

## If the QA check finds problems

Thresholds are at the top of `src/clean_blocks.py`:

| Problem in QA | What to adjust |
|---|---|
| Headers/footers leaking in | Increase `MARGIN` or lower `REPEAT_PAGES` |
| Real text dropped as footnote | Lower `FOOTNOTE_RATIO` (e.g. 0.8) |
| Table numbers leaking in | Lower `DIGIT_RATIO` |
| Normal text marked as heading | Raise `HEADING_RATIO` |
| Wrong chapter | Fix `start_page` in `config/chapters.csv` |

After changing anything, rerun steps 3 to 5 and record the change in a log.
Report the final error rate and the changes you made in your methodology.

## Notes
- PyMuPDF may print a tip about `pymupdf_layout`; it's harmless.
- `data/raw`, `data/interim` and `data/processed` are git-ignored because they
  are large and can be rebuilt from the scripts.
