# YC Screening Pipeline

A three-stage LangChain + Claude pipeline that evaluates YC companies across **Team**, **Product**, and **Moat/Defensibility** signals, plus a CLI tool for generating VC outreach prompts.

## Live Demo

**[nickb-yc-screener.netlify.app](https://nickb-yc-screener.netlify.app)** — Password: `nickbray`

A vibe-coded front end (`visualization/signal_lab.html`) hosted as a static page on Netlify. Hard-coded against the final enriched dataset — no backend.

---

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
export GITHUB_TOKEN=ghp-...   # optional — raises GitHub rate limits for moat analysis
```

---

## Pipeline Overview

Each pipeline reads a CSV of YC companies and appends signal columns. Run them in order — each stage's output feeds the next.

```
Pipeline 1 (Team)  →  Pipeline 2 (Product)  →  Pipeline 3 (Moat)  →  outreach_helper.py
```

---

## Pipeline 1 — Team Signals

**Script:** `data-collection/final/1 - team/yc_screener_v2.py`

Evaluates founder backgrounds and team composition.

**Signals produced:**
- Domain relevance score (1–10) and team completeness score (1–10)
- Booleans: worked in target industry, held target function, prior exit, repeat YC, from top company, technical/commercial co-founder, domain expert, solo founder
- Overall signal: `STRONG` | `MODERATE` | `WEAK` | `PASS`

```bash
cd "data-collection/final/1 - team"

# Single company
python yc_screener_v2.py https://www.ycombinator.com/companies/trycardinal-ai

# Batch from a file of URLs (one per line)
cat urls.txt | python yc_screener_v2.py --csv=team_signals.csv

# Batch (space-separated URLs)
python yc_screener_v2.py URL1 URL2 URL3 --csv=team_signals.csv
```

---

## Pipeline 2 — Product Signals

**Scripts:**
- Module: `data-collection/final/2 - product/product_enricher.py`
- Batch runner: `data-collection/final/2 - product/run_product_batch.py`

Searches company websites, ProductHunt, app stores, GitHub, HN, and news to assess product maturity and founder user-research depth.

**Signals produced:**
- Product maturity score (1–10) and user understanding score (1–10)
- Booleans: live product, paying customers, user evidence, public iteration, specific user problem identified, beachhead segment defined
- Text: strongest evidence, biggest product gap

```bash
cd "data-collection/final/2 - product"

python run_product_batch.py companies.csv --output product_enriched.csv
python run_product_batch.py companies.csv --output product_enriched.csv --limit 10       # test first 10
python run_product_batch.py companies.csv --output product_enriched.csv --concurrency 5  # parallel workers
python run_product_batch.py companies.csv --output product_enriched.csv --filter-active  # active companies only
python run_product_batch.py companies.csv --output product_enriched.csv --resume         # skip already-done rows
```

---

## Pipeline 3 — Moat / Defensibility Signals

**Scripts:**
- Module: `data-collection/final/3 - moat/defensibility_enricher.py`
- Batch runner: `data-collection/final/3 - moat/defensibility_batch.py`

Analyzes competitive density, incumbent funding, OSS alternatives, patents, and intra-batch competition. The full batch is passed to each enrichment call so the model can identify companies competing with each other within the same cohort.

**Signals produced:**
- Market crowding: `LOW` | `MEDIUM` | `HIGH`
- Moat type and moat evidence (text)
- Booleans: funded incumbents, patent signal, OSS alternative
- Batch competitor count and names
- Competitive context brief and key question for meeting

```bash
cd "data-collection/final/3 - moat"

python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv
python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv --limit 10
python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv --concurrency 3
python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv --resume
```

---

## Outreach Helper

**Script:** `analysis/outreach_helper.py`

Looks up any company in the final enriched CSV, prints a formatted signal card, then generates and copies a VC outreach metaprompt to your clipboard. Paste the metaprompt into Claude or ChatGPT to draft a personalized first-touch email.

**Requires** the final enriched CSV at `visualization/defensibility_enriched.csv` (output of Pipeline 3).

```bash
cd analysis

python outreach_helper.py --company "Cardinal"
python outreach_helper.py --company mendral    # case-insensitive, partial match ok
```

The script will:
1. Print an animated signal card (team + product + moat signals)
2. Print a ready-to-paste metaprompt with all context pre-filled
3. Auto-copy the metaprompt to your clipboard (`pbcopy` on Mac)

---

## Batch CLI Flags Reference

| Flag | Pipelines | Description |
|------|-----------|-------------|
| `--output FILE` | 2, 3 | Output CSV path |
| `--concurrency N` | 2, 3 | Parallel workers (default: 3) |
| `--limit N` | 2, 3 | Process only first N rows (for testing) |
| `--resume` | 2, 3 | Skip rows already present in output CSV |
| `--filter-active` | 2 | Only process companies with `status=Active` |


---
© 2026 Nicholas Bray. All rights reserved.
