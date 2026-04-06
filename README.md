# Greylock YC Screening Pipeline

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

Evaluates founder backgrounds and team composition. No dedicated CLI batch runner — call the module directly.

**Signals produced:**
- Domain relevance score (1–10) and team completeness score (1–10)
- Booleans: worked in target industry, held target function, prior exit, repeat YC, from top company, technical/commercial co-founder, domain expert, solo founder
- Overall signal: `STRONG` | `MODERATE` | `WEAK` | `PASS`

**Single company:**
```python
import asyncio
from yc_screener_v2 import YCScreeningPipeline

async def main():
    pipeline = YCScreeningPipeline()
    card = await pipeline.screen_fast(
        "https://www.ycombinator.com/companies/trycardinal-ai"
    )
    print(card.to_row())  # flat dict, ready for CSV

asyncio.run(main())
```

**Batch:**
```python
async def main():
    pipeline = YCScreeningPipeline(max_concurrent=3)
    urls = ["https://www.ycombinator.com/companies/co1", ...]

    results = await pipeline.screen_batch_fast(urls)

    from yc_screener_v2 import export_csv
    export_csv(results, "team_signals.csv")

asyncio.run(main())
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

**Batch (CLI):**
```bash
cd "data-collection/final/2 - product"

python run_product_batch.py companies.csv --output product_enriched.csv
python run_product_batch.py companies.csv --output product_enriched.csv --limit 10       # test first 10
python run_product_batch.py companies.csv --output product_enriched.csv --concurrency 5  # parallel workers
python run_product_batch.py companies.csv --output product_enriched.csv --filter-active  # active companies only
python run_product_batch.py companies.csv --output product_enriched.csv --resume         # skip already-done rows
```

**Single company:**
```python
import asyncio
from product_enricher import ProductEnricher

async def main():
    enricher = ProductEnricher()
    signals = await enricher.enrich(
        company_name="Cardinal",
        website_url="https://trycardinal.ai",
        yc_url="https://www.ycombinator.com/companies/trycardinal-ai",
        one_liner="AI Platform for Precision Outbound",
    )
    print(signals.to_row())

asyncio.run(main())
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

**Batch (CLI):**
```bash
cd "data-collection/final/3 - moat"

python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv
python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv --limit 10
python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv --concurrency 3
python defensibility_batch.py product_enriched.csv --output defensibility_enriched.csv --resume
```

**Single company:**
```python
import asyncio
from defensibility_enricher import DefensibilityEnricher

async def main():
    enricher = DefensibilityEnricher()
    card = await enricher.enrich(
        company_name="Cardinal",
        one_liner="AI Platform for Precision Outbound",
        website="https://trycardinal.ai",
        batch_companies=[  # pass full batch for peer analysis
            {"name": "OtherCo", "one_liner": "AI sales outreach"},
        ],
    )
    print(card.market_crowding, card.moat_type)

asyncio.run(main())
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