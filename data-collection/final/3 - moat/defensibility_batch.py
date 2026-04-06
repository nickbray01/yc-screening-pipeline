"""
Batch Defensibility Enrichment Runner
======================================
Reads your existing YC company CSV and runs the defensibility enricher
on every company, appending defensibility signals as new columns.

The full batch is passed to each enrichment call so the LLM can identify
intra-batch competitors (the batch peer analysis feature).

Usage:
    python defensibility_batch.py companies.csv --output defensibility.csv
    python defensibility_batch.py companies.csv --output defensibility.csv --concurrency 3
    python defensibility_batch.py companies.csv --output defensibility.csv --limit 10  # test on first 10

    # Resume from where you left off (skips already-enriched rows)
    python defensibility_batch.py companies.csv --output defensibility.csv --resume

    # Run on already product-enriched CSV
    python defensibility_batch.py product_enriched.csv --output defensibility.csv

Requires:
    pip install langchain langchain-anthropic httpx beautifulsoup4 pydantic

    export ANTHROPIC_API_KEY=sk-ant-...
    export GITHUB_TOKEN=ghp-...  # optional, increases GitHub rate limits
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path

from defensibility_enricher import (
    DefensibilityEnricher,
    DefensibilitySignalCard,
    format_defensibility_card,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("defensibility_batch")


# ─── CSV I/O ──────────────────────────────────────────────────────────────────

def read_companies(filepath: str) -> list[dict]:
    """Read the YC company CSV into a list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_existing_results(filepath: str) -> dict[str, dict]:
    """Load already-enriched results for resume mode."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        results = {}
        for row in csv.DictReader(f):
            key = row.get("slug", row.get("company_name", row.get("name", "")))
            if row.get("market_crowding"):  # defensibility sentinel column
                results[key] = row
        return results


def write_results(
    filepath: str,
    original_rows: list[dict],
    defensibility_results: dict[str, DefensibilitySignalCard | dict],
):
    """Write merged CSV: original columns + defensibility signal columns."""
    if not original_rows:
        return

    original_cols = list(original_rows[0].keys())
    def_cols = DefensibilitySignalCard.csv_columns()
    all_cols = original_cols + def_cols

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()

        for row in original_rows:
            merged = dict(row)
            key = (
                row.get("slug")
                or row.get("company_name")
                or row.get("name")
                or row.get("url", "").split("/")[-1]
                or ""
            )

            if key in defensibility_results:
                card = defensibility_results[key]
                if isinstance(card, DefensibilitySignalCard):
                    merged.update(card.to_row())
                elif isinstance(card, dict) and "error" not in card:
                    merged.update(card)
                else:
                    for col in def_cols:
                        merged[col] = ""
                    if isinstance(card, dict):
                        merged["competitive_context_brief"] = card.get("error", "enrichment failed")
            else:
                for col in def_cols:
                    merged[col] = ""

            writer.writerow(merged)

    logger.info(f"Wrote {len(original_rows)} rows to {filepath}")


# ─── Batch Runner ─────────────────────────────────────────────────────────────

def _extract_key(co: dict) -> str:
    return (
        co.get("slug")
        or co.get("company_name")
        or co.get("name")
        or co.get("url", "").split("/")[-1]
        or ""
    )


async def run_batch(
    companies: list[dict],
    concurrency: int = 3,
    resume_results: dict | None = None,
    github_token: str = "",
) -> dict[str, DefensibilitySignalCard | dict]:
    """
    Run defensibility enrichment on a list of companies.
    Passes the full company list to each call for intra-batch competitor detection.
    Returns dict mapping slug/name → DefensibilitySignalCard.
    """
    enricher = DefensibilityEnricher(github_token=github_token)
    results: dict[str, DefensibilitySignalCard | dict] = {}
    semaphore = asyncio.Semaphore(concurrency)

    if resume_results:
        logger.info(f"Resuming: {len(resume_results)} companies already enriched")
        results.update(resume_results)

    todo = [co for co in companies if _extract_key(co) not in results]

    if not todo:
        logger.info("All companies already enriched. Nothing to do.")
        return results

    logger.info(f"Enriching {len(todo)} companies (concurrency={concurrency})")
    start_time = time.time()
    completed = 0
    errors = 0

    async def enrich_one(co: dict):
        nonlocal completed, errors

        key = _extract_key(co)
        name = (
            co.get("company_name")
            or co.get("name")
            or co.get("company")
            or key
        )
        website = (
            co.get("website")
            or co.get("website_url")
            or co.get("site")
            or ""
        )
        one_liner = (
            co.get("one_liner")
            or co.get("oneliner")
            or co.get("description")
            or co.get("tagline")
            or ""
        )

        if not name:
            logger.warning(f"Skipping row with no company name: {co}")
            return

        async with semaphore:
            try:
                card = await enricher.enrich(
                    company_name=name,
                    one_liner=one_liner,
                    website=website,
                    batch_companies=companies,  # full batch for peer analysis
                )
                results[key] = card
                completed += 1

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - completed) / rate if rate > 0 else 0

                logger.info(
                    f"[{completed}/{len(todo)}] {name}: "
                    f"crowding={card.market_crowding} "
                    f"moat={card.moat_type} "
                    f"batch_competitors={card.competitor_count_batch} "
                    f"incumbents={card.has_funded_incumbents} "
                    f"({rate:.1f}/min, ~{remaining:.0f}s remaining)"
                )

            except Exception as e:
                errors += 1
                results[key] = {"error": str(e)}
                logger.error(f"[{completed + errors}/{len(todo)}] {name}: FAILED — {e}")

    await asyncio.gather(*[enrich_one(co) for co in todo])

    elapsed = time.time() - start_time
    logger.info(
        f"Batch complete: {completed} enriched, {errors} errors "
        f"in {elapsed:.1f}s ({completed / (elapsed / 60):.1f}/min)"
    )
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Run defensibility enrichment on a YC company CSV"
    )
    parser.add_argument("input_csv", help="Path to your YC company CSV")
    parser.add_argument("--output", "-o", default="defensibility_enriched.csv",
                        help="Output CSV path (default: defensibility_enriched.csv)")
    parser.add_argument("--concurrency", "-c", type=int, default=3,
                        help="Max concurrent enrichments (default: 3)")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Only process first N companies (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skip already-enriched rows")
    parser.add_argument("--json", default=None,
                        help="Also output results as JSON to this path")

    args = parser.parse_args()

    companies = read_companies(args.input_csv)
    logger.info(f"Loaded {len(companies)} companies from {args.input_csv}")

    if companies:
        cols = list(companies[0].keys())
        logger.info(f"CSV columns detected: {cols}")
        sample = companies[0]
        logger.info(
            f"First row sample: company_name='{sample.get('company_name', sample.get('name', '???'))}' "
            f"one_liner='{sample.get('one_liner', '???')[:60]}' "
            f"website='{sample.get('website', '???')}'"
        )

    if args.limit:
        companies = companies[:args.limit]
        logger.info(f"Limited to first {len(companies)} companies")

    resume_results = None
    if args.resume:
        resume_results = load_existing_results(args.output)

    github_token = os.environ.get("GITHUB_TOKEN", "")

    def_results = await run_batch(
        companies=companies,
        concurrency=args.concurrency,
        resume_results=resume_results,
        github_token=github_token,
    )

    # Re-read full input for output (limit only affects what we enrich, not what we write)
    all_companies = read_companies(args.input_csv)
    if args.limit:
        all_companies = all_companies[:args.limit]
    write_results(args.output, all_companies, def_results)

    if args.json:
        json_out = {}
        for key, val in def_results.items():
            json_out[key] = val.to_row() if isinstance(val, DefensibilitySignalCard) else val
        with open(args.json, "w") as f:
            json.dump(json_out, f, indent=2, default=str)
        logger.info(f"JSON results saved to {args.json}")

    # Summary
    cards = [v for v in def_results.values() if isinstance(v, DefensibilitySignalCard)]
    if cards:
        crowding_counts: dict[str, int] = {}
        for c in cards:
            crowding_counts[c.market_crowding] = crowding_counts.get(c.market_crowding, 0) + 1

        moat_counts: dict[str, int] = {}
        for c in cards:
            moat_counts[c.moat_type] = moat_counts.get(c.moat_type, 0) + 1

        has_incumbents = sum(1 for c in cards if c.has_funded_incumbents)
        has_oss_risk = sum(1 for c in cards if c.has_oss_alternative)
        has_patents = sum(1 for c in cards if c.has_patent_signal)
        avg_batch_competitors = sum(c.competitor_count_batch for c in cards) / len(cards)

        print(f"\n{'═' * 60}")
        print(f"  DEFENSIBILITY ENRICHMENT SUMMARY — {len(cards)} companies")
        print(f"{'═' * 60}")
        print(f"  Avg batch competitors:  {avg_batch_competitors:.1f}")
        print(f"  Has funded incumbents:  {has_incumbents:>4} ({100*has_incumbents/len(cards):.0f}%)")
        print(f"  OSS alternative risk:   {has_oss_risk:>4} ({100*has_oss_risk/len(cards):.0f}%)")
        print(f"  Patent signals found:   {has_patents:>4} ({100*has_patents/len(cards):.0f}%)")
        print(f"\n  Market crowding breakdown:")
        for label in ["GREENFIELD", "EMERGING", "COMPETITIVE", "CROWDED"]:
            n = crowding_counts.get(label, 0)
            print(f"    {label:<14} {n:>4} ({100*n/len(cards):.0f}%)")
        print(f"\n  Moat type breakdown:")
        for label, n in sorted(moat_counts.items(), key=lambda x: -x[1]):
            print(f"    {label:<20} {n:>4} ({100*n/len(cards):.0f}%)")
        print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
