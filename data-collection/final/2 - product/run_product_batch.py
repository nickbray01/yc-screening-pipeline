"""
Batch Product Enrichment Runner
=================================
Reads your existing YC company CSV and runs the product enricher
on every company, appending product signals as new columns.

Usage:
    python run_product_batch.py companies.csv --output enriched.csv
    python run_product_batch.py companies.csv --output enriched.csv --concurrency 3
    python run_product_batch.py companies.csv --output enriched.csv --limit 10  # test on first 10

    # Filter to only active companies
    python run_product_batch.py companies.csv --output enriched.csv --filter-active

    # Resume from where you left off (skips already-enriched rows)
    python run_product_batch.py companies.csv --output enriched.csv --resume

Requires:
    pip install langchain langchain-anthropic httpx beautifulsoup4 pydantic

    export ANTHROPIC_API_KEY=sk-ant-...
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

from product_enricher import (
    ProductEnricher,
    ProductSignalCard,
    format_product_card,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("batch_runner")


# ─── CSV I/O ──────────────────────────────────────────────────────────────────

def read_companies(filepath: str) -> list[dict]:
    """Read the YC company CSV into a list of dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_existing_results(filepath: str) -> dict[str, dict]:
    """Load already-enriched results for resume mode."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = {}
        for row in reader:
            slug = row.get("slug", row.get("name", ""))
            # Check if product signals exist in this row
            if row.get("has_live_product"):
                results[slug] = row
        return results


def write_results(
    filepath: str,
    original_rows: list[dict],
    product_results: dict[str, ProductSignalCard | dict],
):
    """Write merged CSV: original columns + product signal columns."""
    if not original_rows:
        return

    # Build column list: original + product signals
    original_cols = list(original_rows[0].keys())
    product_cols = ProductSignalCard.csv_columns()
    all_cols = original_cols + product_cols

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()

        for row in original_rows:
            merged = dict(row)
            key = (
                row.get("slug")
                or row.get("name")
                or row.get("company_name")
                or row.get("url", "").split("/")[-1]
                or ""
            )

            if key in product_results:
                card = product_results[key]
                if isinstance(card, ProductSignalCard):
                    merged.update(card.to_row())
                elif isinstance(card, dict) and "error" not in card:
                    merged.update(card)
                else:
                    # Error case — fill with empty
                    for col in product_cols:
                        merged[col] = ""
                    merged["biggest_product_gap"] = card.get("error", "enrichment failed") if isinstance(card, dict) else ""
            else:
                for col in product_cols:
                    merged[col] = ""

            writer.writerow(merged)

    logger.info(f"Wrote {len(original_rows)} rows to {filepath}")


# ─── Batch Runner ─────────────────────────────────────────────────────────────

async def run_batch(
    companies: list[dict],
    concurrency: int = 3,
    resume_results: dict | None = None,
) -> dict[str, ProductSignalCard | dict]:
    """
    Run product enrichment on a list of companies.
    Returns dict mapping slug/name → ProductSignalCard.
    """
    enricher = ProductEnricher()
    results: dict[str, ProductSignalCard | dict] = {}
    semaphore = asyncio.Semaphore(concurrency)

    # Pre-populate with resume results
    if resume_results:
        logger.info(f"Resuming: {len(resume_results)} companies already enriched")
        results.update(resume_results)

    # Filter out already-done companies
    todo = []
    for co in companies:
        key = co.get("slug", co.get("name", ""))
        if key not in results:
            todo.append(co)

    if not todo:
        logger.info("All companies already enriched. Nothing to do.")
        return results

    logger.info(f"Enriching {len(todo)} companies (concurrency={concurrency})")
    start_time = time.time()

    completed = 0
    errors = 0

    async def enrich_one(co: dict):
        nonlocal completed, errors

        # Auto-detect columns — handle different CSV schemas
        key = (
            co.get("slug")
            or co.get("name")
            or co.get("company_name")
            or co.get("url", "").split("/")[-1]  # extract slug from YC URL
            or ""
        )
        name = (
            co.get("name")
            or co.get("company_name")
            or co.get("company")
            or key
        )
        website = (
            co.get("website")
            or co.get("website_url")
            or co.get("site")
            or ""
        )
        yc_url = (
            co.get("yc_url")
            or co.get("url")
            or co.get("yc_link")
            or ""
        )
        one_liner = (
            co.get("one_liner")
            or co.get("oneliner")
            or co.get("description")
            or co.get("tagline")
            or ""
        )

        # Skip rows where we have no company info at all
        if not name or name == key == "":
            logger.warning(f"Skipping row with no company name: {co}")
            return

        async with semaphore:
            try:
                card = await enricher.enrich(
                    company_name=name,
                    website_url=website,
                    yc_url=yc_url,
                    one_liner=one_liner,
                )
                results[key] = card
                completed += 1

                # Progress
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - completed) / rate if rate > 0 else 0

                logger.info(
                    f"[{completed}/{len(todo)}] {name}: "
                    f"maturity={card.product_maturity_score} "
                    f"understanding={card.user_understanding_score} "
                    f"live={card.has_live_product} "
                    f"({rate:.1f}/min, ~{remaining:.0f}s remaining)"
                )

            except Exception as e:
                errors += 1
                results[key] = {"error": str(e)}
                logger.error(f"[{completed + errors}/{len(todo)}] {name}: FAILED — {e}")

    # Run all
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
        description="Run product enrichment on a YC company CSV"
    )
    parser.add_argument("input_csv", help="Path to your YC company CSV")
    parser.add_argument("--output", "-o", default="enriched_products.csv",
                        help="Output CSV path (default: enriched_products.csv)")
    parser.add_argument("--concurrency", "-c", type=int, default=3,
                        help="Max concurrent enrichments (default: 3)")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Only process first N companies (for testing)")
    parser.add_argument("--filter-active", action="store_true",
                        help="Only process companies with status=Active")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skip already-enriched rows")
    parser.add_argument("--json", default=None,
                        help="Also output results as JSON to this path")

    args = parser.parse_args()

    # Read input
    companies = read_companies(args.input_csv)
    logger.info(f"Loaded {len(companies)} companies from {args.input_csv}")

    # Column diagnostic — helps debug empty-field issues
    if companies:
        cols = list(companies[0].keys())
        logger.info(f"CSV columns detected: {cols}")
        sample = companies[0]
        logger.info(
            f"First row sample: name='{sample.get('name', '???')}' "
            f"website='{sample.get('website', '???')}' "
            f"slug='{sample.get('slug', '???')}' "
            f"url='{sample.get('url', '???')}'"
        )

    # Filter
    if args.filter_active:
        companies = [c for c in companies if c.get("status", "").lower() == "active"]
        logger.info(f"Filtered to {len(companies)} active companies")

    # Limit
    if args.limit:
        companies = companies[:args.limit]
        logger.info(f"Limited to first {len(companies)} companies")

    # Resume
    resume_results = None
    if args.resume:
        resume_results = load_existing_results(args.output)

    # Run
    product_results = await run_batch(
        companies=companies,
        concurrency=args.concurrency,
        resume_results=resume_results,
    )

    # Write merged CSV
    all_companies = read_companies(args.input_csv)  # re-read full list for output
    if args.filter_active:
        all_companies = [c for c in all_companies if c.get("status", "").lower() == "active"]
    write_results(args.output, all_companies, product_results)

    # Optional JSON output
    if args.json:
        json_out = {}
        for key, val in product_results.items():
            if isinstance(val, ProductSignalCard):
                json_out[key] = val.to_row()
            else:
                json_out[key] = val
        with open(args.json, "w") as f:
            json.dump(json_out, f, indent=2, default=str)
        logger.info(f"JSON results saved to {args.json}")

    # Print summary
    cards = [v for v in product_results.values() if isinstance(v, ProductSignalCard)]
    if cards:
        live = sum(1 for c in cards if c.has_live_product)
        paying = sum(1 for c in cards if c.has_paying_customers)
        user_ev = sum(1 for c in cards if c.has_user_evidence)
        specific = sum(1 for c in cards if c.has_specific_user_problem)
        beachhead = sum(1 for c in cards if c.has_beachhead_segment)

        print(f"\n{'═' * 60}")
        print(f"  PRODUCT ENRICHMENT SUMMARY — {len(cards)} companies")
        print(f"{'═' * 60}")
        print(f"  Live product:          {live:>4} ({100*live/len(cards):.0f}%)")
        print(f"  Paying customers:      {paying:>4} ({100*paying/len(cards):.0f}%)")
        print(f"  External user evidence: {user_ev:>3} ({100*user_ev/len(cards):.0f}%)")
        print(f"  Specific user problem: {specific:>4} ({100*specific/len(cards):.0f}%)")
        print(f"  Beachhead segment:     {beachhead:>4} ({100*beachhead/len(cards):.0f}%)")
        print(f"  Avg maturity score:    {sum(c.product_maturity_score for c in cards)/len(cards):.1f}/10")
        print(f"  Avg understanding:     {sum(c.user_understanding_score for c in cards)/len(cards):.1f}/10")
        print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())