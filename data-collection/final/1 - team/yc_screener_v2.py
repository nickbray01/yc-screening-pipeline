"""
Greylock YC Screening Pipeline v2
===================================
LangChain pipeline that produces a flat grid of binary signals per YC company,
designed for batch filtering across an entire YC cohort.

Two output layers:
  1. SignalCard — flat row of booleans + scores, one per company. Filterable.
  2. CompanyScreeningResult — full analysis with reasoning. Drill-down.

Usage:
    # Single company → full result + signal card
    result = await screen_company("https://www.ycombinator.com/companies/trycardinal-ai")
    print(result.signals)          # SignalCard with all booleans
    print(result.signals.to_row()) # flat dict for CSV/DB

    # Batch → sorted signal cards, exportable to CSV
    results = await screen_batch([url1, url2, ...])
    export_csv(results, "batch_signals.csv")

Requires:
    pip install langchain langchain-anthropic httpx beautifulsoup4 pydantic

    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ─── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yc_screener")

# ─── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 2        # parallel scraping; LLM calls are serialized separately
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 3           # max retries per company before giving up
RETRY_BASE_DELAY = 30.0   # seconds; doubles on each retry

# Rate limit budget: 50 RPM / 30k input TPM / 8k output TPM
# ~800 output tokens per SignalCard → 10 calls/min max → 1 call per 7s (with headroom)
REQUEST_INTERVAL = 7.0    # minimum seconds between LLM calls


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SIGNAL CARD — the flat, filterable output
# ═══════════════════════════════════════════════════════════════════════════════

class SignalCard(BaseModel):
    """
    Flat grid of binary signals for one YC company.
    Every field is either a bool or a bounded int (1-10).
    Designed to be one row in a spreadsheet / DB table.
    """

    # ── Identity ──
    url: str = Field(default="", description="YC company page URL")
    company_name: str = Field(description="Company name")
    batch: str = Field(description="YC batch (e.g. W26, S25)")
    sector: str = Field(description="Primary sector")
    one_liner: str = Field(description="What they do in <15 words")

    # ── Founder Relevance Signals ──
    founder_worked_in_target_industry: bool = Field(
        description="TRUE if any founder previously worked at a company in the same industry "
                    "they are now building for. E.g., a fintech founder who worked at a bank or "
                    "payments company. FALSE if their experience is adjacent or unrelated."
    )
    founder_held_target_function: bool = Field(
        description="TRUE if any founder previously held the specific role/function that their "
                    "product replaces or augments. E.g., a sales-tool founder who was a sales rep, "
                    "or a DevOps-tool founder who was an SRE. FALSE if they only observed it."
    )
    founder_has_prior_exit: bool = Field(
        description="TRUE if any founder previously founded a company that was acquired or IPO'd."
    )
    founder_is_repeat_yc: bool = Field(
        description="TRUE if any founder has been through YC before (prior batch)."
    )
    founder_from_top_co: bool = Field(
        description="TRUE if any founder previously worked at a company widely recognized as "
                    "elite in tech (FAANG, Stripe, Coinbase, Palantir, top unicorns, etc.)."
    )
    domain_relevance_score: int = Field(
        description="1-10. Best single founder's domain relevance. "
                    "10 = built/ran the exact thing they're disrupting. 1 = no connection."
    )

    # ── Team Completeness Signals ──
    has_technical_cofounder: bool = Field(
        description="TRUE if at least one founder has a strong engineering/CS/ML background "
                    "and can build the product."
    )
    has_commercial_cofounder: bool = Field(
        description="TRUE if at least one founder has sales, marketing, BD, or GTM experience "
                    "and can sell the product."
    )
    has_domain_expert: bool = Field(
        description="TRUE if at least one founder has 3+ years of direct operating experience "
                    "in the industry they're building for."
    )
    is_solo_founder: bool = Field(
        description="TRUE if there is only one founder."
    )
    team_completeness_score: int = Field(
        description="1-10. 10 = all three dimensions covered (technical + commercial + domain) "
                    "by complementary founders. 5 = missing one dimension. 1 = solo non-technical."
    )

    # ── Composite ──
    overall_signal: str = Field(
        description="One of: STRONG, MODERATE, WEAK, PASS. Based on the combination of "
                    "founder relevance and team completeness."
    )
    one_line_risk: str = Field(
        description="The single biggest risk for this company in <15 words."
    )
    one_line_strength: str = Field(
        description="The single biggest strength for this company in <15 words."
    )

    def to_row(self) -> dict:
        """Export as flat dict for CSV/DataFrame."""
        return self.model_dump()

    @staticmethod
    def csv_columns() -> list[str]:
        """Column order for CSV export."""
        return [
            "url", "company_name", "batch", "sector", "one_liner",
            "overall_signal",
            "founder_worked_in_target_industry", "founder_held_target_function",
            "founder_has_prior_exit", "founder_is_repeat_yc", "founder_from_top_co",
            "domain_relevance_score",
            "has_technical_cofounder", "has_commercial_cofounder", "has_domain_expert",
            "is_solo_founder", "team_completeness_score",
            "one_line_risk", "one_line_strength",
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FULL ANALYSIS MODEL (for drill-down)
# ═══════════════════════════════════════════════════════════════════════════════

class FounderDetail(BaseModel):
    """Per-founder detail (only generated when you want the full analysis)."""
    name: str
    role: str
    is_technical: bool
    is_commercial: bool
    domain_relevance_score: int = Field(description="1-10")
    prior_companies: list[str]
    relevance_rationale: str = Field(description="2-3 sentences on why their background does/doesn't fit")


class FullAnalysis(BaseModel):
    """Deep analysis. Includes the SignalCard plus per-founder detail."""
    signals: SignalCard
    founders: list[FounderDetail]
    executive_summary: str = Field(description="3-4 sentences a partner can scan in 15 seconds")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCRAPER (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScrapedCompanyData:
    url: str
    name: str = ""
    one_liner: str = ""
    description: str = ""
    sector: str = ""
    batch: str = ""
    team_size: str = ""
    location: str = ""
    website: str = ""
    founders: list[dict] = field(default_factory=list)
    raw_text: str = ""

    def to_context_string(self) -> str:
        lines = [
            f"Company: {self.name}",
            f"URL: {self.url}",
            f"One-liner: {self.one_liner}",
            f"Description: {self.description}",
            f"Sector: {self.sector}",
            f"Batch: {self.batch}",
            f"Team Size: {self.team_size}",
            f"Location: {self.location}",
            f"Website: {self.website}",
            "",
            "Founders:",
        ]
        for f in self.founders:
            lines.append(f"  - {f.get('name', 'Unknown')}: {f.get('bio', 'No bio available')}")
        if not self.founders:
            lines.append("  (No founder data found on page)")
        return "\n".join(lines)


async def scrape_yc_company(url: str) -> ScrapedCompanyData:
    data = ScrapedCompanyData(url=url)

    if not url.startswith("http"):
        url = f"https://www.ycombinator.com/companies/{url}"
    data.url = url

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    next_data_tag = soup.find("script", id="__NEXT_DATA__")
    if next_data_tag and next_data_tag.string:
        try:
            next_data = json.loads(next_data_tag.string)
            props = next_data.get("props", {}).get("pageProps", {})
            company = props.get("company", {})

            if company:
                data.name = company.get("name", "")
                data.one_liner = company.get("one_liner", "")
                data.description = company.get("long_description", "") or company.get("description", "")
                data.sector = company.get("industries", [""])[0] if company.get("industries") else ""
                data.batch = company.get("batch_name", "")
                data.team_size = str(company.get("team_size", ""))
                data.location = company.get("location", "")
                data.website = company.get("website", "")

                for f in company.get("founders", []):
                    data.founders.append({
                        "name": f.get("full_name", ""),
                        "title": f.get("title", ""),
                        "bio": f.get("bio", ""),
                        "linkedin": f.get("linkedin_url", ""),
                    })

                logger.info(f"Scraped {data.name} ({len(data.founders)} founders)")
                return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"__NEXT_DATA__ parse failed: {e}")

    # HTML fallback
    h1 = soup.find("h1")
    if h1:
        data.name = h1.get_text(strip=True)

    tagline = soup.find("div", class_=re.compile(r"tagline|one.?liner|subtitle", re.I))
    if tagline:
        data.one_liner = tagline.get_text(strip=True)

    desc_section = soup.find("section", class_=re.compile(r"description|about", re.I))
    if desc_section:
        data.description = desc_section.get_text(strip=True)

    data.raw_text = soup.get_text(separator="\n", strip=True)[:5000]

    founder_sections = soup.find_all("div", class_=re.compile(r"founder", re.I))
    for fs in founder_sections:
        name_el = fs.find(["h3", "h4", "strong", "span"])
        bio_el = fs.find("p")
        data.founders.append({
            "name": name_el.get_text(strip=True) if name_el else "Unknown",
            "bio": bio_el.get_text(strip=True) if bio_el else "",
        })

    logger.info(f"Scraped {data.name or 'Unknown'} via HTML fallback ({len(data.founders)} founders)")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LANGCHAIN CHAINS — two modes: fast (signals only) and full (signals + detail)
# ═══════════════════════════════════════════════════════════════════════════════

# ── FAST MODE: signals only ──
# Optimized for batch processing. Smaller output schema = fewer tokens.

SIGNAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a VC screening system. Analyze the YC company below and produce a 
structured signal card with binary flags and scores.

RULES:
- founder_worked_in_target_industry: TRUE only if a founder was EMPLOYED at a company in the 
  same industry as this startup. Adjacent industries don't count.
- founder_held_target_function: TRUE only if a founder personally DID the job this product 
  replaces. A PM who watched sales reps doesn't count. A sales rep building a sales tool counts.
- founder_from_top_co: Use a high bar. FAANG, Stripe, Coinbase, Palantir, Databricks, 
  Snowflake, etc. Not any random startup.
- domain_relevance_score: rate the BEST founder, not the average.
- overall_signal: STRONG = relevance ≥7 AND completeness ≥7. 
  MODERATE = one of them ≥7. WEAK = both 4-6. PASS = either <4.
- If info is sparse, score conservatively and note it in one_line_risk.

{format_instructions}"""),
    ("human", """{company_context}"""),
])


# ── FULL MODE: signals + per-founder detail ──

FULL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a VC screening system at Greylock Partners. Produce both:
1. A SignalCard with binary flags and scores (same rules as signal-only mode)
2. Per-founder detail with relevance rationale

RULES FOR SIGNALS:
- founder_worked_in_target_industry: TRUE only if a founder was EMPLOYED at a company in the 
  same industry as this startup. Adjacent industries don't count.
- founder_held_target_function: TRUE only if a founder personally DID the job this product 
  replaces. A PM who watched sales reps doesn't count. A sales rep building a sales tool counts.
- founder_from_top_co: Use a high bar. FAANG, Stripe, Coinbase, Palantir, Databricks, etc.
- domain_relevance_score: rate the BEST founder, not the average.
- overall_signal: STRONG = relevance ≥7 AND completeness ≥7. 
  MODERATE = one of them ≥7. WEAK = both 4-6. PASS = either <4.
- If info is sparse, score conservatively and note it in one_line_risk.

{format_instructions}"""),
    ("human", """{company_context}"""),
])


def build_signal_chain(model_name: str = DEFAULT_MODEL):
    """Fast chain: produces SignalCard only."""
    llm = ChatAnthropic(model=model_name, temperature=0.1, max_tokens=2048, max_retries=0)
    parser = PydanticOutputParser(pydantic_object=SignalCard)
    chain = SIGNAL_PROMPT | llm | parser
    return chain, parser


def build_full_chain(model_name: str = DEFAULT_MODEL):
    """Full chain: produces FullAnalysis (SignalCard + per-founder detail)."""
    llm = ChatAnthropic(model=model_name, temperature=0.2, max_tokens=4096, max_retries=0)
    parser = PydanticOutputParser(pydantic_object=FullAnalysis)
    chain = FULL_PROMPT | llm | parser
    return chain, parser


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class YCScreeningPipeline:
    """
    Main pipeline. Two modes:
      - screen()      → FullAnalysis (signals + per-founder detail)
      - screen_fast() → SignalCard only (optimized for batch)
    
    Usage:
        pipeline = YCScreeningPipeline()

        # Full analysis (single company deep-dive)
        full = await pipeline.screen("https://www.ycombinator.com/companies/trycardinal-ai")
        print(full.signals)     # SignalCard
        print(full.founders)    # per-founder detail

        # Fast signals only (batch mode)
        card = await pipeline.screen_fast("https://www.ycombinator.com/companies/trycardinal-ai")
        print(card.to_row())    # flat dict

        # Batch → list of SignalCards
        cards = await pipeline.screen_batch_fast(["url1", "url2", ...])
        export_csv(cards, "signals.csv")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_concurrent: int = MAX_CONCURRENT,
    ):
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pace_lock = asyncio.Lock()   # serializes LLM calls
        self._last_call_time: float = 0.0

        # Build both chains
        self._signal_chain, self._signal_parser = build_signal_chain(model_name)
        self._full_chain, self._full_parser = build_full_chain(model_name)

    async def _pace(self) -> None:
        """Block until REQUEST_INTERVAL has elapsed since the last LLM call."""
        async with self._pace_lock:
            now = asyncio.get_event_loop().time()
            wait = REQUEST_INTERVAL - (now - self._last_call_time)
            if wait > 0:
                logger.debug(f"Pacing: waiting {wait:.1f}s before next LLM call")
                await asyncio.sleep(wait)
            self._last_call_time = asyncio.get_event_loop().time()

    async def _scrape(self, url: str) -> ScrapedCompanyData:
        scraped = await scrape_yc_company(url)
        if not scraped.founders and scraped.raw_text:
            scraped.description += f"\n\nRaw page text:\n{scraped.raw_text[:3000]}"
        return scraped

    async def screen_fast(self, url: str) -> SignalCard:
        """
        Fast mode: scrape + produce SignalCard only.
        Optimized for batch — fewer output tokens.
        """
        scraped = await self._scrape(url)
        logger.info(f"[fast] Analyzing: {scraped.name or url}")

        await self._pace()
        card = await self._signal_chain.ainvoke({
            "company_context": scraped.to_context_string(),
            "format_instructions": self._signal_parser.get_format_instructions(),
        })

        # Patch url and batch from scraper
        card.url = scraped.url
        if not card.batch and scraped.batch:
            card.batch = scraped.batch

        logger.info(
            f"[fast] {card.company_name} → {card.overall_signal} | "
            f"relevance={card.domain_relevance_score} completeness={card.team_completeness_score}"
        )
        return card

    async def screen(self, url: str) -> FullAnalysis:
        """
        Full mode: scrape + produce FullAnalysis (SignalCard + per-founder detail).
        Use for single-company deep dives.
        """
        scraped = await self._scrape(url)
        logger.info(f"[full] Analyzing: {scraped.name or url}")

        await self._pace()
        result = await self._full_chain.ainvoke({
            "company_context": scraped.to_context_string(),
            "format_instructions": self._full_parser.get_format_instructions(),
        })

        result.signals.url = scraped.url
        if not result.signals.batch and scraped.batch:
            result.signals.batch = scraped.batch

        logger.info(
            f"[full] {result.signals.company_name} → {result.signals.overall_signal} | "
            f"relevance={result.signals.domain_relevance_score} "
            f"completeness={result.signals.team_completeness_score}"
        )
        return result

    async def _fast_with_semaphore(self, url: str) -> SignalCard | dict:
        async with self._semaphore:
            delay = RETRY_BASE_DELAY
            for attempt in range(MAX_RETRIES + 1):
                try:
                    return await self.screen_fast(url)
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "rate" in err_str or "overloaded" in err_str
                    if attempt < MAX_RETRIES and is_rate_limit:
                        logger.warning(
                            f"Rate limit on {url}, waiting {delay:.0f}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(f"Failed: {url} → {e}")
                        return {"company_name": url, "error": str(e)}

    async def screen_batch_fast(self, urls: list[str]) -> list[SignalCard | dict]:
        """
        Batch mode: screen all URLs concurrently, return list of SignalCards.
        Sorted by signal strength.
        """
        logger.info(f"Batch: {len(urls)} companies (concurrency={self.max_concurrent})")

        tasks = [self._fast_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Sort: STRONG first, then by relevance score descending
        signal_order = {"STRONG": 0, "MODERATE": 1, "WEAK": 2, "PASS": 3}
        def sort_key(r):
            if isinstance(r, dict):
                return (4, 0)
            return (signal_order.get(r.overall_signal, 3), -r.domain_relevance_score)
        results.sort(key=sort_key)

        ok = [r for r in results if isinstance(r, SignalCard)]
        logger.info(
            f"Batch done: {sum(1 for r in ok if r.overall_signal == 'STRONG')} STRONG, "
            f"{sum(1 for r in ok if r.overall_signal == 'MODERATE')} MODERATE, "
            f"{len(results) - len(ok)} errors"
        )
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OUTPUT / EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_csv(results: list[SignalCard | dict], filepath: str) -> str:
    """Export signal cards to CSV. Returns the filepath."""
    columns = SignalCard.csv_columns()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if isinstance(r, SignalCard):
                writer.writerow(r.to_row())
            else:
                writer.writerow({"company_name": r.get("company_name", "ERROR"), "one_liner": r.get("error", "")})
    logger.info(f"Exported {len(results)} rows to {filepath}")
    return filepath


def export_json(results: list[SignalCard | dict], filepath: str) -> str:
    """Export signal cards to JSON."""
    output = []
    for r in results:
        if isinstance(r, SignalCard):
            output.append(r.to_row())
        else:
            output.append(r)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Exported {len(results)} results to {filepath}")
    return filepath


def format_signal_card(card: SignalCard) -> str:
    """Pretty-print a single signal card for terminal output."""
    def flag(b: bool) -> str:
        return "✓" if b else "✗"

    return f"""
{'═' * 70}
  {card.company_name}  [{card.overall_signal}]  {card.batch}
{'═' * 70}
  {card.one_liner}
  Sector: {card.sector}

  ┌── Founder Relevance ──────────────────────────────────
  │  Worked in target industry    {flag(card.founder_worked_in_target_industry)}
  │  Held target function         {flag(card.founder_held_target_function)}
  │  Prior exit                   {flag(card.founder_has_prior_exit)}
  │  Repeat YC founder            {flag(card.founder_is_repeat_yc)}
  │  From top company             {flag(card.founder_from_top_co)}
  │  Domain relevance score       {card.domain_relevance_score}/10
  │
  ├── Team Completeness ──────────────────────────────────
  │  Technical co-founder         {flag(card.has_technical_cofounder)}
  │  Commercial co-founder        {flag(card.has_commercial_cofounder)}
  │  Domain expert                {flag(card.has_domain_expert)}
  │  Solo founder                 {flag(card.is_solo_founder)}
  │  Team completeness score      {card.team_completeness_score}/10
  │
  ├── Assessment ─────────────────────────────────────────
  │  Strength: {card.one_line_strength}
  │  Risk:     {card.one_line_risk}
  └───────────────────────────────────────────────────────
"""


def format_batch_table(results: list[SignalCard | dict]) -> str:
    """Print a filterable batch table to terminal."""
    header = (
        f"  {'#':<4} {'Company':<22} {'Signal':<10} "
        f"{'Ind':>3} {'Fn':>3} {'Ex':>3} {'YC':>3} {'Top':>3} {'Rel':>4} "
        f"{'Tech':>4} {'Comm':>4} {'Dom':>4} {'Solo':>4} {'Cmp':>4} "
        f"{'Risk':<30}"
    )
    sep = f"  {'─' * 120}"
    lines = [
        f"\n{'═' * 124}",
        f"  YC SCREENING — {len(results)} companies",
        f"{'═' * 124}",
        f"  Legend: Ind=Industry Fn=Function Ex=Exit YC=RepeatYC Top=TopCo Rel=Relevance",
        f"          Tech=Technical Comm=Commercial Dom=Domain Solo=Solo Cmp=Completeness",
        sep,
        header,
        sep,
    ]

    def f(b):
        return "✓" if b else "·"

    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            lines.append(f"  {i:<4} {'ERROR':<22} {'—':<10} {r.get('error', '')[:60]}")
            continue
        lines.append(
            f"  {i:<4} {r.company_name[:21]:<22} {r.overall_signal:<10} "
            f"{f(r.founder_worked_in_target_industry):>3} "
            f"{f(r.founder_held_target_function):>3} "
            f"{f(r.founder_has_prior_exit):>3} "
            f"{f(r.founder_is_repeat_yc):>3} "
            f"{f(r.founder_from_top_co):>3} "
            f"{r.domain_relevance_score:>4} "
            f"{f(r.has_technical_cofounder):>4} "
            f"{f(r.has_commercial_cofounder):>4} "
            f"{f(r.has_domain_expert):>4} "
            f"{f(r.is_solo_founder):>4} "
            f"{r.team_completeness_score:>4} "
            f"{r.one_line_risk[:30]:<30}"
        )

    lines.append(sep)

    # Summary counts
    ok = [r for r in results if isinstance(r, SignalCard)]
    strong = [r for r in ok if r.overall_signal == "STRONG"]
    mod = [r for r in ok if r.overall_signal == "MODERATE"]

    lines.append(f"\n  STRONG: {len(strong)}  |  MODERATE: {len(mod)}  |  "
                 f"WEAK/PASS: {len(ok) - len(strong) - len(mod)}")

    if strong:
        lines.append(f"\n  ★ RECOMMEND FOR PARTNER CALL:")
        for r in strong:
            lines.append(f"    → {r.company_name}: {r.one_line_strength}")

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def screen_company(url: str, model: str = DEFAULT_MODEL) -> FullAnalysis:
    """Screen single company — full analysis with SignalCard + founder details."""
    pipeline = YCScreeningPipeline(model_name=model)
    return await pipeline.screen(url)


async def screen_company_fast(url: str, model: str = DEFAULT_MODEL) -> SignalCard:
    """Screen single company — SignalCard only (fewer tokens)."""
    pipeline = YCScreeningPipeline(model_name=model)
    return await pipeline.screen_fast(url)


async def screen_batch(urls: list[str], model: str = DEFAULT_MODEL) -> list[SignalCard | dict]:
    """Screen batch — returns list of SignalCards sorted by signal strength."""
    pipeline = YCScreeningPipeline(model_name=model)
    return await pipeline.screen_batch_fast(urls)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLI
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    import sys

    args = sys.argv[1:]

    # Check for flags
    full_mode = "--full" in args
    csv_out = None
    json_out = None
    urls = []

    for arg in args:
        if arg == "--full":
            continue
        elif arg.startswith("--csv="):
            csv_out = arg.split("=", 1)[1]
        elif arg.startswith("--json="):
            json_out = arg.split("=", 1)[1]
        else:
            urls.append(arg)

    # Read from stdin if no URL args
    if not urls:
        if not sys.stdin.isatty():
            urls = [line.strip() for line in sys.stdin if line.strip() and not line.startswith("#")]
        else:
            print("Usage:")
            print("  python yc_screener.py URL [URL...] [--full] [--csv=output.csv] [--json=output.json]")
            print("  cat urls.txt | python yc_screener.py --csv=batch.csv")
            print("")
            print("Modes:")
            print("  Default: fast signal cards (optimized for batch)")
            print("  --full:  full analysis with per-founder detail (single company)")
            return

    pipeline = YCScreeningPipeline()

    if len(urls) == 1 and full_mode:
        # Single company, full analysis
        result = await pipeline.screen(urls[0])
        print(format_signal_card(result.signals))
        print(f"  Executive Summary:")
        print(f"  {result.executive_summary}\n")
        for f in result.founders:
            print(f"  ┌─ {f.name} ({f.role}) — Relevance: {f.domain_relevance_score}/10")
            print(f"  │  Technical: {'✓' if f.is_technical else '✗'}  Commercial: {'✓' if f.is_commercial else '✗'}")
            print(f"  │  Prior: {', '.join(f.prior_companies[:5])}")
            print(f"  └─ {f.relevance_rationale}")
            print()
    elif len(urls) == 1:
        # Single company, fast mode
        card = await pipeline.screen_fast(urls[0])
        print(format_signal_card(card))
    else:
        # Batch mode (always fast)
        results = await pipeline.screen_batch_fast(urls)
        print(format_batch_table(results))

        if csv_out:
            export_csv(results, csv_out)
            print(f"  Saved: {csv_out}")
        if json_out:
            export_json(results, json_out)
            print(f"  Saved: {json_out}")

        # Default JSON export
        if not csv_out and not json_out:
            export_json(results, "screening_results.json")
            print(f"  Saved: screening_results.json")


if __name__ == "__main__":
    asyncio.run(main())
