"""
Product Signal Enrichment Module
==================================
Gathers evidence of founder product understanding from multiple web sources,
then uses an LLM to synthesize into filterable boolean signals.

Corinne Riley's product pillar: "Do you deeply understand the user problem?"
Best signal: founders who say "we had X conversations and here's our hypothesis"
Worst signal: vague "AI for X" with no evidence of user contact.

Sources searched (in priority order):
  1. Company website — live product indicators, signup flows, pricing pages
  2. Product Hunt — launches, upvotes, maker comments
  3. App stores — iOS/Android presence, ratings, review count
  4. G2 / Capterra — enterprise product reviews
  5. GitHub — open-source repos, stars, commit recency
  6. HackerNews — Show HN posts, comment sentiment
  7. News / blogs — founder interviews mentioning user research
  8. Twitter/X — founder posts about user feedback, product decisions

Architecture:
  gather_product_evidence(company) → EvidenceBundle (raw, from multiple sources)
  score_product_signals(evidence)  → ProductSignalCard (booleans + scores, from LLM)

Usage:
    enricher = ProductEnricher()
    signals = await enricher.enrich("https://www.ycombinator.com/companies/trycardinal-ai")
    print(signals.has_live_product)       # True/False
    print(signals.product_evidence_score) # 1-10
    print(signals.to_row())              # flat dict for CSV

Requires:
    pip install langchain langchain-anthropic httpx beautifulsoup4 pydantic

    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger("product_enricher")

DEFAULT_MODEL = "claude-sonnet-4-20250514"
REQUEST_TIMEOUT = 15.0
MAX_CONCURRENT_FETCHES = 8


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EVIDENCE MODEL — raw data gathered from the web
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceResult:
    """One piece of evidence from one source."""
    source: str          # e.g. "company_website", "product_hunt", "github"
    url: str
    found: bool          # did we find anything?
    signals: dict = field(default_factory=dict)   # structured signals extracted
    raw_text: str = ""   # relevant text snippet (truncated)
    error: str = ""


@dataclass
class EvidenceBundle:
    """All evidence gathered for one company."""
    company_name: str
    company_url: str
    company_website: str
    one_liner: str
    results: list[SourceResult] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Format all evidence for the LLM."""
        lines = [
            f"Company: {self.company_name}",
            f"YC URL: {self.company_url}",
            f"Website: {self.company_website}",
            f"One-liner: {self.one_liner}",
            "",
            "=== PRODUCT EVIDENCE FROM WEB SOURCES ===",
            "",
        ]
        for r in self.results:
            status = "FOUND" if r.found else "NOT FOUND"
            lines.append(f"--- {r.source.upper()} [{status}] ---")
            if r.error:
                lines.append(f"  Error: {r.error}")
            if r.signals:
                for k, v in r.signals.items():
                    lines.append(f"  {k}: {v}")
            if r.raw_text:
                lines.append(f"  Context: {r.raw_text[:500]}")
            lines.append("")
        return "\n".join(lines)

    @property
    def sources_found(self) -> int:
        return sum(1 for r in self.results if r.found)

    @property
    def sources_checked(self) -> int:
        return len(self.results)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRODUCT SIGNAL CARD — the filterable output
# ═══════════════════════════════════════════════════════════════════════════════

class ProductSignalCard(BaseModel):
    """
    Filterable product signals for one company.
    Designed to merge into the main SignalCard as additional columns.
    """

    # ── Binary signals ──
    has_live_product: bool = Field(
        description="TRUE if there is evidence of a live, usable product — signup flow, "
                    "app store listing, active GitHub repo, or user-facing demo. "
                    "A landing page with only a waitlist does NOT count."
    )
    has_paying_customers: bool = Field(
        description="TRUE if there is evidence of paid users — pricing page with real tiers, "
                    "customer logos, case studies, revenue mentions, or app store paid downloads. "
                    "FALSE if free-only or no evidence."
    )
    has_user_evidence: bool = Field(
        description="TRUE if there is external evidence that real users exist — app store reviews, "
                    "G2/Capterra reviews, Product Hunt upvotes from non-team-members, "
                    "HackerNews comments from users, or testimonials from named customers."
    )
    has_public_iteration: bool = Field(
        description="TRUE if there is evidence the product has iterated based on user feedback — "
                    "changelog, version history, GitHub commit activity, Product Hunt 'maker' comments "
                    "responding to feedback, blog posts about pivots or user-driven changes."
    )
    has_specific_user_problem: bool = Field(
        description="TRUE if the company description or founder communications articulate a "
                    "SPECIFIC user problem (not a vague 'AI for X'). The test: could you name "
                    "the exact job title of the person who has this problem and describe their "
                    "workflow before and after the product?"
    )
    has_beachhead_segment: bool = Field(
        description="TRUE if there is evidence of a specific initial customer segment — not "
                    "'everyone' or 'all enterprises' but a named vertical, persona, or use case "
                    "that defines the first adopters."
    )

    # ── Scores ──
    product_maturity_score: int = Field(
        description="1-10 product maturity. 10 = live product with paying customers, reviews, "
                    "and evidence of iteration. 5 = live product but no external validation. "
                    "1 = idea stage with no evidence of a built product."
    )
    user_understanding_score: int = Field(
        description="1-10 evidence of founder understanding of the user problem. "
                    "10 = specific problem, named user persona, evidence of customer discovery. "
                    "5 = reasonable problem statement but generic. "
                    "1 = buzzword soup with no evidence of user contact."
    )

    # ── Evidence summary ──
    sources_with_evidence: int = Field(
        description="How many distinct sources (out of those checked) had product evidence."
    )
    strongest_evidence: str = Field(
        description="One sentence: the single strongest piece of product evidence found."
    )
    biggest_product_gap: str = Field(
        description="One sentence: the biggest concern about this product from the evidence."
    )

    def to_row(self) -> dict:
        return self.model_dump()

    @staticmethod
    def csv_columns() -> list[str]:
        return [
            "has_live_product", "has_paying_customers", "has_user_evidence",
            "has_public_iteration", "has_specific_user_problem", "has_beachhead_segment",
            "product_maturity_score", "user_understanding_score",
            "sources_with_evidence", "strongest_evidence", "biggest_product_gap",
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SOURCE SCRAPERS — each returns a SourceResult
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    """Safe fetch with timeout and error handling."""
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


# ── 3a. Company Website ──────────────────────────────────────────────────────

async def scrape_company_website(client: httpx.AsyncClient, website_url: str) -> SourceResult:
    """
    Check the company's own website for product indicators:
    - Login/signup links (live product)
    - Pricing page (paying customers)
    - Changelog or docs (iteration)
    - Customer logos or testimonials
    - Specific problem framing on the homepage
    """
    result = SourceResult(source="company_website", url=website_url, found=False)

    if not website_url or website_url in ("", "http://", "https://"):
        result.error = "No website URL available"
        return result

    resp = await _fetch(client, website_url)
    if not resp:
        result.error = "Could not fetch website"
        return result

    text = resp.text.lower()
    soup = BeautifulSoup(resp.text, "html.parser")
    visible_text = soup.get_text(separator=" ", strip=True)[:3000]

    # Signal detection
    signals = {}

    # Login/signup flow
    has_auth = bool(re.search(
        r'(sign\s*up|sign\s*in|log\s*in|create\s*account|get\s*started|start\s*free|try\s*free|book\s*a\s*demo|request\s*demo|schedule\s*demo)',
        text
    ))
    signals["has_auth_flow"] = has_auth

    # Pricing page
    pricing_links = soup.find_all("a", href=re.compile(r"pricing", re.I))
    has_pricing = len(pricing_links) > 0 or bool(re.search(r'(pricing|plans|per\s*month|\$\d+|free\s*tier|enterprise\s*plan)', text))
    signals["has_pricing_page"] = has_pricing

    # Docs / changelog
    has_docs = bool(soup.find_all("a", href=re.compile(r"(docs|documentation|changelog|release|api)", re.I)))
    signals["has_docs_or_changelog"] = has_docs

    # Customer logos / testimonials
    has_social_proof = bool(re.search(
        r'(trusted\s*by|used\s*by|customer|testimonial|case\s*stud|logo|partner)',
        text
    ))
    signals["has_social_proof"] = has_social_proof

    # Waitlist only (negative signal)
    is_waitlist_only = bool(re.search(r'(waitlist|wait\s*list|coming\s*soon|launching\s*soon)', text)) and not has_auth
    signals["is_waitlist_only"] = is_waitlist_only

    result.signals = signals
    result.found = has_auth or has_pricing or has_docs or has_social_proof
    result.raw_text = visible_text[:800]

    return result


# ── 3b. Product Hunt ─────────────────────────────────────────────────────────

async def scrape_product_hunt(client: httpx.AsyncClient, company_name: str) -> SourceResult:
    """Search Product Hunt for the company. Extract upvotes, comments, maker activity."""
    result = SourceResult(source="product_hunt", url="", found=False)

    search_url = f"https://www.producthunt.com/search?q={company_name.replace(' ', '+')}"
    result.url = search_url

    resp = await _fetch(client, search_url)
    if not resp:
        result.error = "Could not search Product Hunt"
        return result

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # Check if company name appears in results
    name_lower = company_name.lower()
    if name_lower in text.lower():
        result.found = True
        result.signals["listed_on_product_hunt"] = True
        result.raw_text = text[:500]

        # Try to extract upvote count (PH uses various formats)
        upvote_match = re.search(r'(\d+)\s*upvote', text.lower())
        if upvote_match:
            result.signals["upvotes"] = int(upvote_match.group(1))
    else:
        result.signals["listed_on_product_hunt"] = False

    return result


# ── 3c. GitHub ───────────────────────────────────────────────────────────────

async def scrape_github(client: httpx.AsyncClient, company_name: str, website_url: str = "") -> SourceResult:
    """Search GitHub for repos by the company. Check stars, recent commits."""
    result = SourceResult(source="github", url="", found=False)

    await asyncio.sleep(12)

    search_url = f"https://api.github.com/search/repositories?q={company_name}&sort=stars&per_page=5"
    result.url = search_url

    github_token = os.environ.get("GITHUB_TOKEN")
    extra_headers = {"Authorization": f"Bearer {github_token}"} if github_token else {}
    try:
        raw = await client.get(search_url, headers=extra_headers, follow_redirects=True)
        raw.raise_for_status()
        resp = raw
    except Exception:
        resp = None

    if not resp:
        result.error = "Could not search GitHub"
        return result

    try:
        data = resp.json()
        items = data.get("items", [])

        # Filter to repos that likely belong to this company
        name_lower = company_name.lower().replace(" ", "")
        relevant = [
            i for i in items
            if name_lower in i.get("full_name", "").lower().replace(" ", "")
            or name_lower in (i.get("description") or "").lower().replace(" ", "")
            or (website_url and website_url.replace("https://", "").replace("http://", "").split("/")[0]
                in (i.get("homepage") or ""))
        ]

        if relevant:
            top = relevant[0]
            result.found = True
            result.signals = {
                "repo_name": top.get("full_name", ""),
                "stars": top.get("stargazers_count", 0),
                "forks": top.get("forks_count", 0),
                "open_issues": top.get("open_issues_count", 0),
                "language": top.get("language", ""),
                "last_push": top.get("pushed_at", ""),
                "description": top.get("description", ""),
            }
            result.raw_text = f"GitHub repo: {top.get('full_name')} — {top.get('description', '')} — {top.get('stargazers_count', 0)} stars"
        else:
            result.signals["has_public_repo"] = False

    except (json.JSONDecodeError, KeyError):
        result.error = "Failed to parse GitHub response"

    return result


# ── 3d. App Stores ───────────────────────────────────────────────────────────

async def scrape_app_store_presence(client: httpx.AsyncClient, company_name: str) -> SourceResult:
    """Check for iOS App Store presence via iTunes Search API."""
    result = SourceResult(source="app_store", url="", found=False)

    search_url = f"https://itunes.apple.com/search?term={company_name.replace(' ', '+')}&entity=software&limit=5"
    result.url = search_url

    resp = await _fetch(client, search_url)
    if not resp:
        result.error = "Could not search App Store"
        return result

    try:
        data = resp.json()
        results_list = data.get("results", [])

        name_lower = company_name.lower()
        relevant = [
            r for r in results_list
            if name_lower in r.get("trackName", "").lower()
            or name_lower in r.get("sellerName", "").lower()
        ]

        if relevant:
            app = relevant[0]
            result.found = True
            result.signals = {
                "app_name": app.get("trackName", ""),
                "rating": app.get("averageUserRating", 0),
                "rating_count": app.get("userRatingCount", 0),
                "price": app.get("price", 0),
                "description_snippet": (app.get("description") or "")[:300],
            }
            result.raw_text = f"App: {app.get('trackName')} — Rating: {app.get('averageUserRating', 'N/A')} ({app.get('userRatingCount', 0)} reviews)"
        else:
            result.signals["has_ios_app"] = False

    except (json.JSONDecodeError, KeyError):
        result.error = "Failed to parse App Store response"

    return result


# ── 3e. Hacker News ──────────────────────────────────────────────────────────

async def scrape_hacker_news(client: httpx.AsyncClient, company_name: str) -> SourceResult:
    """Search Hacker News (Algolia API) for Show HN posts or discussions."""
    result = SourceResult(source="hacker_news", url="", found=False)

    search_url = f"https://hn.algolia.com/api/v1/search?query={company_name}&tags=story&hitsPerPage=5"
    result.url = search_url

    resp = await _fetch(client, search_url)
    if not resp:
        result.error = "Could not search Hacker News"
        return result

    try:
        data = resp.json()
        hits = data.get("hits", [])

        name_lower = company_name.lower()
        relevant = [
            h for h in hits
            if name_lower in (h.get("title") or "").lower()
            or name_lower in (h.get("story_text") or "").lower()
        ]

        if relevant:
            top = relevant[0]
            result.found = True
            result.signals = {
                "title": top.get("title", ""),
                "points": top.get("points", 0),
                "num_comments": top.get("num_comments", 0),
                "is_show_hn": (top.get("title") or "").lower().startswith("show hn"),
                "date": top.get("created_at", ""),
            }
            result.raw_text = f"HN: '{top.get('title')}' — {top.get('points', 0)} points, {top.get('num_comments', 0)} comments"
        else:
            result.signals["found_on_hn"] = False

    except (json.JSONDecodeError, KeyError):
        result.error = "Failed to parse HN response"

    return result


# ── 3f. G2 Reviews ───────────────────────────────────────────────────────────

async def scrape_g2(client: httpx.AsyncClient, company_name: str) -> SourceResult:
    """Check G2 for enterprise product reviews."""
    result = SourceResult(source="g2_reviews", url="", found=False)

    # G2 doesn't have a public API, but we can check if a product page exists
    slug = company_name.lower().replace(" ", "-").replace(".", "")
    g2_url = f"https://www.g2.com/products/{slug}/reviews"
    result.url = g2_url

    resp = await _fetch(client, g2_url)
    if not resp or resp.status_code == 404:
        # Try search page
        result.signals["has_g2_page"] = False
        return result

    text = resp.text.lower()
    if "reviews" in text and company_name.lower() in text:
        result.found = True
        result.signals["has_g2_page"] = True

        # Try to extract rating
        rating_match = re.search(r'(\d+\.?\d*)\s*out\s*of\s*5', text)
        if rating_match:
            result.signals["g2_rating"] = float(rating_match.group(1))

        review_count_match = re.search(r'(\d+)\s*review', text)
        if review_count_match:
            result.signals["g2_review_count"] = int(review_count_match.group(1))

        result.raw_text = f"G2 page found for {company_name}"
    else:
        result.signals["has_g2_page"] = False

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EVIDENCE GATHERER — runs all scrapers concurrently
# ═══════════════════════════════════════════════════════════════════════════════

async def gather_product_evidence(
    company_name: str,
    website_url: str = "",
    yc_url: str = "",
    one_liner: str = "",
) -> EvidenceBundle:
    """
    Run all source scrapers concurrently. Returns an EvidenceBundle
    with results from every source (including failures).
    """
    bundle = EvidenceBundle(
        company_name=company_name,
        company_url=yc_url,
        company_website=website_url,
        one_liner=one_liner,
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, headers=headers) as client:
        tasks = [
            scrape_company_website(client, website_url),
            scrape_product_hunt(client, company_name),
            scrape_github(client, company_name, website_url),
            scrape_app_store_presence(client, company_name),
            scrape_hacker_news(client, company_name),
            scrape_g2(client, company_name),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, SourceResult):
                bundle.results.append(r)
            elif isinstance(r, Exception):
                bundle.results.append(SourceResult(
                    source="unknown", url="", found=False, error=str(r)
                ))

    logger.info(
        f"Evidence gathered for {company_name}: "
        f"{bundle.sources_found}/{bundle.sources_checked} sources with data"
    )
    return bundle


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LLM SCORING — synthesize evidence into ProductSignalCard
# ═══════════════════════════════════════════════════════════════════════════════

PRODUCT_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a VC product analyst. Given web evidence about a YC company, 
produce a structured product signal card.

RULES:
- has_live_product: needs signup/login, app store listing, or active repo. Waitlist-only = FALSE.
- has_paying_customers: needs pricing tiers, revenue mentions, or paid app. Free-only = FALSE.
- has_user_evidence: needs EXTERNAL validation (reviews, ratings, upvotes from non-founders).
  Self-reported "we have X customers" without corroboration = FALSE.
- has_public_iteration: needs changelog, version history, commit activity, or documented pivots.
- has_specific_user_problem: the company description must name a specific workflow, persona, 
  or pain point. "AI for sales" = FALSE. "AI that writes cold outbound emails for SDRs at 
  mid-market SaaS companies" = TRUE.
- has_beachhead_segment: is there a named first-adopter group? Not "everyone" or "all enterprises."

Score conservatively. Only mark TRUE when there is clear evidence in the data provided.
If a source returned no data, that is absence of evidence, not evidence of absence — 
but it should lower your confidence.

{format_instructions}"""),
    ("human", """{evidence_context}"""),
])


def build_product_scoring_chain(model_name: str = DEFAULT_MODEL):
    llm = ChatAnthropic(model=model_name, temperature=0.1, max_tokens=2048)
    parser = PydanticOutputParser(pydantic_object=ProductSignalCard)
    chain = PRODUCT_SCORING_PROMPT | llm | parser
    return chain, parser


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENRICHER — the main class
# ═══════════════════════════════════════════════════════════════════════════════

class ProductEnricher:
    """
    Full pipeline: gather evidence → score with LLM → return ProductSignalCard.

    Usage:
        enricher = ProductEnricher()
        card = await enricher.enrich("Cardinal", "https://trycardinal.ai", yc_url="...")
        print(card.has_live_product)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.chain, self.parser = build_product_scoring_chain(model_name)

    async def enrich(
        self,
        company_name: str,
        website_url: str = "",
        yc_url: str = "",
        one_liner: str = "",
    ) -> ProductSignalCard:
        """Gather evidence and score product signals for one company."""

        # Step 1: Gather evidence from all sources
        evidence = await gather_product_evidence(
            company_name=company_name,
            website_url=website_url,
            yc_url=yc_url,
            one_liner=one_liner,
        )

        # Step 2: Score with LLM
        card = await self.chain.ainvoke({
            "evidence_context": evidence.to_context_string(),
            "format_instructions": self.parser.get_format_instructions(),
        })

        logger.info(
            f"Product signals for {company_name}: "
            f"maturity={card.product_maturity_score} "
            f"understanding={card.user_understanding_score} "
            f"live={card.has_live_product} "
            f"paying={card.has_paying_customers}"
        )
        return card

    async def enrich_batch(
        self,
        companies: list[dict],
        max_concurrent: int = 3,
    ) -> list[tuple[str, ProductSignalCard | dict]]:
        """
        Enrich a batch of companies.

        Args:
            companies: list of dicts with keys: name, website, yc_url, one_liner
            max_concurrent: parallel enrichment limit (keep low — lots of HTTP per company)

        Returns:
            list of (company_name, ProductSignalCard) tuples, sorted by maturity score
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _enrich_one(co: dict) -> tuple[str, ProductSignalCard | dict]:
            async with sem:
                try:
                    card = await self.enrich(
                        company_name=co["name"],
                        website_url=co.get("website", ""),
                        yc_url=co.get("yc_url", ""),
                        one_liner=co.get("one_liner", ""),
                    )
                    return (co["name"], card)
                except Exception as e:
                    logger.error(f"Product enrichment failed for {co['name']}: {e}")
                    return (co["name"], {"error": str(e)})

        results = await asyncio.gather(*[_enrich_one(co) for co in companies])

        # Sort by product maturity descending
        def sort_key(item):
            name, card = item
            if isinstance(card, dict):
                return -1
            return card.product_maturity_score

        return sorted(results, key=sort_key, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INTEGRATION HELPER — merge into main SignalCard
# ═══════════════════════════════════════════════════════════════════════════════

def merge_product_signals(main_row: dict, product_card: ProductSignalCard) -> dict:
    """
    Merge product signals into a main SignalCard row.
    Call this after you have both the team signals and product signals.

    Usage:
        team_card = await pipeline.screen_fast(url)
        product_card = await enricher.enrich(name, website)
        merged = merge_product_signals(team_card.to_row(), product_card)
    """
    merged = {**main_row}
    for k, v in product_card.to_row().items():
        merged[f"product_{k}"] if k.startswith("has_") or k.startswith("product_") else None
        merged[k] = v
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_product_card(card: ProductSignalCard, company_name: str = "") -> str:
    def f(b: bool) -> str:
        return "✓" if b else "✗"

    return f"""
{'═' * 60}
  {company_name}  — Product Signals
{'═' * 60}

  ┌── Product Maturity ───────────────────────────────
  │  Live product              {f(card.has_live_product)}
  │  Paying customers          {f(card.has_paying_customers)}
  │  External user evidence    {f(card.has_user_evidence)}
  │  Public iteration          {f(card.has_public_iteration)}
  │  Maturity score            {card.product_maturity_score}/10
  │
  ├── User Understanding ─────────────────────────────
  │  Specific user problem     {f(card.has_specific_user_problem)}
  │  Beachhead segment         {f(card.has_beachhead_segment)}
  │  Understanding score       {card.user_understanding_score}/10
  │
  ├── Evidence ───────────────────────────────────────
  │  Sources with data         {card.sources_with_evidence}
  │  Strongest evidence        {card.strongest_evidence}
  │  Biggest gap               {card.biggest_product_gap}
  └───────────────────────────────────────────────────
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python product_enricher.py <company_name> <website_url> [yc_url]")
        print("Example: python product_enricher.py Cardinal https://trycardinal.ai")
        return

    name = sys.argv[1]
    website = sys.argv[2]
    yc_url = sys.argv[3] if len(sys.argv) > 3 else ""

    enricher = ProductEnricher()
    card = await enricher.enrich(name, website, yc_url)
    print(format_product_card(card, name))


if __name__ == "__main__":
    asyncio.run(main())
