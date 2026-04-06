"""
Defensibility Signal Enrichment Module
========================================
Gathers evidence of competitive moats and defensibility risks from multiple
web sources, then synthesizes into filterable boolean signals.

Corinne Riley's defensibility pillar: "When people come after you, what keeps
you as number one?" She doesn't come in with a predetermined answer — she wants
the founder's hypothesis. Our job is to arm her with the competitive landscape
so she can evaluate that hypothesis.

What this module detects:
  1. Competitive density — how many similar companies exist (batch + market)
  2. Incumbent presence — are big companies already here
  3. OSS exposure — is there a popular open-source alternative
  4. Patent/IP signals — has the company filed any patents
  5. Moat type classification — what kind of moat does the description claim

What this module does NOT do:
  - Score whether the moat is "good" (that's Corinne's job)
  - Predict defensibility outcomes
  - Replace the founder conversation about moat hypothesis

Architecture:
  Same as product_enricher: gather evidence → LLM synthesis → filterable card

Usage:
    enricher = DefensibilityEnricher()
    card = await enricher.enrich(
        company_name="Cardinal",
        one_liner="AI Platform for Precision Outbound",
        website="https://trycardinal.ai",
        batch_peers=[{"name": "OtherCo", "one_liner": "AI sales outreach"}, ...]
    )
    print(card.competitor_count_batch)
    print(card.moat_type)

Requires:
    pip install langchain langchain-anthropic httpx beautifulsoup4 pydantic

    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import asyncio
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

logger = logging.getLogger("defensibility_enricher")

DEFAULT_MODEL = "claude-sonnet-4-20250514"
REQUEST_TIMEOUT = 15.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. EVIDENCE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceResult:
    source: str
    url: str
    found: bool
    signals: dict = field(default_factory=dict)
    raw_text: str = ""
    error: str = ""


@dataclass
class DefensibilityEvidence:
    company_name: str
    one_liner: str
    website: str
    batch_peer_analysis: str = ""
    results: list[SourceResult] = field(default_factory=list)

    def to_context_string(self) -> str:
        lines = [
            f"Company: {self.company_name}",
            f"One-liner: {self.one_liner}",
            f"Website: {self.website}",
            "",
        ]
        if self.batch_peer_analysis:
            lines.append("=== BATCH PEER ANALYSIS ===")
            lines.append(self.batch_peer_analysis)
            lines.append("")

        lines.append("=== WEB EVIDENCE ===")
        lines.append("")
        for r in self.results:
            status = "FOUND" if r.found else "NOT FOUND"
            lines.append(f"--- {r.source.upper()} [{status}] ---")
            if r.error:
                lines.append(f"  Error: {r.error}")
            if r.signals:
                for k, v in r.signals.items():
                    lines.append(f"  {k}: {v}")
            if r.raw_text:
                lines.append(f"  Context: {r.raw_text[:600]}")
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DEFENSIBILITY SIGNAL CARD
# ═══════════════════════════════════════════════════════════════════════════════

class DefensibilitySignalCard(BaseModel):
    """Filterable defensibility signals for one company."""

    # ── Competitive landscape ──
    competitor_count_batch: int = Field(
        description="Number of companies in the same YC batch building a substantially "
                    "similar product (0 = unique in batch, 3+ = crowded)."
    )
    batch_competitors: str = Field(
        description="Comma-separated names of batch peers in the same space. "
                    "Empty string if none."
    )
    has_funded_incumbents: bool = Field(
        description="TRUE if there are well-funded startups (Series B+) or public companies "
                    "already operating in this exact space. FALSE if greenfield."
    )
    incumbent_names: str = Field(
        description="Names of key incumbents found, comma-separated. Empty if none."
    )
    market_crowding: str = Field(
        description="One of: GREENFIELD (no direct competitors found), "
                    "EMERGING (a few early-stage competitors), "
                    "COMPETITIVE (multiple funded players), "
                    "CROWDED (many competitors including incumbents)."
    )

    # ── Technical defensibility ──
    has_patent_signal: bool = Field(
        description="TRUE if any patent filings were found associated with the company "
                    "or founders. FALSE if none found (absence of evidence, not evidence of absence)."
    )
    has_oss_alternative: bool = Field(
        description="TRUE if there is a well-known open-source project (1000+ GitHub stars) "
                    "that solves substantially the same problem. This is a risk signal."
    )
    oss_alternative_name: str = Field(
        description="Name and star count of the most relevant OSS alternative. "
                    "Empty string if none found."
    )

    # ── Moat classification ──
    moat_type: str = Field(
        description="Based on the company description and evidence, classify the likely moat "
                    "as one of: DATA_MOAT (proprietary data or data flywheel), "
                    "NETWORK_EFFECT (value increases with users), "
                    "SWITCHING_COST (hard to leave once adopted), "
                    "REGULATORY (licenses, compliance, certifications required), "
                    "TECHNICAL_IP (patents, proprietary algorithms, hardware), "
                    "BRAND_TRUST (reputation in regulated/high-stakes domain), "
                    "NONE_OBVIOUS (no clear moat from available evidence). "
                    "Pick the single strongest. If genuinely unclear, use NONE_OBVIOUS."
    )
    moat_evidence: str = Field(
        description="One sentence: what specific evidence supports this moat classification?"
    )

    # ── Context for Corinne ──
    competitive_context_brief: str = Field(
        description="2-3 sentences summarizing the competitive landscape. Written for a VC "
                    "partner to scan in 10 seconds before a meeting. Focus on: how crowded "
                    "is this space, who are the biggest threats, and what would the founder "
                    "need to argue to convince you their moat is real."
    )
    key_question_for_meeting: str = Field(
        description="The single best question to ask this founder about defensibility, "
                    "based on the evidence gathered. Under 20 words."
    )

    def to_row(self) -> dict:
        return self.model_dump()

    @staticmethod
    def csv_columns() -> list[str]:
        return [
            "competitor_count_batch", "batch_competitors",
            "has_funded_incumbents", "incumbent_names", "market_crowding",
            "has_patent_signal", "has_oss_alternative", "oss_alternative_name",
            "moat_type", "moat_evidence",
            "competitive_context_brief", "key_question_for_meeting",
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BATCH PEER ANALYSIS — in-batch competitive density (no API needed)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_batch_peers(
    target_name: str,
    target_one_liner: str,
    all_companies: list[dict],
) -> str:
    """
    Build a context string listing all companies in the batch for the LLM
    to identify which ones are in the same space as the target.

    This is cheap — no API calls, just formatting your existing CSV data.
    The LLM does the semantic matching.
    """
    peers = []
    for co in all_companies:
        name = co.get("name") or co.get("slug", "")
        liner = co.get("one_liner") or co.get("description", "")
        industry = co.get("industry", "")
        subindustry = co.get("subindustry", "")

        if name.lower() == target_name.lower():
            continue

        peers.append(f"  - {name}: {liner} [{industry} / {subindustry}]")

    header = (
        f"TARGET COMPANY: {target_name}\n"
        f"TARGET ONE-LINER: {target_one_liner}\n"
        f"\n"
        f"ALL OTHER COMPANIES IN BATCH ({len(peers)} total):\n"
    )
    # Include all — the LLM will filter to relevant ones
    return header + "\n".join(peers)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SOURCE SCRAPERS
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


# ── 4a. GitHub OSS alternatives ──────────────────────────────────────────────

async def search_oss_alternatives(
    client: httpx.AsyncClient,
    one_liner: str,
    github_token: str = "",
) -> SourceResult:
    """
    Search GitHub for open-source projects that solve the same problem.
    Uses the company's one-liner as a semantic search query.
    """
    result = SourceResult(source="github_oss_alternatives", url="", found=False)

    # Extract key terms from the one-liner (strip filler words)
    query = re.sub(r'\b(the|a|an|for|and|or|of|in|to|with|that|is|are)\b', '', one_liner.lower())
    query = " ".join(query.split()[:6])  # first 6 meaningful words

    if not query.strip():
        result.error = "No meaningful search terms from one-liner"
        return result

    search_url = f"https://api.github.com/search/repositories?q={query}&sort=stars&per_page=5"
    result.url = search_url

    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    try:
        resp = await client.get(search_url, headers=headers, follow_redirects=True)
        if resp.status_code == 403:
            result.error = "GitHub rate limit — set GITHUB_TOKEN for higher limits"
            return result
        resp.raise_for_status()
    except Exception as e:
        result.error = f"GitHub search failed: {e}"
        return result

    try:
        data = resp.json()
        items = data.get("items", [])

        # Filter to repos with meaningful traction
        notable = [i for i in items if i.get("stargazers_count", 0) >= 500]

        if notable:
            top = notable[0]
            result.found = True
            result.signals = {
                "top_oss_repo": top.get("full_name", ""),
                "top_oss_stars": top.get("stargazers_count", 0),
                "top_oss_description": top.get("description", "")[:200],
                "top_oss_language": top.get("language", ""),
                "notable_oss_count": len(notable),
            }
            repos_summary = "; ".join(
                f"{r['full_name']} ({r['stargazers_count']} stars)"
                for r in notable[:3]
            )
            result.raw_text = f"OSS alternatives found: {repos_summary}"
        else:
            result.signals["notable_oss_count"] = 0

    except (json.JSONDecodeError, KeyError) as e:
        result.error = f"Failed to parse GitHub response: {e}"

    return result


# ── 4b. Patent search (USPTO) ────────────────────────────────────────────────

async def search_patents(
    client: httpx.AsyncClient,
    company_name: str,
    founder_names: list[str] | None = None,
) -> SourceResult:
    """Search USPTO PatentsView API for patent filings by company or founders."""
    result = SourceResult(source="patents_uspto", url="", found=False)

    # Search by company name via PatentsView API
    query = company_name.replace(" ", "%20")
    search_url = (
        f"https://api.patentsview.org/patents/query?"
        f"q={{\"_text_any\":{{\"patent_title\":\"{query}\"}}}}"
        f"&f=[\"patent_number\",\"patent_title\",\"patent_date\",\"assignee_organization\"]"
        f"&o={{\"per_page\":5}}"
    )
    result.url = search_url

    resp = await _fetch(client, search_url)
    if not resp:
        # Try alternate: search by assignee
        alt_url = (
            f"https://api.patentsview.org/patents/query?"
            f"q={{\"_contains\":{{\"assignee_organization\":\"{query}\"}}}}"
            f"&f=[\"patent_number\",\"patent_title\",\"patent_date\"]"
            f"&o={{\"per_page\":5}}"
        )
        resp = await _fetch(client, alt_url)

    if not resp:
        result.error = "USPTO search failed"
        return result

    try:
        data = resp.json()
        patents = data.get("patents", [])
        if patents:
            result.found = True
            result.signals = {
                "patent_count": len(patents),
                "latest_patent": patents[0].get("patent_title", ""),
                "latest_patent_date": patents[0].get("patent_date", ""),
            }
            result.raw_text = f"Found {len(patents)} patent(s). Latest: {patents[0].get('patent_title', '')}"
        else:
            result.signals["patent_count"] = 0
    except (json.JSONDecodeError, KeyError):
        result.error = "Failed to parse USPTO response"

    return result


# ── 4c. Funded competitor search (Hacker News + general web) ─────────────────

async def search_funded_competitors(
    client: httpx.AsyncClient,
    one_liner: str,
) -> SourceResult:
    """
    Search HN for funded companies in the same space.
    Uses the one-liner keywords to find relevant discussions.
    """
    result = SourceResult(source="funded_competitors_hn", url="", found=False)

    # Build search query from one-liner
    keywords = re.sub(r'\b(the|a|an|for|and|or|of|in|to|with|that|is|are|ai|platform)\b', '', one_liner.lower())
    terms = [w for w in keywords.split() if len(w) > 3][:4]
    query = " ".join(terms)

    if not query.strip():
        result.error = "No meaningful search terms"
        return result

    search_url = f"https://hn.algolia.com/api/v1/search?query={query} funding raised&tags=story&hitsPerPage=10"
    result.url = search_url

    resp = await _fetch(client, search_url)
    if not resp:
        result.error = "HN search failed"
        return result

    try:
        data = resp.json()
        hits = data.get("hits", [])

        # Filter to hits mentioning funding/raises
        relevant = [
            h for h in hits
            if any(kw in (h.get("title") or "").lower()
                   for kw in ["raise", "fund", "series", "seed", "million", "billion", "launch", "yc"])
        ]

        if relevant:
            result.found = True
            titles = [h.get("title", "")[:100] for h in relevant[:5]]
            result.signals = {
                "funded_competitor_mentions": len(relevant),
                "top_mentions": titles,
            }
            result.raw_text = "Funded competitor signals: " + " | ".join(titles[:3])
        else:
            result.signals["funded_competitor_mentions"] = 0

    except (json.JSONDecodeError, KeyError):
        result.error = "Failed to parse HN response"

    return result


# ── 4d. Company website moat signals ─────────────────────────────────────────

async def scrape_website_moat_signals(
    client: httpx.AsyncClient,
    website_url: str,
) -> SourceResult:
    """Check company website for defensibility language."""
    result = SourceResult(source="website_moat_signals", url=website_url, found=False)

    if not website_url:
        result.error = "No website URL"
        return result

    resp = await _fetch(client, website_url)
    if not resp:
        result.error = "Could not fetch website"
        return result

    text = resp.text.lower()
    visible = BeautifulSoup(resp.text, "html.parser").get_text(separator=" ", strip=True)[:3000]

    signals = {}

    # Data moat language
    signals["mentions_proprietary_data"] = bool(re.search(
        r'(proprietary\s*data|unique\s*data|exclusive\s*data|our\s*data\s*set|training\s*data|data\s*moat|data\s*flywheel)',
        text
    ))

    # Network effect language
    signals["mentions_network_effect"] = bool(re.search(
        r'(network\s*effect|community|marketplace|two.sided|platform\s*effect|more\s*users)',
        text
    ))

    # Regulatory / compliance
    signals["mentions_regulatory"] = bool(re.search(
        r'(hipaa|soc\s*2|gdpr|fda|finra|sec\s*complian|licensed|certified|regulatory|compliance)',
        text
    ))

    # Patents / IP
    signals["mentions_patents"] = bool(re.search(
        r'(patent|intellectual\s*property|trade\s*secret|proprietary\s*algorithm|novel\s*approach)',
        text
    ))

    # Integration / switching cost
    signals["mentions_integrations"] = bool(re.search(
        r'(integrat|connect\s*with|works\s*with|plugin|api\s*access|embed)',
        text
    ))

    # Enterprise / compliance features (implies switching cost)
    signals["mentions_enterprise"] = bool(re.search(
        r'(enterprise|sso|saml|audit\s*log|role.based|permission|admin\s*console)',
        text
    ))

    if any(signals.values()):
        result.found = True

    result.signals = signals
    result.raw_text = visible[:500]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVIDENCE GATHERER
# ═══════════════════════════════════════════════════════════════════════════════

async def gather_defensibility_evidence(
    company_name: str,
    one_liner: str,
    website: str = "",
    batch_companies: list[dict] | None = None,
    github_token: str = "",
) -> DefensibilityEvidence:
    """Run all defensibility scrapers concurrently."""

    evidence = DefensibilityEvidence(
        company_name=company_name,
        one_liner=one_liner,
        website=website,
    )

    # Batch peer analysis (no API needed)
    if batch_companies:
        evidence.batch_peer_analysis = analyze_batch_peers(
            company_name, one_liner, batch_companies
        )

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, headers=headers) as client:
        tasks = [
            search_oss_alternatives(client, one_liner, github_token),
            search_patents(client, company_name),
            search_funded_competitors(client, one_liner),
            scrape_website_moat_signals(client, website),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, SourceResult):
                evidence.results.append(r)
            elif isinstance(r, Exception):
                evidence.results.append(SourceResult(
                    source="unknown", url="", found=False, error=str(r)
                ))

    found = sum(1 for r in evidence.results if r.found)
    logger.info(f"Defensibility evidence for {company_name}: {found}/{len(evidence.results)} sources with data")
    return evidence


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LLM SCORING
# ═══════════════════════════════════════════════════════════════════════════════

DEFENSIBILITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a VC competitive analyst preparing briefings for a Greylock partner.

Given evidence about a YC company and its batch peers, produce a defensibility signal card.

RULES:
- competitor_count_batch: count companies in the BATCH PEERS list that are building a 
  substantially similar product. "Similar" means a customer choosing between them, not 
  just same industry. Two AI companies in different verticals are NOT competitors.
- has_funded_incumbents: TRUE only if specific funded companies or public companies are 
  identified that directly compete. General "the market is competitive" does not count.
- has_oss_alternative: TRUE only if a specific open-source repo with 1000+ stars is found 
  that solves the same core problem. A vaguely related repo does not count.
- moat_type: classify based on the STRONGEST evidence. If the website mentions proprietary 
  data AND regulatory compliance, pick the one that seems more central to the business.
  Use NONE_OBVIOUS only when there truly is no signal.
- key_question_for_meeting: make this specific and useful. Not generic "what's your moat?" 
  but something like "With 3 other AI coding tools in this batch, what makes your approach 
  technically differentiated enough that developers won't switch when a competitor ships a 
  similar feature?"

The competitive_context_brief should be written for a partner scanning it in 10 seconds 
before walking into a meeting. Be direct, factual, no fluff.

{format_instructions}"""),
    ("human", """{evidence_context}"""),
])


def build_defensibility_chain(model_name: str = DEFAULT_MODEL):
    llm = ChatAnthropic(model=model_name, temperature=0.1, max_tokens=2048)
    parser = PydanticOutputParser(pydantic_object=DefensibilitySignalCard)
    chain = DEFENSIBILITY_PROMPT | llm | parser
    return chain, parser


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ENRICHER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DefensibilityEnricher:
    """
    Full pipeline: gather evidence → LLM synthesis → DefensibilitySignalCard.

    Usage:
        enricher = DefensibilityEnricher()
        card = await enricher.enrich(
            company_name="Cardinal",
            one_liner="AI Platform for Precision Outbound",
            website="https://trycardinal.ai",
            batch_companies=[...],   # your full CSV as list of dicts
        )
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, github_token: str = ""):
        self.model_name = model_name
        self.github_token = github_token
        self.chain, self.parser = build_defensibility_chain(model_name)

    async def enrich(
        self,
        company_name: str,
        one_liner: str,
        website: str = "",
        batch_companies: list[dict] | None = None,
    ) -> DefensibilitySignalCard:

        evidence = await gather_defensibility_evidence(
            company_name=company_name,
            one_liner=one_liner,
            website=website,
            batch_companies=batch_companies,
            github_token=self.github_token,
        )

        card = await self.chain.ainvoke({
            "evidence_context": evidence.to_context_string(),
            "format_instructions": self.parser.get_format_instructions(),
        })

        logger.info(
            f"Defensibility signals for {company_name}: "
            f"crowding={card.market_crowding} moat={card.moat_type} "
            f"batch_competitors={card.competitor_count_batch} "
            f"incumbents={card.has_funded_incumbents}"
        )
        return card

    async def enrich_batch(
        self,
        companies: list[dict],
        max_concurrent: int = 3,
    ) -> list[tuple[str, DefensibilitySignalCard | dict]]:
        """
        Enrich all companies in a batch. Passes the full company list to each
        enrichment call so the LLM can identify intra-batch competitors.
        """
        sem = asyncio.Semaphore(max_concurrent)
        results = []

        async def _enrich_one(co: dict):
            name = co.get("name") or co.get("slug", "")
            liner = co.get("one_liner") or co.get("description", "")
            website = co.get("website", "")

            async with sem:
                try:
                    card = await self.enrich(
                        company_name=name,
                        one_liner=liner,
                        website=website,
                        batch_companies=companies,
                    )
                    return (name, card)
                except Exception as e:
                    logger.error(f"Defensibility enrichment failed for {name}: {e}")
                    return (name, {"error": str(e)})

        tasks = [_enrich_one(co) for co in companies]
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda x: x[1].competitor_count_batch if isinstance(x[1], DefensibilitySignalCard) else -1, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════

def format_defensibility_card(card: DefensibilitySignalCard, company_name: str = "") -> str:
    def f(b: bool) -> str:
        return "✓" if b else "✗"

    return f"""
{'═' * 60}
  {company_name}  — Defensibility Signals
{'═' * 60}

  ┌── Competitive Landscape ──────────────────────────
  │  Batch competitors         {card.competitor_count_batch} ({card.batch_competitors or 'none'})
  │  Funded incumbents         {f(card.has_funded_incumbents)} {('(' + card.incumbent_names + ')') if card.incumbent_names else ''}
  │  Market crowding           {card.market_crowding}
  │
  ├── Technical Defensibility ────────────────────────
  │  Patent signals            {f(card.has_patent_signal)}
  │  OSS alternative exists    {f(card.has_oss_alternative)} {card.oss_alternative_name or ''}
  │  Moat type                 {card.moat_type}
  │  Moat evidence             {card.moat_evidence}
  │
  ├── Context for Partner ────────────────────────────
  │  {card.competitive_context_brief}
  │
  │  Ask in meeting: "{card.key_question_for_meeting}"
  └───────────────────────────────────────────────────
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    import sys
    import os

    if len(sys.argv) < 3:
        print("Usage: python defensibility_enricher.py <company_name> <one_liner> [website] [batch_csv]")
        print('Example: python defensibility_enricher.py Cardinal "AI Platform for Precision Outbound" https://trycardinal.ai companies.csv')
        return

    name = sys.argv[1]
    liner = sys.argv[2]
    website = sys.argv[3] if len(sys.argv) > 3 else ""
    batch_csv = sys.argv[4] if len(sys.argv) > 4 else ""

    batch_companies = []
    if batch_csv:
        import csv
        with open(batch_csv, "r") as f:
            batch_companies = list(csv.DictReader(f))

    github_token = os.environ.get("GITHUB_TOKEN", "")
    enricher = DefensibilityEnricher(github_token=github_token)
    card = await enricher.enrich(name, liner, website, batch_companies or None)
    print(format_defensibility_card(card, name))


if __name__ == "__main__":
    asyncio.run(main())
