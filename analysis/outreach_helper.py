#!/usr/bin/env python3
"""
company_card.py — Look up a company's signals and generate a VC outreach metaprompt.

Usage:
    python company_card.py --company "Mendral"
    python company_card.py --company mendral          # case-insensitive
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

_SEC = "\x00SECTION"  # sentinel: pause 1s before printing next line

CSV_PATH = Path(__file__).parent.parent / "visualization" / "defensibility_enriched.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def bool_icon(val: str) -> str:
    return "✓" if str(val).strip().upper() == "TRUE" else "✗"

def score_bar(val: str, max_score: int = 10) -> str:
    try:
        n = int(float(val))
        filled = "█" * n
        empty  = "░" * (max_score - n)
        return f"{filled}{empty} {n}/{max_score}"
    except (ValueError, TypeError):
        return str(val)

def wrap(text: str, width: int = 72, indent: str = "  ") -> str:
    """Simple word-wrap."""
    words = str(text).split()
    lines, current = [], []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(indent + " ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(indent + " ".join(current))
    return "\n".join(lines)

def section(title: str, width: int = 60) -> list:
    """Returns lines for a section header, preceded by a section-break sentinel."""
    bar = "─" * width
    return [_SEC, "", bar, f"  {title}", bar]

def copy_to_clipboard(text: str) -> bool:
    try:
        proc = subprocess.run(["pbcopy"], input=text.encode(), check=True)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            subprocess.run(["xclip", "-selection", "clipboard"],
                           input=text.encode(), check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False


# ── card renderer ─────────────────────────────────────────────────────────────

def render_card(r: dict) -> list:
    """Return a flat list of printable lines, with _SEC sentinels before section headers."""
    out = []
    W = 60

    def add(text: str = ""):
        """Append, splitting on any embedded newlines into separate entries."""
        for part in str(text).split("\n"):
            out.append(part)

    # Header
    add("╔" + "═" * W + "╗")
    add("║" + f"  {r['company_name']}  ({r['batch']})".ljust(W) + "║")
    add("║" + f"  {r['sector']}".ljust(W) + "║")
    add("╚" + "═" * W + "╝")
    add()
    add(f"  {r['one_liner']}")
    add(f"  Website : {r['website']}")
    add(f"  YC URL  : {r['yc_url']}")
    add(f"  Signal  : {r.get('overall_signal', 'N/A')}")

    # ── Founder Signals ──────────────────────────────────────────────────────
    out.extend(section("FOUNDER SIGNALS"))
    add(f"  Domain relevance score : {score_bar(r['domain_relevance_score'])}")
    add(f"  Team completeness score: {score_bar(r['team_completeness_score'])}")
    add()
    add(f"  {bool_icon(r['founder_worked_in_target_industry'])}  Worked in target industry")
    add(f"  {bool_icon(r['founder_held_target_function'])}  Held target function")
    add(f"  {bool_icon(r['founder_has_prior_exit'])}  Prior exit")
    add(f"  {bool_icon(r['founder_is_repeat_yc'])}  Repeat YC founder")
    add(f"  {bool_icon(r['founder_from_top_co'])}  From top company")
    add(f"  {bool_icon(r['has_technical_cofounder'])}  Technical co-founder")
    add(f"  {bool_icon(r['has_commercial_cofounder'])}  Commercial co-founder")
    add(f"  {bool_icon(r['has_domain_expert'])}  Domain expert on team")
    add(f"  {bool_icon(r['is_solo_founder'])}  Solo founder")
    add()
    add(f"  Strength : {r['one_line_strength']}")
    add(f"  Risk     : {r['one_line_risk']}")

    # ── Product Signals ──────────────────────────────────────────────────────
    out.extend(section("PRODUCT SIGNALS"))
    add(f"  Product maturity score  : {score_bar(r['product_maturity_score'], 10)}")
    add(f"  User understanding score: {score_bar(r['user_understanding_score'], 10)}")
    add()
    add(f"  {bool_icon(r['has_live_product'])}  Live product")
    add(f"  {bool_icon(r['has_paying_customers'])}  Paying customers")
    add(f"  {bool_icon(r['has_user_evidence'])}  User evidence")
    add(f"  {bool_icon(r['has_public_iteration'])}  Public iteration")
    add(f"  {bool_icon(r['has_specific_user_problem'])}  Specific user problem identified")
    add(f"  {bool_icon(r['has_beachhead_segment'])}  Beachhead segment defined")
    add()
    add("  Strongest evidence:")
    add(wrap(r['strongest_evidence']))
    add()
    add("  Biggest product gap:")
    add(wrap(r['biggest_product_gap']))

    # ── Defensibility Signals ────────────────────────────────────────────────
    out.extend(section("DEFENSIBILITY SIGNALS"))
    add(f"  Moat type      : {r['moat_type']}")
    add(f"  Market crowding: {r['market_crowding']}")
    add(f"  {bool_icon(r['has_funded_incumbents'])}  Funded incumbents  {('→ ' + r['incumbent_names']) if str(r['has_funded_incumbents']).upper() == 'TRUE' else ''}")
    add(f"  {bool_icon(r['has_patent_signal'])}  Patent signal")
    add(f"  {bool_icon(r['has_oss_alternative'])}  OSS alternative  {('→ ' + r['oss_alternative_name']) if str(r['has_oss_alternative']).upper() == 'TRUE' else ''}")
    add()
    add("  Moat evidence:")
    add(wrap(r['moat_evidence']))
    add()
    add("  Competitive context:")
    add(wrap(r['competitive_context_brief']))
    batch_comps = r.get('batch_competitors', '')
    if batch_comps:
        add()
        add(f"  Batch competitors ({r['competitor_count_batch']}):")
        for c in str(batch_comps).split(","):
            c = c.strip()
            if c:
                add(f"    • {c}")
    add()
    add("  Key question for meeting:")
    add(wrap(r['key_question_for_meeting']))
    add()
    add("─" * W)

    return out


def print_animated(lines: list):
    """Print lines one at a time; pause 1s before section headers, 0.02s between regular lines."""
    for item in lines:
        if item == _SEC:
            time.sleep(0.1)
        else:
            print(item, flush=True)
            time.sleep(0.02)


# ── metaprompt builder ────────────────────────────────────────────────────────

def build_metaprompt(r: dict) -> str:
    def b(val: str) -> str:
        return "Yes" if str(val).strip().upper() == "TRUE" else "No"

    return f"""You are a partner at Greylock Partners, a leading venture capital firm. Write a warm, \
compelling first-touch outreach email from a Greylock partner to the founders of {r['company_name']}.

--- COMPANY CONTEXT ---
Company name    : {r['company_name']}
YC batch        : {r['batch']}
Sector          : {r['sector']}
One-liner       : {r['one_liner']}
Website         : {r['website']}
Overall signal  : {r['overall_signal']}

--- FOUNDER SIGNALS ---
Worked in target industry : {b(r['founder_worked_in_target_industry'])}
Held target function      : {b(r['founder_held_target_function'])}
Prior exit                : {b(r['founder_has_prior_exit'])}
Repeat YC founder         : {b(r['founder_is_repeat_yc'])}
From top company          : {b(r['founder_from_top_co'])}
Has technical co-founder  : {b(r['has_technical_cofounder'])}
Has commercial co-founder : {b(r['has_commercial_cofounder'])}
Has domain expert         : {b(r['has_domain_expert'])}
Solo founder              : {b(r['is_solo_founder'])}
Domain relevance score    : {r['domain_relevance_score']}/10
Team completeness score   : {r['team_completeness_score']}/10
Strength                  : {r['one_line_strength']}
Risk                      : {r['one_line_risk']}

--- PRODUCT SIGNALS ---
Live product              : {b(r['has_live_product'])}
Paying customers          : {b(r['has_paying_customers'])}
User evidence             : {b(r['has_user_evidence'])}
Public iteration          : {b(r['has_public_iteration'])}
Specific user problem     : {b(r['has_specific_user_problem'])}
Beachhead segment         : {b(r['has_beachhead_segment'])}
Product maturity score    : {r['product_maturity_score']}/10
User understanding score  : {r['user_understanding_score']}/10
Strongest evidence        : {r['strongest_evidence']}
Biggest product gap       : {r['biggest_product_gap']}

--- DEFENSIBILITY SIGNALS ---
Moat type                 : {r['moat_type']}
Moat evidence             : {r['moat_evidence']}
Market crowding           : {r['market_crowding']}
Funded incumbents         : {b(r['has_funded_incumbents'])}{(' — ' + r['incumbent_names']) if str(r['has_funded_incumbents']).upper()=='TRUE' else ''}
Patent signal             : {b(r['has_patent_signal'])}
OSS alternative           : {b(r['has_oss_alternative'])}{(' — ' + r['oss_alternative_name']) if str(r['has_oss_alternative']).upper()=='TRUE' else ''}
Batch competitors ({r['competitor_count_batch']}): {r['batch_competitors']}
Competitive context       : {r['competitive_context_brief']}
Key question for meeting  : {r['key_question_for_meeting']}

--- INSTRUCTIONS ---
Write a short (150–200 word) outreach email that:
1. Opens with a genuine, specific hook tied to the founders' background or product traction.
2. Names one compelling reason Greylock is particularly well-positioned to help (e.g., portfolio companies, network, domain expertise).
3. Acknowledges the competitive landscape without dwelling on it.
4. Closes with a single, low-friction CTA (e.g., a 20-minute call).
5. Sounds human — not like a template. Avoid boilerplate VC phrases like "excited about your vision".

Format: Subject line, then the email body. Sign off as a named partner (you can invent a plausible name).
"""


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print a company signal card and generate a VC outreach metaprompt."
    )
    parser.add_argument(
        "--company", required=True,
        help="Company name to look up (case-insensitive, partial match supported)"
    )
    args = parser.parse_args()
    query = args.company.strip().lower()

    if not CSV_PATH.exists():
        print(f"[error] CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Strip leading unnamed index column if present
    for row in rows:
        if "" in row:
            del row[""]

    matches = [r for r in rows if query in r.get("company_name", "").lower()]

    if not matches:
        names = sorted({r["company_name"] for r in rows})
        print(f"[error] No company matching '{args.company}'. Available companies:")
        for n in names:
            print(f"  • {n}")
        sys.exit(1)

    if len(matches) > 1:
        print(f"[info] Multiple matches found:")
        for m in matches:
            print(f"  • {m['company_name']}")
        print(f"\nUsing first match: {matches[0]['company_name']}\n")

    row = matches[0]

    # Print the card with animation
    print_animated(render_card(row))

    # Build & copy metaprompt
    metaprompt = build_metaprompt(row)

    # Animate the metaprompt header, then each line of the metaprompt
    time.sleep(0.15)
    meta_lines = [
        "",
        "═" * 60,
        "  METAPROMPT (paste into Claude or ChatGPT)",
        "═" * 60,
    ] + metaprompt.split("\n")
    for line in meta_lines:
        # Treat "---" section markers in the metaprompt as section breaks
        if line.startswith("---"):
            time.sleep(0.15)
        print(line, flush=True)
        time.sleep(0.02)

    copied = copy_to_clipboard(metaprompt)
    if copied:
        print("\n✓ Metaprompt copied to clipboard.")
    else:
        print("\n[note] Could not auto-copy (pbcopy/xclip not available). Copy the text above manually.")


if __name__ == "__main__":
    main()
