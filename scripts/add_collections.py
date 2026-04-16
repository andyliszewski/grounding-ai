#!/usr/bin/env python3
"""
Add collection tags to documents for agent corpus filtering.

This script updates meta.yaml files and _index.json to add collection tags
that enable agents to filter their corpus appropriately.
"""

import json
import yaml
from pathlib import Path

CORPUS_DIR = Path("./corpus")

# Document slug -> collections to ADD (won't remove existing)
COLLECTION_MAPPINGS = {
    # =========================================================================
    # IP AGENT: patent-law, mpep, compliance, intellectual-property
    # =========================================================================

    # MPEP chapters
    "mpep-0000-table-of-contents": ["mpep", "patent-law"],
    "mpep-0005-change-summary": ["mpep", "patent-law"],
    "mpep-0010-title-page": ["mpep", "patent-law"],
    "mpep-0015-foreword": ["mpep", "patent-law"],
    "mpep-0020-introduction": ["mpep", "patent-law"],
    "mpep-0100": ["mpep", "patent-law"],
    "mpep-0200": ["mpep", "patent-law"],
    "mpep-0300": ["mpep", "patent-law"],
    "mpep-0400": ["mpep", "patent-law"],
    "mpep-0500": ["mpep", "patent-law"],
    "mpep-0600": ["mpep", "patent-law"],
    "mpep-0700": ["mpep", "patent-law"],
    "mpep-0800": ["mpep", "patent-law"],
    "mpep-0900": ["mpep", "patent-law"],
    "mpep-1000": ["mpep", "patent-law"],
    "mpep-1100": ["mpep", "patent-law"],
    "mpep-1200": ["mpep", "patent-law"],
    "mpep-1300": ["mpep", "patent-law"],
    "mpep-1400": ["mpep", "patent-law"],
    "mpep-1500": ["mpep", "patent-law"],
    "mpep-1600": ["mpep", "patent-law"],
    "mpep-1700": ["mpep", "patent-law"],
    "mpep-1800": ["mpep", "patent-law"],
    "mpep-1900": ["mpep", "patent-law"],
    "mpep-2000": ["mpep", "patent-law"],
    "mpep-2100": ["mpep", "patent-law"],
    "mpep-2200": ["mpep", "patent-law"],
    "mpep-2300": ["mpep", "patent-law"],
    "mpep-2400": ["mpep", "patent-law"],
    "mpep-2500": ["mpep", "patent-law"],
    "mpep-2600": ["mpep", "patent-law"],
    "mpep-2700": ["mpep", "patent-law"],
    "mpep-2800": ["mpep", "patent-law"],
    "mpep-2900": ["mpep", "patent-law"],
    "mpep-9005-appx-i": ["mpep", "patent-law"],
    "mpep-9010-appx-ii": ["mpep", "patent-law"],
    "mpep-9015-appx-l": ["mpep", "patent-law"],
    "mpep-9020-appx-r": ["mpep", "patent-law"],
    "mpep-9025-appx-t": ["mpep", "patent-law"],
    "mpep-9030-appx-ai": ["mpep", "patent-law"],
    "mpep-9035-appx-p": ["mpep", "patent-law"],
    "mpep-9090-subject-matter-index": ["mpep", "patent-law"],
    "mpep-9095-form-paragraph-chapter": ["mpep", "patent-law"],

    # Patent statutes and regulations
    "uscode-2011-title35": ["patent-law"],
    "cfr-2020-title37-vol1": ["patent-law"],

    # Patent case law
    "alice-corp-v-cls-bank-intl": ["patent-law"],
    "mayo-collab-v-prometheus-lab": ["patent-law"],
    "markman-et-al-v-westview-instruments-inc-et-al-517-us-370-1996": ["patent-law"],
    "herbert-markman-and-positek-inc-petitioners-v-westview-instruments-inc-and-althon-enterprises-inc-supreme-court-us-law-lii-legal-information-institute": ["patent-law"],

    # Patent claim drafting and construction
    "claim-construction-and-markman": ["patent-law"],
    "claim-drafting": ["patent-law"],
    "3-nyc-01-clinic1-patentclaimwriting": ["patent-law"],
    "patent-claim-construction-a-modern-synthesis-and-structured-framework": ["patent-law"],
    "wipo-pub-867-23-en-wipo-patent-drafting-manual": ["patent-law"],
    "lemley-claim": ["patent-law"],

    # Section 101 / Patent eligibility
    "2106-patent-subject-matter-eligibility": ["patent-law"],
    "peg-oct-2019-update": ["patent-law"],
    "checking-in-with-alice-section-101-developments-at-the-federal-circuit-district-courts-uspto-and-congress": ["patent-law"],
    "section-101-examples-on-subject-matter-eligibility-from-the-uspto-bitlaw": ["patent-law"],
    "how-to-respond-to-and-overcome-a-section-101-rejection-bitlaw-guidance": ["patent-law"],
    "top-section-101-patent-eligibility-stories-of-2024": ["patent-law"],
    "101-examples-37to42-20190107": ["patent-law"],
    "4-patent-law-101-i-know-it-when-i-see-it": ["patent-law"],

    # Federal Circuit / USPTO updates
    "federal-circuit-clarifies-limits-of-patent-eligibility-for-machine-learning-claims": ["patent-law"],
    "federal-circuit-rules-on-101-eligibility-of-ai-machine-learning-patents-omelveny": ["patent-law"],
    "federal-circuit-update-september-2024": ["patent-law"],
    "2024-federal-circuit-ip-appeals-patent-eligibility-in-2024-a-sextet-of-precedential-decisions-sterne-kessler": ["patent-law"],
    "guidelines-for-computer-related-inventions": ["patent-law"],
    "memo-berkheimer-20180419": ["patent-law"],
    "trial-practice-guide-uspto": ["patent-law"],
    "uspto-patent-prosecution-highway-guidelines": ["patent-law"],

    # IP general
    "charmasson-patents-copyrights-trademarks-for-dummies": ["patent-law", "intellectual-property"],
    "22-berkeley-tech-l-j-1389-1402": ["patent-law"],

    # Compliance (IP agent also covers compliance)
    "pci-dss-v4-0-1": ["compliance"],
    "gdpr-financial-guidelines-en": ["compliance"],

    # =========================================================================
    # HARRY SELDON AGENT: game-theory, economics, history, systems-thinking, influence
    # =========================================================================

    # Game theory
    "a-course-in-game-theory": ["game-theory", "economics"],
    "art-of-strategy": ["game-theory"],

    # Economics - macro/micro
    "book-n-gregory-mankiw-principles-of-economics-dr-jwan": ["economics"],
    "microeconomic-theory-oxford-university-press-1995": ["economics", "microeconomics"],
    "advanced-macroeconomics-mcgraw-hill-education-2018": ["economics", "macroeconomics"],
    "managerial-economics-8th": ["economics"],
    "brealey-myers-principles-of-corporate-finance-7e": ["economics"],
    "the-intelligent-investor-benjamin-graham": ["economics"],
    "valuation": ["economics"],

    # History
    "british-history-for-dummies": ["history"],
    "daliochangingworldordercharts": ["history", "geopolitics"],

    # Systems thinking
    "limits-to-growth": ["systems-thinking"],
    "springerbriefs-in-energy-ugo-bardi-auth-the-limits-to-growth-revisited-2011-springer-101007-978-1-4419-9416-5-libgenli": ["systems-thinking"],
    "meadows-2008-thinking-in-systems": ["systems-thinking"],
    "thefifthdiscipline": ["systems-thinking"],
    "the-model-thinker-pdf": ["systems-thinking"],

    # Influence and social dynamics
    "robert-cialdini-influence-science-and-practice-4th-ed1": ["influence", "social-dynamics"],
    "a-robert-greene-the-48-laws-of-power": ["influence", "social-dynamics"],
    "the48lawsofpower": ["influence", "social-dynamics"],
    "social-engineering-hacking-systems-nations-and-societies": ["influence", "social-dynamics"],

    # Behavioral economics / Decision making
    "daniel-kahneman-thinking-fast-and-slow": ["economics", "social-dynamics"],
    "daniel-kahneman-thinking-fast-and-slow-5117dd17": ["economics", "social-dynamics"],
    "kahneman-aer-2003": ["economics"],
    "jones-bounded-rationality-1999": ["economics"],
    "richard-h-thaler-cass-r-sunstein-nudge-improv-35db5867": ["economics", "influence"],
    "misbehaving-pdf-248efeb8": ["economics"],
    "thinking-in-bets-pdf": ["game-theory"],
    "antifragile-pdf": ["systems-thinking"],

    # Strategy
    "strategic-thinking-in-complex-problem-solving": ["game-theory"],
    "strategic-management": ["game-theory"],
    "good-strategy-bad-strategy-pdf": ["game-theory"],
    "good-strategy-bad-strategy-pdf-a3ab465e": ["game-theory"],

    # Philosophy (Seldon draws on historical/philosophical thought)
    "plato-the-republic": ["history"],
    "ernest-becker-the-denial-of-death": ["social-dynamics"],
    "meditationsofmar00marc": ["history"],

    # =========================================================================
    # MARKETING DIRECTOR AGENT: marketing, growth, product-market-fit
    # =========================================================================

    # Growth
    "hacking-growth-how-todays-fastest-growing-companies-drive-breakout-success-pdf-room": ["marketing", "growth"],
    "product-led-growth-pdf": ["marketing", "growth"],
    "lean-analytics-pdf": ["marketing", "growth"],

    # Product-market fit
    "crossing-the-chasm-3rd-edition": ["marketing", "product-market-fit"],
    "crossing-the-chasm-3rd-edition-moore": ["marketing", "product-market-fit"],
    "obviously-awesome-pdf": ["marketing", "product-market-fit"],
    "the-innovators-dilemma-clayton-m-christensen2000": ["marketing", "product-market-fit"],

    # Brand and messaging
    "pdfcoffeecom-building-a-storybrand-pdf-free": ["marketing"],
    "monetizing-innovation-pdf": ["marketing"],

    # Behavioral design (marketing applications)
    "hooked-how-to-build-habit-forming-products-nir-eyal": ["marketing", "growth"],
    "deceptive-patterns-sample-chapters-27-july-2023": ["marketing"],
    "the-design-of-everyday-things-by-don-norman": ["marketing"],
    "the-art-of-choosing": ["marketing"],
    "payoff-the-hidden-logic-that-shapes-our-motivations-pdf-room": ["marketing"],

    # Analytics (marketing measurement)
    "croll-yoskovitz-lean-analytics": ["marketing", "growth"],
    "wendel-designing-for-behavior-change": ["marketing"],

    # =========================================================================
    # SCIENTIST AGENT: science, biology, chemistry, logic, engineering
    # =========================================================================

    # Biology
    "ap-biology-for-dummies": ["science", "biology"],
    "biochemistry-for-dummies": ["science", "biology", "chemistry"],

    # Chemistry
    "ap-chemistry-for-dummies-p-mikulecky-et-al-wiley-2009-ww": ["science", "chemistry"],

    # Logic and critical thinking
    "a-a-a-a-concise-introduction-to-logic-hurley-7e": ["logic", "critical-thinking"],

    # Engineering
    "electronics-for-dummies": ["engineering", "science"],
    "the-art-of-modeling-in-science-and-engineering-crc-press-1999": ["engineering", "science"],

    # Scientific methodology
    "how-people-learn": ["science"],

    # =========================================================================
    # VENTURE ATTORNEY AGENT: venture-law (add to docs not yet tagged)
    # =========================================================================

    # Corporate finance (relevant to VC)
    "valuation": ["venture-law"],
    "brealey-myers-principles-of-corporate-finance-7e": ["venture-law"],
    "the-intelligent-investor-benjamin-graham": ["venture-law"],
    "the-outsiders-william-n-thorndike-jr": ["venture-law"],

    # Founders/startups
    "founders": ["venture-law"],
    "2477the-hard-thing-about-hard-things": ["venture-law"],
    "amp-it-up-pdf": ["venture-law"],

    # M&A (venture attorney does exit strategies)
    "art-of-m-and-a-book": ["venture-law"],
}


def update_meta_yaml(slug: str, new_collections: list[str]) -> bool:
    """Update a document's meta.yaml with new collections."""
    meta_path = CORPUS_DIR / slug / "meta.yaml"
    if not meta_path.exists():
        print(f"  SKIP: {slug} - meta.yaml not found")
        return False

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    existing = set(meta.get("collections", []))
    to_add = set(new_collections) - existing

    if not to_add:
        print(f"  SKIP: {slug} - already has {new_collections}")
        return False

    # Merge collections
    meta["collections"] = sorted(existing | set(new_collections))

    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    print(f"  UPDATED: {slug} + {sorted(to_add)}")
    return True


def update_manifest(updated_slugs: set[str]):
    """Update _index.json to reflect meta.yaml changes."""
    manifest_path = CORPUS_DIR / "_index.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    for doc in manifest.get("docs", []):
        slug = doc.get("slug")
        if slug in updated_slugs:
            meta_path = CORPUS_DIR / slug / "meta.yaml"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = yaml.safe_load(f)
                doc["collections"] = meta.get("collections", [])

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nUpdated _index.json with {len(updated_slugs)} document changes")


def main():
    print("Adding collection tags to documents...\n")

    updated = set()
    for slug, collections in sorted(COLLECTION_MAPPINGS.items()):
        if update_meta_yaml(slug, collections):
            updated.add(slug)

    if updated:
        update_manifest(updated)
        print(f"\nTotal documents updated: {len(updated)}")
    else:
        print("\nNo updates needed")

    # Summary by agent
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)

    collection_counts = {}
    for slug, collections in COLLECTION_MAPPINGS.items():
        for col in collections:
            collection_counts[col] = collection_counts.get(col, 0) + 1

    for col, count in sorted(collection_counts.items()):
        print(f"  {col}: {count} documents")


if __name__ == "__main__":
    main()
