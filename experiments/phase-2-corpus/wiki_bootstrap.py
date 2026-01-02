#!/usr/bin/env python3
"""
Wikipedia Corpus Bootstrapper

Collects text from Wikipedia for multilingual corpus.
Streams articles, filters by quality, saves incrementally.

Usage:
    python3 wiki_bootstrap.py --lang ara --target-mb 100
    python3 wiki_bootstrap.py --lang all --target-mb 100
    python3 wiki_bootstrap.py --stats
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

WIKI_MAP = {
    "eng": "en", "fra": "fr", "deu": "de", "ara": "ar",
    "heb": "he", "jpn": "ja", "zho": "zh", "kor": "ko",
    "rus": "ru", "hin": "hi", "tha": "th", "vie": "vi",
    "fin": "fi", "tur": "tr",
}

# Quality thresholds
MIN_ARTICLE_CHARS = 500
MAX_ARTICLE_CHARS = 50000
MIN_PARAGRAPHS = 2


def estimate_tokens(text):
    """Rough token estimate: chars / 4 for Latin, chars / 2 for CJK."""
    return len(text) // 3  # average


def is_quality_article(text):
    """Basic quality filter."""
    if not text or len(text) < MIN_ARTICLE_CHARS:
        return False
    if len(text) > MAX_ARTICLE_CHARS:
        return False

    # Must have multiple paragraphs
    paragraphs = [p for p in text.split('\n\n') if len(p.strip()) > 50]
    if len(paragraphs) < MIN_PARAGRAPHS:
        return False

    # Reject lists (too many short lines)
    lines = text.split('\n')
    short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 30)
    if short_lines > len(lines) * 0.5:
        return False

    return True


def collect_wikipedia(lang, wiki_code, target_mb, output_dir):
    """Collect Wikipedia articles for one language."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = output_dir / f"{lang}.txt"
    meta_file = output_dir / f"{lang}.meta.json"

    # Resume from existing
    existing_bytes = corpus_file.stat().st_size if corpus_file.exists() else 0
    existing_mb = existing_bytes / (1024 * 1024)

    if existing_mb >= target_mb:
        print(f"{lang}: Already at {existing_mb:.1f} MB (target: {target_mb})")
        return

    print(f"{lang}: Loading Wikipedia ({wiki_code})...")
    print(f"  Existing: {existing_mb:.1f} MB, Target: {target_mb} MB")

    wiki = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{wiki_code}",
        split="train",
        streaming=True
    )

    collected = 0
    accepted = 0
    rejected = 0

    with open(corpus_file, 'a', encoding='utf-8') as f:
        for article in wiki:
            text = article.get("text", "")
            collected += 1

            if is_quality_article(text):
                # Write with document separator
                f.write(text.strip())
                f.write("\n\n<|endofdoc|>\n\n")
                accepted += 1

                # Check size
                current_mb = f.tell() / (1024 * 1024) + existing_mb

                if accepted % 100 == 0:
                    print(f"  {lang}: {current_mb:.1f} MB ({accepted} articles)")

                if current_mb >= target_mb:
                    break
            else:
                rejected += 1

            # Safety limit
            if collected > 100000:
                print(f"  {lang}: Hit collection limit")
                break

    # Save metadata
    final_size = corpus_file.stat().st_size / (1024 * 1024)
    meta = {
        "lang": lang,
        "wiki_code": wiki_code,
        "collected": collected,
        "accepted": accepted,
        "rejected": rejected,
        "size_mb": final_size,
        "timestamp": datetime.now().isoformat(),
    }
    meta_file.write_text(json.dumps(meta, indent=2))

    print(f"  {lang}: Done. {final_size:.1f} MB, {accepted} articles")
    return meta


def show_stats(output_dir):
    """Show corpus statistics."""
    output_dir = Path(output_dir)

    print("\nCorpus Statistics:")
    print("-" * 50)
    print(f"{'Lang':<6} {'Size (MB)':<12} {'Articles':<10} {'Rate':<8}")
    print("-" * 50)

    total_mb = 0
    total_articles = 0

    for lang in sorted(WIKI_MAP.keys()):
        meta_file = output_dir / f"{lang}.meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            size = meta["size_mb"]
            articles = meta["accepted"]
            rate = meta["accepted"] / max(1, meta["collected"]) * 100
            print(f"{lang:<6} {size:<12.1f} {articles:<10} {rate:<.0f}%")
            total_mb += size
            total_articles += articles
        else:
            print(f"{lang:<6} {'—':<12} {'—':<10} {'—':<8}")

    print("-" * 50)
    print(f"{'Total':<6} {total_mb:<12.1f} {total_articles:<10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="eng", help="Language code or 'all'")
    parser.add_argument("--target-mb", type=int, default=100, help="Target size in MB")
    parser.add_argument("--output", default="corpus", help="Output directory")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    args = parser.parse_args()

    if args.stats:
        show_stats(args.output)
        return

    if args.lang == "all":
        for lang, wiki_code in sorted(WIKI_MAP.items()):
            try:
                collect_wikipedia(lang, wiki_code, args.target_mb, args.output)
            except Exception as e:
                print(f"  {lang}: FAILED — {e}")
    else:
        if args.lang not in WIKI_MAP:
            print(f"Unknown language: {args.lang}")
            print(f"Available: {', '.join(sorted(WIKI_MAP.keys()))}")
            return
        collect_wikipedia(args.lang, WIKI_MAP[args.lang], args.target_mb, args.output)


if __name__ == "__main__":
    main()
