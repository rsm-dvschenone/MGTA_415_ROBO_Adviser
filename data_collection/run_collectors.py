#!/usr/bin/env python3
"""
Unified collector runner.

Outputs (written next to this script):
- articles.json        (NewsAPI, list[dict])
- reddit_posts.json    (Reddit, list[dict])
- prices.csv           (yfinance, DataFrame -> CSV)
- sec_sections.json    (SEC EDGAR, dict with current/previous texts + metadata)

ENV VARS (set before running):
- NEWSAPI_KEY=...
- REDDIT_CLIENT_ID=...
- REDDIT_CLIENT_SECRET=...
- REDDIT_USER_AGENT=nvda-bot
- REDDIT_USERNAME=...
- REDDIT_PASSWORD=...
- SEC_CONTACT_EMAIL=you@example.com

Optional:
- TICKER (default: NVDA)
- QUERY  (default: NVIDIA)
- NEWS_TOTAL (default: 50)
- REDDIT_LIMIT (default: 25)
- YF_PERIOD (default: 60d)
- YF_INTERVAL (default: 1d)
- SEC_FORM (default: 10-Q)
- SEC_MIN_YEAR (default: 2019)
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import asdict

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent              # .../data_collection
sys.path.append(str(ROOT.parent))                   # add project root so 'data_collection.*' imports work

# ---------- imports from your package ----------
from data_collection.newapi import fetch_news
from data_collection.reddit import collect as collect_reddit
from data_collection.yfinance import collect as collect_prices
from data_collection.SEC_Edgar_Downloader import prepare_compare_payload


def main() -> None:
    # ------------ config ------------
    ticker = os.getenv("TICKER", "NVDA")
    query = os.getenv("QUERY", "NVIDIA")

    news_total = int(os.getenv("NEWS_TOTAL", "50"))
    reddit_limit = int(os.getenv("REDDIT_LIMIT", "25"))
    yf_period = os.getenv("YF_PERIOD", "60d")
    yf_interval = os.getenv("YF_INTERVAL", "1d")

    sec_form = os.getenv("SEC_FORM", "10-Q")
    sec_min_year = int(os.getenv("SEC_MIN_YEAR", "2019"))
    sec_email = os.getenv("SEC_CONTACT_EMAIL")
    if not sec_email:
        raise SystemExit("Missing SEC_CONTACT_EMAIL env var (required by sec-edgar-downloader).")

    news_api_key = os.getenv("NEWSAPI_KEY")
    if not news_api_key:
        raise SystemExit("Missing NEWSAPI_KEY env var.")

    # Keep EDGAR downloads under your repo: data_collection/sec-edgar-filings/
    sec_data_dir = ROOT / "sec-edgar-filings"
    sec_data_dir.mkdir(parents=True, exist_ok=True)

    # ------------ 1) NewsAPI articles ------------
    articles = fetch_news(
        api_key=news_api_key,
        query=query,
        total_articles=news_total,
    )

    # ------------ 2) Reddit posts ------------
    posts = collect_reddit(
        query=query,
        limit=reddit_limit,
    )

    # ------------ 3) yfinance prices ------------
    prices = collect_prices(
        ticker=ticker,
        period=yf_period,
        interval=yf_interval,
    )

    # ------------ 4) SEC EDGAR sections ------------
    current_text, previous_text, meta_curr, meta_prev = prepare_compare_payload(
        ticker=ticker,
        form=sec_form,
        email=sec_email,
        data_dir=sec_data_dir,
        section_preference="auto",
        min_year=sec_min_year,
    )
    sec_payload = {
        "ticker": ticker,
        "form": sec_form,
        "current": {
            "text": (current_text or ""),
            "metadata": asdict(meta_curr),
        },
        "previous": {
            "text": (previous_text or ""),
            "metadata": asdict(meta_prev),
        },
    }

    # ------------ write outputs ------------
    (ROOT / "articles.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "reddit_posts.json").write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")
    prices.to_csv(ROOT / "prices.csv", index=False)
    (ROOT / "sec_sections.json").write_text(json.dumps(sec_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Wrote:")
    print(" -", (ROOT / "articles.json").name)
    print(" -", (ROOT / "reddit_posts.json").name)
    print(" -", (ROOT / "prices.csv").name)
    print(" -", (ROOT / "sec_sections.json").name)


if __name__ == "__main__":
    main()


# deps
# pip install newsapi-python praw yfinance pandas numpy sec-edgar-downloader beautifulsoup4 lxml

# env
# export NEWSAPI_KEY=...
# export REDDIT_CLIENT_ID=...
# export REDDIT_CLIENT_SECRET=...
# export REDDIT_USER_AGENT=nvda-bot
# export REDDIT_USERNAME=...
# export REDDIT_PASSWORD=...
# export SEC_CONTACT_EMAIL=you@example.com

# run
# python data_collection/run_collectors.py