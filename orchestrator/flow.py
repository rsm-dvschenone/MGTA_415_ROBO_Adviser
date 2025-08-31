import os, sys, json, time, logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env from repo root (works locally; on HF Spaces use Repo Secrets)
load_dotenv(dotenv_path=Path.cwd() / ".env")

# --- your existing modules ---
from data_collection.run_collectors import main as collect
from data_analysis.reddit_post_analysis import analyze_reddit_sentiment
from data_analysis.news_headline_analysis import analyze_news
from data_analysis.price_indicators import analyze_rsi_macd
from data_analysis.sec_risk_compare import compare_sec_risk_sections

import pandas as pd

# ---------- logging ----------
LOGS_DIR = Path("logs"); LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "flow.log", encoding="utf-8")
    ],
)

def retry(n=3, delay=3.0):
    def wrap(fn):
        def inner(*a, **kw):
            last = None
            for i in range(n):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    last = e
                    logging.warning("Retry %d/%d after error: %s", i+1, n, e)
                    time.sleep(delay)
            raise last
        return inner
    return wrap

@retry()
def step_collect():
    logging.info("Collecting raw data…")
    # Writes: articles.json, reddit_posts.json, prices.csv, sec_sections.json
    return collect()

def step_load_outputs():
    logging.info("Loading collected artifacts…")
    with open("articles.json", "r") as f:
        articles = json.load(f)
    with open("reddit_posts.json", "r") as f:
        posts = json.load(f)
    prices = pd.read_csv("prices.csv", parse_dates=["Date"])
    with open("sec_sections.json", "r") as f:
        sec = json.load(f)
    return articles, posts, prices, sec

def step_analyze(articles, posts, prices_df, sec):
    logging.info("Analyzing…")
    news = analyze_news(articles, backend="finbert")
    reddit = analyze_reddit_sentiment(posts, backend="finbert")
    prices = analyze_rsi_macd(prices_df)
    sec_res = compare_sec_risk_sections(
        sec["current"]["text"], sec["previous"]["text"],
        metadata_current=sec["current"].get("metadata"),
        metadata_previous=sec["previous"].get("metadata"),
    )
    return {"news": news, "reddit": reddit, "prices": prices, "sec": sec_res}

def _artifact_dir():
    # dated run folder: artifacts/YYYY-MM-DD_HHMMSS
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = Path("artifacts") / ts
    d.mkdir(parents=True, exist_ok=True)
    return d

def _copy_into(run_dir: Path):
    # Keep raw outputs per run
    for name in ["articles.json", "reddit_posts.json", "prices.csv", "sec_sections.json"]:
        p = Path(name)
        if p.exists():
            (run_dir / name).write_bytes(p.read_bytes())

def main():
    start = time.time()
    run_dir = _artifact_dir()
    logging.info("Run folder: %s", run_dir)

    step_collect()
    articles, posts, prices_df, sec = step_load_outputs()
    results = step_analyze(articles, posts, prices_df, sec)

    summary = {
        "summary": {
            "news": results["news"].get("summary"),
            "reddit": results["reddit"].get("summary"),
            "prices": results["prices"].get("summary"),
            "sec": results["sec"].get("summary"),
        },
        "meta": {
            "ticker": os.getenv("TICKER", "NVDA"),
            "query": os.getenv("QUERY", "NVIDIA"),
            "run_started": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
            "run_ended": datetime.now().isoformat(timespec="seconds"),
        }
    }

    # Write a top-level summary and a per-run copy
    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

    # Save copies of raw artifacts (per-run)
    _copy_into(run_dir)

    logging.info("Done. Wrote analysis_summary.json and %s", run_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
