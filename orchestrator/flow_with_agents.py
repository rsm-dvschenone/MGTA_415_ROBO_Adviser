# orchestrator/flow_with_agents.py
import os, sys, json, time, logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env locally; on HF Spaces use Repo Secrets (os.getenv)
load_dotenv(dotenv_path=Path.cwd() / ".env")

from data_collection.run_collectors import main as collect
from data_analysis.reddit_post_analysis import analyze_reddit_sentiment
from data_analysis.news_headline_analysis import analyze_news
from data_analysis.price_indicators import analyze_rsi_macd
from data_analysis.sec_risk_compare import compare_sec_risk_sections

from agents.agents import planner_agent, writer_agent

import pandas as pd

# ---------- logging ----------
LOGS_DIR = Path("logs"); LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "flow_with_agents.log", encoding="utf-8")
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

def _make_meta(articles, posts, prices_df, sec):
    return {
        "articles_count": len(articles or []),
        "posts_count": len(posts or []),
        "prices_rows": int(getattr(prices_df, "shape", [0])[0]),
        "has_sec": bool(sec and "current" in sec and "previous" in sec),
        "ticker": os.getenv("TICKER", "NVDA"),
        "query": os.getenv("QUERY", "NVIDIA"),
    }

def step_analyze_selected(run_plan, articles, posts, prices_df, sec):
    results = {}
    if run_plan.get("run_news"):
        logging.info("Running news analysis…")
        results["news"] = analyze_news(articles, backend="finbert")
    if run_plan.get("run_reddit"):
        logging.info("Running reddit analysis…")
        results["reddit"] = analyze_reddit_sentiment(posts, backend="finbert")
    if run_plan.get("run_prices"):
        logging.info("Running price indicators…")
        results["prices"] = analyze_rsi_macd(prices_df)
    if run_plan.get("run_sec"):
        logging.info("Running SEC comparison…")
        results["sec"] = compare_sec_risk_sections(
            sec["current"]["text"], sec["previous"]["text"],
            metadata_current=sec["current"].get("metadata"),
            metadata_previous=sec["previous"].get("metadata"),
        )
    return results

def _artifact_dir():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = Path("artifacts") / ts
    d.mkdir(parents=True, exist_ok=True)
    return d

def _copy_into(run_dir: Path):
    for name in ["articles.json", "reddit_posts.json", "prices.csv", "sec_sections.json"]:
        p = Path(name)
        if p.exists():
            (run_dir / name).write_bytes(p.read_bytes())

def main():
    start = time.time()
    run_dir = _artifact_dir()
    logging.info("Run folder: %s", run_dir)

    # 1) Collect & load
    step_collect()
    articles, posts, prices_df, sec = step_load_outputs()

    # 2) Plan (AI agent decides what to run)
    meta = _make_meta(articles, posts, prices_df, sec)
    plan = planner_agent(
        objectives=f"Assess {meta['ticker']} near-term risks & sentiment for daily brief.",
        artifact_meta=meta
    )
    logging.info("Planner decision: %s", plan)

    # 3) Execute selected analyses
    results = step_analyze_selected(plan, articles, posts, prices_df, sec)

    # 4) Summaries for writer
    summaries = {
        "news": results.get("news", {}).get("summary", "N/A"),
        "reddit": results.get("reddit", {}).get("summary", "N/A"),
        "prices": results.get("prices", {}).get("summary", "N/A"),
        "sec": results.get("sec", {}).get("summary", "N/A"),
        "bottom_line": "Hold pending stronger momentum or positive catalysts."
    }
    findings_for_writer = {
        "summaries": summaries,
        "signals": {
            "news": results.get("news", {}),
            "reddit": results.get("reddit", {}),
            "prices": results.get("prices", {}),
            "sec": results.get("sec", {}),
        },
        "meta": meta,
        "plan_notes": plan.get("notes", "")
    }

    # 5) Writer agent produces the brief
    brief_md = writer_agent(findings_for_writer, audience="PM")
    (run_dir / "brief.md").write_text(brief_md, encoding="utf-8")

    # 6) Persist machine-readable summary
    summary_payload = {"summary": summaries, "meta": meta, "plan": plan}
    Path("analysis_summary.json").write_text(json.dumps(summary_payload, indent=2))
    (run_dir / "analysis_summary.json").write_text(json.dumps(summary_payload, indent=2))

    # 7) Save copies of raw artifacts
    _copy_into(run_dir)

    logging.info("Done. Artifacts: %s", run_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
