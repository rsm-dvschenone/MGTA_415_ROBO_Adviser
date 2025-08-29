
# run_robo_advisor.py
# -------------------
# Full pipeline: collect data → analyze → generate human-readable summary report.

import os
import json
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from data_collection.run_collectors import main as collect_data
from data_analysis.reddit_post_analysis import analyze_reddit_sentiment
from data_analysis.news_headline_analysis import analyze_news
from data_analysis.price_indicators import analyze_rsi_macd
from data_analysis.sec_risk_compare import compare_sec_risk_sections
from report_generator import generate_report

def run_robo_advisor(model_path="models/finbert"):
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    # 1. Collect data (writes files)
    collect_data()

    # 2. Load collected files
    with open("data_collection/reddit_posts.json", encoding="utf-8") as f:
        reddit_posts = json.load(f)

    with open("data_collection/articles.json", encoding="utf-8") as f:
        news_articles = json.load(f)

    prices_df = pd.read_csv("data_collection/prices.csv", parse_dates=["Date"])

    with open("data_collection/sec_sections.json", encoding="utf-8") as f:
        sec = json.load(f)
        current_text = sec["current"]["text"] or None
        previous_text = sec["previous"]["text"]
        meta_curr = sec["current"]["metadata"]
        meta_prev = sec["previous"]["metadata"]

    # 3. Analyze
    reddit = analyze_reddit_sentiment(reddit_posts, backend="finbert", model_name=model_path)
    news = analyze_news(news_articles, backend="finbert")
    prices = analyze_rsi_macd(prices_df)
    sec_res = compare_sec_risk_sections(current_text, previous_text,
                                        metadata_current=meta_curr,
                                        metadata_previous=meta_prev)

    # 4. Format for report
    last = prices["summary"]
    price = last["last_close"]
    change = round((prices_df["Close"].iloc[-1] - prices_df["Close"].iloc[-2]) / prices_df["Close"].iloc[-2] * 100, 2)
    rsi_val = last["RSI_last"]
    macd_signal = f'{last["MACD_state_last"].title()} (cross: {last["MACD_cross_last"]})'

    # Simple sentiment summary
    def fmt_sentiment(s):
        if s["avg_sentiment"] > 0.1: return "Positive"
        if s["avg_sentiment"] < -0.1: return "Negative"
        return "Neutral"

    news_sent = fmt_sentiment(news["summary"])
    reddit_sent = fmt_sentiment(reddit["summary"])
    sec_note = sec_res["summary"].get("message") or f"Similarity: {sec_res['summary']['similarity_unigrams']*100:.1f}%"

    # Naive rule-based signal (demo only)
    if rsi_val > 70 and news_sent == "Negative":
        signal = "⚠️ Take Profit"
    elif rsi_val < 30 and reddit_sent == "Positive":
        signal = "🟢 Buy the Dip"
    elif news_sent == "Positive" and reddit_sent == "Positive":
        signal = "🟢 Bullish Momentum"
    else:
        signal = "🤔 Hold"

    report = generate_report({
        "price": f"{price:.2f}",
        "change": f"{change:+.2f}%",
        "rsi": rsi_val,
        "macd_signal": macd_signal,
        "news_sentiment": news_sent,
        "reddit_sentiment": reddit_sent,
        "sec_summary": sec_note,
        "final_signal": signal,
    })
    print(report)
    return report


if __name__ == "__main__":
    run_robo_advisor()
