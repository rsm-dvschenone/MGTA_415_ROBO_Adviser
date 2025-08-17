
"""
run_all_dummy_analysis.py
-------------------------
One-shot test for ALL analysis modules (Reddit, News, Prices/RSI-MACD, SEC) using dummy data.
- Forces text-only Transformers mode (no torchvision).
- Uses FinBERT via a local folder if you pass --model-path (e.g., models/finbert).
- Falls back gracefully if FinBERT fails (uses VADER for News/Reddit).

Usage:
    python run_all_dummy_analysis.py --model-path models/finbert
"""
import os, argparse, sys
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from pprint import pprint
import pandas as pd
import numpy as np

# --- Optional: patch Transformers to use a local FinBERT folder ---
def patch_transformers_to_local(model_path: str):
    try:
        import transformers
    except Exception as e:
        print("[WARN] transformers not available; cannot patch:", e)
        return False

    try:
        orig_tok = transformers.AutoTokenizer.from_pretrained
        orig_mdl = transformers.AutoModelForSequenceClassification.from_pretrained

        def tok_fp(name, *args, **kwargs):
            if isinstance(name, str) and name.lower() == "prosusai/finbert":
                name = model_path
                kwargs.setdefault("local_files_only", True)
            return orig_tok(name, *args, **kwargs)

        def mdl_fp(name, *args, **kwargs):
            if isinstance(name, str) and name.lower() == "prosusai/finbert":
                name = model_path
                kwargs.setdefault("local_files_only", True)
            return orig_mdl(name, *args, **kwargs)

        transformers.AutoTokenizer.from_pretrained = tok_fp
        transformers.AutoModelForSequenceClassification.from_pretrained = mdl_fp
        print(f"[INFO] Patched Transformers to load FinBERT from: {model_path}")
        return True
    except Exception as e:
        print("[WARN] Failed to patch Transformers:", e)
        return False

def run_reddit(model_path: str | None):
    from data_analysis.reddit_post_analysis import analyze_reddit_sentiment
    posts = [
        "NVIDIA (NVDA) posts record data center revenue; guidance raised.",
        "NVDA valuation looks stretched; risk of a sharp pullback.",
        "Neutral: waiting for Blackwell details before adding more shares.",
        "Positive: CUDA ecosystem and partnerships strengthen NVIDIA's moat.",
        "Bear case: export restrictions and power limits could cap deployments."
    ]
    try:
        print("\n[Reddit] FinBERT analysis...")
        res = analyze_reddit_sentiment(
            posts, backend="finbert",
            model_name=(model_path or "ProsusAI/finbert"),
            include_items=True,
        )
    except Exception as e:
        print("[WARN] FinBERT failed for Reddit -> falling back to VADER:", e)
        res = analyze_reddit_sentiment(posts, backend="vader", include_items=True)
    pprint(res["summary"])
    return res

def run_news(model_path: str | None):
    # Patch Transformers redirection BEFORE importing the module (so its internal loads use local path)
    if model_path:
        patch_transformers_to_local(model_path)

    from data_analysis.news_headline_analysis import analyze_news
    articles = [
        {"title": "NVIDIA (NVDA) lifts guidance as AI chip demand soars",
         "description": "Data center revenue sets a record; shares rise in premarket.",
         "source": "BloomWorld", "published_at": "2025-08-15T12:35:00Z", "url": "https://example.com/a1"},
        {"title": "Concerns mount over NVDA valuation amid competition from custom silicon",
         "description": "Analysts warn growth may moderate as hyperscaler budgets normalize.",
         "source": "StreetDaily", "published_at": "2025-08-15T14:10:00Z", "url": "https://example.com/a2"},
        {"title": "Neutral take: waiting for Blackwell details before revising estimates",
         "description": "Wall Street looks to next-gen roadmap clarity.",
         "source": "TechTape", "published_at": "2025-08-16T09:00:00Z", "url": "https://example.com/a3"},
    ]
    try:
        print("\n[News] FinBERT analysis...")
        res = analyze_news(articles, backend="finbert", device=None)
    except Exception as e:
        print("[WARN] FinBERT failed for News -> falling back to VADER:", e)
        res = analyze_news(articles, backend="vader")
    pprint(res["summary"])
    # Save outputs
    pd.DataFrame(res["items"]).to_csv("outputs/news_items_scored_all_dummy.csv", index=False)
    if hasattr(res["by_source"], "to_csv"): res["by_source"].to_csv("outputs/news_by_source_all_dummy.csv", index=False)
    if hasattr(res["by_date"], "to_csv") and not res["by_date"].empty: res["by_date"].to_csv("outputs/news_by_date_all_dummy.csv", index=False)
    return res

def run_prices():
    from data_analysis.price_indicators import analyze_rsi_macd
    n = 260
    dates = pd.bdate_range("2024-08-01", periods=n)
    mu_daily = 0.0009
    sigma_daily = 0.022
    rng = np.random.default_rng(123)
    shocks = rng.normal(mu_daily, sigma_daily, size=n)
    close = 450.0 * np.exp(np.cumsum(shocks))
    df = pd.DataFrame({"Date": dates, "Close": close})
    print("\n[Prices] RSI & MACD analysis...")
    res = analyze_rsi_macd(df, price_col="Close")
    pprint(res["summary"])
    res["frame"].to_csv("outputs/yfinance_rsi_macd_all_dummy.csv")
    return res

def run_sec():
    from data_analysis.sec_risk_compare import compare_sec_risk_sections
    prev_text = (
        "Our results may be affected by supply chain disruptions and macro conditions. "
        "Dependence on third-party foundries may constrain our ability to meet demand. "
        "Export controls and changes in trade policies may limit sales to certain regions. "
        "We face intense competition in AI accelerators."
    )
    curr_text = (
        "We continue to face supply constraints and logistics challenges that may affect availability of our AI accelerators. "
        "Power availability and data center capacity limitations could restrict customer deployments. "
        "Export controls, including potential new restrictions on advanced AI hardware, may reduce sales. "
        "Competition from custom silicon designs is increasing."
    )
    print("\n[SEC] Risk section comparison...")
    res = compare_sec_risk_sections(
        current_text=curr_text,
        previous_text=prev_text,
        metadata_current={"ticker":"NVDA","company":"NVIDIA","form":"10-Q","date":"2025-08-16"},
        metadata_previous={"ticker":"NVDA","company":"NVIDIA","form":"10-Q","date":"2025-05-10"},
        top_k=10,
    )
    pprint(res["summary"])
    res["term_changes"].to_csv("outputs/sec_term_changes_all_dummy.csv", index=False)
    res["sentence_changes"].to_csv("outputs/sec_sentence_changes_all_dummy.csv", index=False)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default=None, help="Local FinBERT folder (e.g., models/finbert).")
    args = ap.parse_args()

    if args.model_path:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    r = run_reddit(args.model_path)
    n = run_news(args.model_path)
    p = run_prices()
    s = run_sec()

    print("\nAll modules tested. Outputs saved in ./outputs/.")

if __name__ == "__main__":
    main()
