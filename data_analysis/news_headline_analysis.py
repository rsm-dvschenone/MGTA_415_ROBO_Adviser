
"""
data_analysis/news_headline_analysis.py
---------------------------------------
Sentiment + rollup for NewsAPI-style articles (headline-focused).
Accepts list[dict] with keys like: title, description, content, source, published_at, url.
Uses FinBERT if available; falls back to VADER.

API:
    from data_analysis.news_headline_analysis import analyze_news
    result = analyze_news(articles, backend="auto")
    print(result["summary"])
    items = result["items"]              # per-article with scores
    by_source = result["by_source"]      # DataFrame
    by_date = result["by_date"]          # DataFrame

    
INPUT SHAPE (articles): list[dict] with:
- title (str)        REQUIRED
- description (str)  OPTIONAL ("" if missing)
- source (str)       REQUIRED (plain string)
- published_at (str) REQUIRED ISO8601 e.g. "2025-08-16T13:25:00Z"
- url (str)          REQUIRED

"""

from __future__ import annotations
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

LABELS = ("positive", "neutral", "negative")
LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def _clean_text(s: str) -> str:
    s = URL_RE.sub("", s or "")
    return " ".join(s.split()).strip()

def _compose_text(a: Dict[str, Any]) -> str:
    title = a.get("title", "") or ""
    desc = a.get("description", "") or ""
    # Favor headline; append description lightly
    combo = f"{title}. {desc}".strip()
    return _clean_text(combo)

# ---- FinBERT backend ----
def _finbert_scores(texts: List[str], *, device: int | None, batch_size: int) -> List[Dict[str, float]]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    clf = pipeline(
        "text-classification", model=mdl, tokenizer=tok,
        truncation=True, return_all_scores=True, top_k=None,
        device=device if device is not None else -1,
    )
    outs: List[Dict[str, float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = clf(batch)
        for raw in preds:
            dist = {d["label"].lower(): float(d["score"]) for d in raw}
            for lab in LABELS: dist.setdefault(lab, 0.0)
            s = sum(dist.values()) or 1.0
            for k in list(dist.keys()): dist[k] = dist[k] / s
            outs.append(dist)
    return outs

# ---- VADER backend ----
def _vader_scores(texts: List[str]) -> List[Dict[str, float]]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    outs: List[Dict[str, float]] = []
    for t in texts:
        vs = sid.polarity_scores(t)
        dist = {"positive": float(vs.get("pos", 0.0)), "neutral": float(vs.get("neu", 0.0)), "negative": float(vs.get("neg", 0.0))}
        s = sum(dist.values()) or 1.0
        for k in list(dist.keys()): dist[k] = dist[k] / s
        outs.append(dist)
    return outs

def analyze_news(
    articles: List[Dict[str, Any]],
    *,
    backend: str = "auto",   # "auto" | "finbert" | "vader"
    device: int | None = None,
    batch_size: int = 16,
    min_chars: int = 10,
) -> Dict[str, Any]:
    if not articles:
        return {"summary": {"total": 0, "analyzed": 0, "backend": backend},
                "items": [], "by_source": pd.DataFrame(), "by_date": pd.DataFrame()}

    rows = []
    for a in articles:
        text = _compose_text(a)
        if len(text) < min_chars: continue
        rows.append({
            "text": text,
            "source": (a.get("source") or a.get("source_name") or "unknown"),
            "published_at": a.get("published_at"),
            "url": a.get("url"),
            "title": a.get("title"),
        })

    if not rows:
        return {"summary": {"total": len(articles), "analyzed": 0, "backend": backend},
                "items": [], "by_source": pd.DataFrame(), "by_date": pd.DataFrame()}

    texts = [r["text"] for r in rows]

    chosen = backend
    scores: List[Dict[str, float]] = []
    if backend == "finbert":
        scores = _finbert_scores(texts, device=device, batch_size=batch_size)
    elif backend == "vader":
        scores = _vader_scores(texts)
    else:
        try:
            scores = _finbert_scores(texts, device=device, batch_size=batch_size)
            chosen = "finbert"
        except Exception:
            scores = _vader_scores(texts)
            chosen = "vader"

    # attach predictions
    items = []
    pos = neu = neg = 0
    w_total = 0.0
    w_net = 0.0
    for r, dist in zip(rows, scores):
        label = max(dist, key=dist.get)
        if label == "positive": pos += 1
        elif label == "negative": neg += 1
        else: neu += 1
        score_val = (LABEL_TO_SCORE["positive"] * dist["positive"] +
                     LABEL_TO_SCORE["neutral"] * dist["neutral"] +
                     LABEL_TO_SCORE["negative"] * dist["negative"])
        w_total += 1.0
        w_net += score_val
        items.append({
            **r,
            "label": label,
            "score_pos": dist["positive"],
            "score_neu": dist["neutral"],
            "score_neg": dist["negative"],
            "sentiment": score_val,
        })

    analyzed_n = len(items)
    share_pos = pos / analyzed_n if analyzed_n else 0.0
    share_neg = neg / analyzed_n if analyzed_n else 0.0
    avg_sent = (w_net / w_total) if w_total else 0.0

    df = pd.DataFrame(items)
    # Rollups
    df["date"] = pd.to_datetime(df["published_at"]).dt.date if "published_at" in df.columns else None
    by_source = df.groupby("source").agg(
        n=("text","count"),
        pos=("label", lambda s: int((s=="positive").sum())),
        neu=("label", lambda s: int((s=="neutral").sum())),
        neg=("label", lambda s: int((s=="negative").sum())),
        avg_sent=("sentiment","mean"),
    ).reset_index().sort_values("n", ascending=False)

    if "date" in df.columns and df["date"].notna().any():
        by_date = df.groupby("date").agg(
            n=("text","count"),
            pos=("label", lambda s: int((s=="positive").sum())),
            neu=("label", lambda s: int((s=="neutral").sum())),
            neg=("label", lambda s: int((s=="negative").sum())),
            avg_sent=("sentiment","mean"),
        ).reset_index().sort_values("date")
    else:
        by_date = pd.DataFrame()

    summary = {
        "total": len(articles),
        "analyzed": analyzed_n,
        "positive": pos, "neutral": neu, "negative": neg,
        "share_positive": round(share_pos, 3),
        "share_negative": round(share_neg, 3),
        "avg_sentiment": round(avg_sent, 4),
        "backend": chosen,
    }
    return {"summary": summary, "items": items, "by_source": by_source, "by_date": by_date}
