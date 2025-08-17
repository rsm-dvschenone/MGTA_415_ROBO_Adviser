
"""
reddit_post_analysis.py
-----------------------
Sentiment analysis for Reddit posts with two backends:
  - FinBERT (ProsusAI/finbert) via Hugging Face Transformers (preferred when available)
  - VADER (lightweight fallback) when transformers/torch/model download are unavailable

Usage:
    from reddit_post_analysis import analyze_reddit_sentiment
    res = analyze_reddit_sentiment(posts_list, backend="auto")
    print(res["summary"])

Backends:
    backend="auto"  -> try FinBERT, else VADER
    backend="finbert" -> require FinBERT
    backend="vader"   -> force VADER fallback

Install (for FinBERT):
    pip install transformers torch --upgrade

Install (for VADER):
    pip install vaderSentiment

    
INPUT SHAPE (posts):
- list[str] OR list[dict] with:
  - text (str)   REQUIRED
  - score (int)  OPTIONAL (weights sentiment)
Example: [{"text":"NVDA lifts guidance...", "score":124}]

"""

from __future__ import annotations
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import re
from typing import List, Dict, Any, Union, Optional, Tuple

LABELS = ("positive", "neutral", "negative")
LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# ------- helpers -------
def _clean_text(s: str) -> str:
    s = URL_RE.sub("", s or "")
    return " ".join(s.split()).strip()

def _extract_text_and_weight(item: Union[str, Dict[str, Any]]) -> Optional[Tuple[str, float]]:
    if isinstance(item, str):
        text = _clean_text(item)
        return (text, 1.0) if text else None
    if isinstance(item, dict):
        text = item.get("text")
        if not text:
            title = item.get("title", "") or ""
            body = item.get("body", "") or item.get("selftext", "") or ""
            text = f"{title}\n{body}".strip()
        text = _clean_text(text)
        if not text:
            return None
        weight = item.get("weight")
        if weight is None:
            weight = item.get("score") or item.get("upvotes") or 1.0
        try:
            w = float(weight)
            if not (w == w) or w <= 0:
                w = 1.0
        except Exception:
            w = 1.0
        return (text, w)
    return None

# ------- FinBERT backend -------
def _load_finbert_pipeline(model_name: str = "ProsusAI/finbert", device: Optional[int] = None):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    except Exception as e:
        raise ImportError("transformers is required for FinBERT. Install with: pip install transformers torch --upgrade") from e
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        truncation=True,
        return_all_scores=True,
        top_k=None,
        device=device if device is not None else -1,
    )
    return clf

def _finbert_scores(texts: List[str], *, batch_size: int, device: Optional[int]) -> List[Dict[str, float]]:
    clf = _load_finbert_pipeline(device=device)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = clf(batch)
        # normalize to dict[label]=prob
        for raw in preds:
            dist: Dict[str, float] = {d["label"].lower(): float(d["score"]) for d in raw}
            for lab in LABELS:
                dist.setdefault(lab, 0.0)
            s = sum(dist.values()) or 1.0
            for k in list(dist.keys()):
                dist[k] = dist[k] / s
            out.append(dist)
    return out

# ------- VADER backend -------
def _vader_scores(texts: List[str]) -> List[Dict[str, float]]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as e:
        raise ImportError("vaderSentiment is required for VADER fallback. Install with: pip install vaderSentiment") from e
    sid = SentimentIntensityAnalyzer()
    outs: List[Dict[str, float]] = []
    for t in texts:
        vs = sid.polarity_scores(t)  # {'neg':x, 'neu':y, 'pos':z, 'compound':c}
        # Map to FinBERT-style labels
        dist = {"positive": float(vs.get("pos", 0.0)), "neutral": float(vs.get("neu", 0.0)), "negative": float(vs.get("neg", 0.0))}
        s = sum(dist.values()) or 1.0
        for k in list(dist.keys()):
            dist[k] = dist[k] / s
        outs.append(dist)
    return outs

# ------- main API -------
def analyze_reddit_sentiment(
    posts: List[Union[str, Dict[str, Any]]],
    *,
    model_name: str = "ProsusAI/finbert",
    device: Optional[int] = None,
    use_weight: bool = True,
    filter_keywords: Optional[List[str]] = None,
    min_chars: int = 10,
    batch_size: int = 16,
    include_items: bool = True,
    backend: str = "auto",   # "auto" | "finbert" | "vader"
) -> Dict[str, Any]:
    """
    Returns dict with keys:
      - summary: {total_posts, analyzed, positive, neutral, negative, share_positive, share_negative,
                  weighted_net, avg_sentiment, examples: {positive, negative}, backend}
      - items: list of {text, weight, label, scores:{positive,neutral,negative}} (if include_items)
    """
    if not posts:
        return {"summary": {"total_posts": 0, "analyzed": 0, "backend": backend}, "items": []}

    # Normalize & filter
    norm: List[Tuple[str, float]] = []
    kw = [k.lower() for k in (filter_keywords or [])]
    for it in posts:
        tup = _extract_text_and_weight(it)
        if not tup:
            continue
        text, w = tup
        if len(text) < min_chars:
            continue
        if kw and not any(k in text.lower() for k in kw):
            continue
        norm.append((text, w))

    total_posts = len(posts)
    analyzed_n = len(norm)
    if analyzed_n == 0:
        return {
            "summary": {
                "total_posts": total_posts,
                "analyzed": 0,
                "note": "No posts matched filters / min length.",
                "backend": backend,
            },
            "items": [],
        }

    texts = [t for (t, _) in norm]
    weights = [w for (_, w) in norm]

    # Choose backend
    chosen = backend
    scores: List[Dict[str, float]] = []
    if backend == "finbert":
        scores = _finbert_scores(texts, batch_size=batch_size, device=device)
    elif backend == "vader":
        scores = _vader_scores(texts)
    else:  # auto
        try:
            scores = _finbert_scores(texts, batch_size=batch_size, device=device)
            chosen = "finbert"
        except Exception:
            scores = _vader_scores(texts)
            chosen = "vader"

    # Aggregate
    items = []
    pos = neu = neg = 0
    w_total = 0.0
    w_net = 0.0
    best_pos = (None, -1.0)
    best_neg = (None, -1.0)

    for (text, w), dist in zip(norm, scores):
        label = max(dist, key=dist.get)
        if label == "positive":
            pos += 1
            if dist["positive"] > best_pos[1]:
                best_pos = (text, dist["positive"])
        elif label == "negative":
            neg += 1
            if dist["negative"] > best_neg[1]:
                best_neg = (text, dist["negative"])
        else:
            neu += 1

        if use_weight:
            w_total += w
            w_net += w * (LABEL_TO_SCORE["positive"] * dist["positive"] +
                          LABEL_TO_SCORE["neutral"] * dist["neutral"] +
                          LABEL_TO_SCORE["negative"] * dist["negative"])
        else:
            w_total += 1.0
            w_net += (LABEL_TO_SCORE["positive"] * dist["positive"] +
                      LABEL_TO_SCORE["neutral"] * dist["neutral"] +
                      LABEL_TO_SCORE["negative"] * dist["negative"])

        if include_items:
            items.append({
                "text": text,
                "weight": w,
                "label": label,
                "scores": {"positive": dist["positive"], "neutral": dist["neutral"], "negative": dist["negative"]},
            })

    share_pos = pos / analyzed_n if analyzed_n else 0.0
    share_neg = neg / analyzed_n if analyzed_n else 0.0
    avg_sent = (w_net / w_total) if w_total else 0.0

    summary = {
        "total_posts": total_posts,
        "analyzed": analyzed_n,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "share_positive": round(share_pos, 3),
        "share_negative": round(share_neg, 3),
        "weighted_net": round(w_net, 4),
        "avg_sentiment": round(avg_sent, 4),
        "examples": {"positive": best_pos[0], "negative": best_neg[0]},
        "backend": chosen,
    }
    return {"summary": summary, "items": items if include_items else None}

if __name__ == "__main__":
    demo = [
        "NVIDIA crushes earnings again; datacenter demand is off the charts.",
        "I think NVDA valuation is insane right now, stay cautious.",
        "Neutral take: waiting for next quarter guidance.",
    ]
    out = analyze_reddit_sentiment(demo, backend="auto")
    from pprint import pprint
    pprint(out["summary"])
