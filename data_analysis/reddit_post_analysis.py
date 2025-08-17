
"""
reddit_post_analysis.py
-----------------------
FinBERT-powered sentiment analysis for Reddit posts.

Usage (module):
    from reddit_post_analysis import analyze_reddit_sentiment

    results = analyze_reddit_sentiment(posts_list)
    print(results["summary"])

Notes:
  - Requires: transformers (and a backend like torch). First time will download the model.
  - Model: ProsusAI/finbert (financial-domain BERT fine-tuned for sentiment).
  - Input flexibility: Accepts a list of strings, or dicts with keys like
        {"text": "...", "score": 123} or {"title": "...", "body": "...", "upvotes": 99}
  - Weighting: If a numeric field like "score"/"upvotes" is present, it is used as a weight (optional).
  - Output: Aggregates counts, weighted net score (-1..+1), and returns per-item predictions.

(c) Your team – MIT / Apache2 style licensing is fine if you want to open-source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional, Tuple

# We import lazily so downstream code can import this module even if transformers isn't installed yet.
_TRANSFORMERS_OK = True
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
except Exception:
    _TRANSFORMERS_OK = False

LABELS = ("positive", "neutral", "negative")
LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def _clean_text(s: str) -> str:
    # Strip URLs and extra whitespace; keep finance tickers/slang intact for FinBERT.
    s = URL_RE.sub("", s or "")
    return " ".join(s.split()).strip()

def _extract_text_and_weight(item: Union[str, Dict[str, Any]]) -> Optional[Tuple[str, float]]:
    if isinstance(item, str):
        text = _clean_text(item)
        return (text, 1.0) if text else None
    if isinstance(item, dict):
        # Prefer explicit "text"; else combine typical reddit fields.
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
            if not (w == w) or w <= 0:  # NaN or non-positive -> default
                w = 1.0
        except Exception:
            w = 1.0
        return (text, w)
    # Unknown type
    return None

def _load_finbert_pipeline(model_name: str = "ProsusAI/finbert", device: Optional[int] = None):
    if not _TRANSFORMERS_OK:
        raise ImportError(
            "transformers is required. Try: pip install transformers torch --upgrade"
        )
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        truncation=True,
        return_all_scores=True,
        top_k=None,
        device=device if device is not None else -1,  # -1 CPU, or CUDA device id
    )
    return clf

def _dist_from_pipeline_output(one: Any) -> Dict[str, float]:
    """
    Pipeline with return_all_scores=True returns a list[dict] per input where each dict
    has keys: {'label': str, 'score': float}. We normalize labels to lower-case.
    """
    dist: Dict[str, float] = {}
    if isinstance(one, list):
        for d in one:
            lab = str(d.get("label", "")).lower()
            sc = float(d.get("score", 0.0))
            dist[lab] = sc
    elif isinstance(one, dict):  # fallback if pipeline returns single dict
        lab = str(one.get("label", "")).lower()
        sc = float(one.get("score", 0.0))
        dist[lab] = sc
    # Ensure all expected labels exist; fill missing with 0
    for lab in LABELS:
        dist.setdefault(lab, 0.0)
    # Re-normalize in case of drift
    s = sum(dist.values()) or 1.0
    for k in list(dist.keys()):
        dist[k] = dist[k] / s
    return dist

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
) -> Dict[str, Any]:
    """
    Run FinBERT over a list of reddit posts (strings or dicts). Returns an aggregate summary and per-item predictions.

    Returns dict with keys:
      - summary: {total_posts, analyzed, positive, neutral, negative, share_positive, share_negative,
                  weighted_net, avg_sentiment, examples: {positive, negative}}
      - items: list of {text, weight, label, scores:{pos,neu,neg}} (if include_items)
    """
    if not posts:
        return {"summary": {"total_posts": 0, "analyzed": 0}, "items": []}

    # Normalize and (optional) keyword filter
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
            },
            "items": [],
        }

    clf = _load_finbert_pipeline(model_name=model_name, device=device)

    texts = [t for (t, _) in norm]
    weights = [w for (_, w) in norm]

    # Batch the inputs for efficiency
    preds: List[Any] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds.extend(clf(batch))

    items = []
    pos = neu = neg = 0
    # For weighted metrics
    w_total = 0.0
    w_net = 0.0

    # Track best examples
    best_pos = (None, -1.0)  # (text, confidence)
    best_neg = (None, -1.0)

    for (text, w), raw in zip(norm, preds):
        dist = _dist_from_pipeline_output(raw)
        # Pick label by argmax
        label = max(dist, key=dist.get)
        # Unweighted counts
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
        # Weighted net (positive minus negative)
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
    avg_sent = (w_net / w_total) if w_total else 0.0  # in [-1, 1]

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
        "examples": {
            "positive": best_pos[0],
            "negative": best_neg[0],
        },
    }

    return {"summary": summary, "items": items if include_items else None}

if __name__ == "__main__":
    # Simple CLI demo: put your posts in a JSON file or edit the list below.
    demo_posts = [
        "NVIDIA crushes earnings again; datacenter demand is off the charts.",
        "I think NVDA valuation is insane right now, stay cautious.",
        "Neutral take: waiting for next quarter guidance.",
    ]
    out = analyze_reddit_sentiment(demo_posts)
    from pprint import pprint
    pprint(out["summary"])
