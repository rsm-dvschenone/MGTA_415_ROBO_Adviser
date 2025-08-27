
"""
data_analysis/sec_risk_compare.py
---------------------------------
Compare "Risk Factors" text between the latest SEC filing (10-K/10-Q) and the previous one.

Features (no external NLP deps):
  - Tokenize to words/bigrams; compute frequency deltas
  - Jaccard similarity on token sets as a rough "how similar"
  - Sentence-level added/removed samples
  - Graceful "no new filings" path if there is nothing newer since last check

API:
    from data_analysis.sec_risk_compare import compare_sec_risk_sections

    result = compare_sec_risk_sections(
        current_text=current_text_or_None,
        previous_text=previous_text_or_None,
        metadata_current={"ticker":"NVDA","company":"NVIDIA","form":"10-Q","date":"2025-08-10"},
        metadata_previous={"ticker":"NVDA","company":"NVIDIA","form":"10-Q","date":"2025-05-08"},
        top_k=12,
    )
    print(result["summary"])
    df_terms = result["term_changes"]   # pandas DataFrame
    df_sents = result["sentence_changes"]

Returns dict with keys:
  - summary: high-level metrics & top changes
  - term_changes: DataFrame [term, type, prev_freq, curr_freq, delta]  (type in {"unigram","bigram"})
  - sentence_changes: DataFrame [change, sentence] where change in {"added","removed"}


INPUT SHAPE:
compare_sec_risk_sections(current_text, previous_text, metadata_current, metadata_previous)
- current_text:  str OR None (None => no new filing)
- previous_text: str
- metadata_*: {"ticker":str,"company":str,"form":str,"date":"YYYY-MM-DD"}
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Optional, Tuple, Iterable
from collections import Counter
import pandas as pd


##ADDED IN BY DOM HERE
from data_collection.SEC_Edgar_Downloader import prepare_compare_payload

current_text, previous_text, meta_curr, meta_prev = prepare_compare_payload(
    ticker="NVDA",
    form="10-Q",
    email="dvschenone@ucsd.edu",
    section_preference="auto",   # "auto" (10-Q: 1A→Item3), or "risk", or "item3"
    min_year=2019,
)
def compare_sec_risk_sections(current_text, previous_text, metadata_current, metadata_previous):
    # your logic here
    ...
    
result = compare_sec_risk_sections(current_text, previous_text, meta_curr.__dict__, meta_prev.__dict__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+")

_STOP = {
    # generic stopwords (small set)
    "the","and","or","of","to","a","in","for","on","as","by","with","is","are","was","were","be","been","being",
    "that","this","it","its","from","at","an","we","our","us","you","your","their","they","them","will","may","can",
    "could","should","would","not","no","if","any","all","such","other","than","more","most","including","include",
    "over","under","between","within","because","due","there","these","those","which","who","while","both","into",
    "have","has","had","do","does","did","also",
    # finance/domain boilerplate
    "company","risk","risks","factor","factors","business","operations","operating","future","financial","market",
    "results","could","may","might","material","materially","adverse","uncertain","uncertainty","changes","change",
    "laws","regulations","regulatory","legal","government","management","investors","common","stock","shares",
    "quarter","annual","form","10-k","10-q","sec","report","filing","part","item","section",
}

def _tokens(text: str) -> List[str]:
    if not text: return []
    toks = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    return [t for t in toks if t not in _STOP]

def _bigrams(tokens: List[str]) -> List[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a not in _STOP and b not in _STOP]

def _sentences(text: str) -> List[str]:
    if not text: return []
    # normalize whitespace
    text = " ".join(text.split())
    return _SENT_SPLIT.split(text)

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def _freq_delta(prev: Counter, curr: Counter) -> pd.DataFrame:
    terms = sorted(set(prev) | set(curr))
    rows = []
    for t in terms:
        pv = prev.get(t, 0)
        cv = curr.get(t, 0)
        rows.append({"term": t, "prev_freq": pv, "curr_freq": cv, "delta": cv - pv})
    return pd.DataFrame(rows)

def compare_sec_risk_sections(
    current_text: Optional[str],
    previous_text: Optional[str],
    *,
    metadata_current: Optional[Dict[str, Any]] = None,
    metadata_previous: Optional[Dict[str, Any]] = None,
    top_k: int = 12,
) -> Dict[str, Any]:
    meta_c = metadata_current or {}
    meta_p = metadata_previous or {}

    # Handle "no new filings" if current is missing or not newer
    c_date = (meta_c.get("date") or "")[:10]
    p_date = (meta_p.get("date") or "")[:10]
    if not current_text or (p_date and c_date and c_date <= p_date):
        return {
            "summary": {
                "ticker": meta_c.get("ticker") or meta_p.get("ticker"),
                "company": meta_c.get("company") or meta_p.get("company"),
                "form_type_current": meta_c.get("form"),
                "date_current": c_date or None,
                "form_type_previous": meta_p.get("form"),
                "date_previous": p_date or None,
                "no_new_filings": True,
                "message": f"No new SEC filings since {p_date or 'last check'}.",
            },
            "term_changes": pd.DataFrame(columns=["term","type","prev_freq","curr_freq","delta"]),
            "sentence_changes": pd.DataFrame(columns=["change","sentence"]),
        }

    # Tokenize
    toks_p = _tokens(previous_text or "")
    toks_c = _tokens(current_text or "")
    big_p = _bigrams(toks_p)
    big_c = _bigrams(toks_c)

    # Similarity
    sim_uni = _jaccard(toks_p, toks_c)
    sim_bi = _jaccard(big_p, big_c)

    # Frequency deltas
    df_u = _freq_delta(Counter(toks_p), Counter(toks_c))
    df_u["type"] = "unigram"
    df_b = _freq_delta(Counter(big_p), Counter(big_c))
    df_b["type"] = "bigram"
    df_terms = pd.concat([df_u, df_b], ignore_index=True)

    # Top added/removed by delta
    added = df_terms.sort_values("delta", ascending=False).query("delta > 0").head(top_k)
    removed = df_terms.sort_values("delta", ascending=True).query("delta < 0").head(top_k)

    # Sentence-level changes (rough)
    s_prev = set([s.strip() for s in _sentences(previous_text or "") if s.strip()])
    s_curr = set([s.strip() for s in _sentences(current_text or "") if s.strip()])
    added_sents = sorted(s_curr - s_prev)[:top_k]
    removed_sents = sorted(s_prev - s_curr)[:top_k]
    df_sents = pd.DataFrame(
        [{"change":"added","sentence":s} for s in added_sents] +
        [{"change":"removed","sentence":s} for s in removed_sents]
    )

    summary = {
        "ticker": meta_c.get("ticker"),
        "company": meta_c.get("company"),
        "form_type_current": meta_c.get("form"),
        "date_current": c_date or None,
        "form_type_previous": meta_p.get("form"),
        "date_previous": p_date or None,
        "no_new_filings": False,
        "similarity_unigrams": round(float(sim_uni), 3),
        "similarity_bigrams": round(float(sim_bi), 3),
        "top_added_terms": added.sort_values(["type","delta","term"], ascending=[True,False,True])["term"].tolist()[:top_k],
        "top_removed_terms": removed.sort_values(["type","delta","term"], ascending=[True,True,True])["term"].tolist()[:top_k],
        "added_sentences_sample": added_sents,
        "removed_sentences_sample": removed_sents,
    }

    return {
        "summary": summary,
        "term_changes": df_terms[["term","type","prev_freq","curr_freq","delta"]].sort_values("delta", ascending=False),
        "sentence_changes": df_sents,
    }
