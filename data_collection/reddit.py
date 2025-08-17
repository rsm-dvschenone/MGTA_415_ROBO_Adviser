
"""
data_collection/reddit.py
-------------------------
Simplified Reddit collector using PRAW with **inline credentials**.
(You said you'll secure this later; this is for getting your analysis unblocked.)

Usage (CLI):
    python -m data_collection.reddit --subreddits stocks,investing --query nvidia --limit 50 --time_filter week
"""

import argparse
import time
from typing import List, Dict, Any, Iterable, Union, Optional

import praw


# ====== INLINE CREDENTIALS (temporary for school project) ======
CLIENT_ID = "PaRTKKaXCdxW5q6biYW5zy0t4EY1Lw"
CLIENT_SECRET = "EfrdPP2uPFX7xv-nMNKCFQ"
USER_AGENT = "script:nvda-bot:0.1 (by u/dschenone22)"
USERNAME = "dschenone22"
PASSWORD = "giggle15"
# ===============================================================


def _init_reddit() -> "praw.Reddit":
    # If USERNAME/PASSWORD are present, authenticate with password grant (user context).
    # Otherwise, use app-only read-only.
    if USERNAME and PASSWORD:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            username=USERNAME,
            password=PASSWORD,
            check_for_async=False,
        )
    else:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
            check_for_async=False,
        )
        reddit.read_only = True
    return reddit


def _normalize_submission(sub) -> Dict[str, Any]:
    body = getattr(sub, "selftext", "") or ""
    return {
        "id": sub.id,
        "subreddit": str(sub.subreddit),
        "title": sub.title or "",
        "body": body,
        "text": f"{sub.title or ''}\n{body}".strip(),
        "score": int(getattr(sub, "score", 0) or 0),
        "num_comments": int(getattr(sub, "num_comments", 0) or 0),
        "created_utc": float(getattr(sub, "created_utc", 0.0) or 0.0),
        "author": str(getattr(sub, "author", "")) if getattr(sub, "author", None) else None,
        "url": getattr(sub, "url", None),
        "permalink": f"https://www.reddit.com{sub.permalink}" if getattr(sub, "permalink", None) else None,
    }


def get_reddit_posts(
    subreddits: Union[str, Iterable[str]] = "stocks",
    query: Optional[str] = None,
    limit: int = 100,
    time_filter: str = "week",   # one of: "all","day","hour","month","week","year"
    sort: str = "relevance",     # "relevance","hot","top","new","comments"
    min_score: int = 0,
    return_strings: bool = True, # if True -> list[str]; else list[dict]
) -> List[Union[str, Dict[str, Any]]]:
    """
    Fetch posts from one or more subreddits. If `query` is provided, uses subreddit search;
    otherwise pulls from 'hot'.

    If return_strings=True (default), returns list[str] combining title/body (good for sentiment).
    If False, returns list[dict] with rich metadata.
    """
    reddit = _init_reddit()

    if isinstance(subreddits, str):
        sub_list = [s.strip() for s in subreddits.split(",") if s.strip()]
    else:
        sub_list = [str(s).strip() for s in subreddits if str(s).strip()]

    results: List[Dict[str, Any]] = []
    per_sub_limit = max(1, limit // max(1, len(sub_list)))

    for sub_name in sub_list:
        sub = reddit.subreddit(sub_name)
        if query:
            it = sub.search(query=query, sort=sort, time_filter=time_filter, limit=per_sub_limit)
        else:
            it = sub.hot(limit=per_sub_limit)
        for submission in it:
            d = _normalize_submission(submission)
            if d["score"] < min_score:
                continue
            results.append(d)
        time.sleep(0.4)  # be gentle

    if return_strings:
        return [r["text"] for r in results if r.get("text")]
    return results


def _cli():
    p = argparse.ArgumentParser(description="Collect Reddit posts (inline-credential version).")
    p.add_argument("--subreddits", type=str, default="stocks", help="Comma-separated, e.g., 'stocks,investing'")
    p.add_argument("--query", type=str, default=None, help="Optional search query")
    p.add_argument("--limit", type=int, default=50, help="Total posts across subreddits")
    p.add_argument("--time_filter", type=str, default="week", choices=["all","day","hour","month","week","year"])
    p.add_argument("--sort", type=str, default="relevance", choices=["relevance","hot","top","new","comments"])
    p.add_argument("--min_score", type=int, default=0)
    p.add_argument("--return_strings", action="store_true", help="Return list[str] instead of list[dict]")
    args = p.parse_args()

    out = get_reddit_posts(
        subreddits=args.subreddits,
        query=args.query,
        limit=args.limit,
        time_filter=args.time_filter,
        sort=args.sort,
        min_score=args.min_score,
        return_strings=args.return_strings or True,  # default True
    )
    print(f"Collected {len(out)} items")
    if out:
        # preview
        print(out[0][:500] if isinstance(out[0], str) else out[0])


if __name__ == "__main__":
    _cli()
