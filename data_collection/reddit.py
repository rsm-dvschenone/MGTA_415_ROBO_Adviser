
"""
data_collection/reddit.py
-------------------------
Collect Reddit posts via the official Reddit API using PRAW.
Returns a list[dict] with consistent keys for downstream analysis.

Auth:
  Provide env vars (recommended) or a .env file at project root:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    - REDDIT_USER_AGENT   (e.g., "msba-project-bot/0.1 by <your_reddit_username>")

Quick start (CLI):
    python -m data_collection.reddit --subreddits stocks,investing --query nvidia --limit 100
"""
from __future__ import annotations

import os
import time
import argparse
from typing import List, Dict, Any, Optional, Iterable, Union

try:
    from dotenv import load_dotenv  # optional, but nice for local dev
except Exception:
    load_dotenv = None  # type: ignore

try:
    import praw
except Exception as e:
    raise ImportError("praw is required for reddit collection. Try: pip install praw python-dotenv") from e


def _init_reddit():
    # Load .env if available
    if load_dotenv:
        load_dotenv()

    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    uagent = os.getenv("REDDIT_USER_AGENT", "msba-project-bot/0.1")
    if not cid or not csec:
        raise RuntimeError(
            "Missing Reddit API credentials. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET "
            "as environment variables or in a .env file."
        )
    reddit = praw.Reddit(
        client_id=cid,
        client_secret=csec,
        user_agent=uagent,
        check_for_async=False,
    )
    return reddit


def _normalize_submission(sub) -> Dict[str, Any]:
    # PRAW objects have many attributes; we extract the ones we need.
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
    limit: int = 200,
    time_filter: str = "week",   # one of: "all","day","hour","month","week","year"
    sort: str = "relevance",     # "relevance","hot","top","new","comments"
    min_score: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetch posts from one or more subreddits. If `query` is provided, uses Reddit search.
    Otherwise pulls from 'hot' for each subreddit.

    Returns a list of dicts with keys: id, subreddit, title, body, text, score, num_comments,
    created_utc, author, url, permalink.
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
        try:
            if query:
                # Reddit search
                # PRAW search supports limit, sort, time_filter
                it = sub.search(query=query, sort=sort, time_filter=time_filter, limit=per_sub_limit)
            else:
                # Without a query, fall back to hot (or new/top if you prefer)
                it = sub.hot(limit=per_sub_limit)
            for submission in it:
                d = _normalize_submission(submission)
                if d["score"] < min_score:
                    continue
                results.append(d)
        except Exception as e:
            # Basic resilience: continue on subreddit error
            print(f"[WARN] Failed fetching from r/{sub_name}: {e}")
            continue
        # brief pause to be gentle
        time.sleep(0.5)

    return results


def _cli():
    p = argparse.ArgumentParser(description="Collect Reddit posts into a JSONL file or stdout.")
    p.add_argument("--subreddits", type=str, default="stocks", help="Comma-separated list, e.g., 'stocks,investing'")
    p.add_argument("--query", type=str, default=None, help="Optional search query")
    p.add_argument("--limit", type=int, default=200, help="Total posts across subreddits")
    p.add_argument("--time_filter", type=str, default="week", choices=["all","day","hour","month","week","year"])
    p.add_argument("--sort", type=str, default="relevance", choices=["relevance","hot","top","new","comments"])
    p.add_argument("--min_score", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="Optional path to write JSONL")
    args = p.parse_args()

    posts = get_reddit_posts(
        subreddits=args.subreddits,
        query=args.query,
        limit=args.limit,
        time_filter=args.time_filter,
        sort=args.sort,
        min_score=args.min_score,
    )

    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            for row in posts:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(posts)} posts to {args.out}")
    else:
        from pprint import pprint
        print(f"Collected {len(posts)} posts")
        pprint(posts[:3])  # preview


if __name__ == "__main__":
    _cli()
