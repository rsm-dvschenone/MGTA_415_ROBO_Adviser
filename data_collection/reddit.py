#!/usr/bin/env python3
"""
reddit_posts.py

Fetch Reddit posts from specified subreddits and return them in the schema:

OUTPUT (posts): list[dict] with:
- text (str)   REQUIRED
- score (int)  OPTIONAL (default 0)

Example:
[{"text": "NVDA lifts guidance...", "score": 0}]

Environment variables required:
- REDDIT_CLIENT_ID
- REDDIT_CLIENT_SECRET
- REDDIT_USER_AGENT
- REDDIT_USERNAME
- REDDIT_PASSWORD

Dependency:
- pip install praw
"""

import os
from typing import List, Dict, Iterable, Optional

try:
    import praw
except ImportError as e:
    raise SystemExit("Missing dependency: praw. Install with `pip install praw`.") from e


def init_reddit_client_from_env():
    """Initialize Reddit client using environment variables."""
    required = {
        "REDDIT_CLIENT_ID": os.getenv("REDDIT_CLIENT_ID"),
        "REDDIT_CLIENT_SECRET": os.getenv("REDDIT_CLIENT_SECRET"),
        "REDDIT_USER_AGENT": os.getenv("REDDIT_USER_AGENT", "nvda-bot"),
        "REDDIT_USERNAME": os.getenv("REDDIT_USERNAME"),
        "REDDIT_PASSWORD": os.getenv("REDDIT_PASSWORD"),
    }
    missing = [k for k, v in required.items() if not v and k != "REDDIT_USER_AGENT"]
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))

    return praw.Reddit(
        client_id=required["REDDIT_CLIENT_ID"],
        client_secret=required["REDDIT_CLIENT_SECRET"],
        user_agent=required["REDDIT_USER_AGENT"] or "nvda-bot",
        username=required["REDDIT_USERNAME"],
        password=required["REDDIT_PASSWORD"],
    )


def fetch_reddit_posts(
    reddit,
    query: str = "NVIDIA",
    subreddits: Iterable[str] = ("stocks", "investing", "wallstreetbets"),
    limit: int = 25,
    time_filter: str = "day",  # hour, day, week, month, year, all
    sort: str = "new",
) -> List[Dict]:
    """
    Search each subreddit and normalize to required schema.
    Returns: [{"text": str, "score": 0}, ...]
    """
    posts: List[Dict] = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.search(query, sort=sort, time_filter=time_filter, limit=limit):
            title = (submission.title or "").strip()
            body = (getattr(submission, "selftext", "") or "").strip()
            text = f"{title} {body}".strip()
            if text and text.lower() not in {"[removed]", "[deleted]"}:
                posts.append({"text": text, "score": 0})
    return posts


def collect(
    query: str = "NVIDIA",
    subreddits: Optional[Iterable[str]] = None,
    limit: int = 25,
    time_filter: str = "day",
    sort: str = "new",
) -> List[Dict]:
    """
    Pipeline-friendly entrypoint. Uses env vars, returns list[dict] with {"text", "score"}.
    """
    if subreddits is None:
        subreddits = ("stocks", "investing", "wallstreetbets")
    reddit = init_reddit_client_from_env()
    return fetch_reddit_posts(reddit, query=query, subreddits=subreddits, limit=limit, time_filter=time_filter, sort=sort)


if __name__ == "__main__":
    import json
    data = collect(query="NVIDIA", subreddits=("stocks", "investing", "wallstreetbets"), limit=10)
    print(json.dumps(data, ensure_ascii=False, indent=2))
