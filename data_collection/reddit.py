"""
reddit_posts.py

Fetch Reddit posts from specified subreddits and return them in the schema:

OUTPUT SHAPE (posts):
- list[dict] with:
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
"""

import os
import praw
from typing import List, Dict


def init_reddit_client():
    """Initialize Reddit client using environment variables."""
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "nvda-bot"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
    )


def fetch_reddit_posts(
    reddit,
    query: str = "NVIDIA",
    subreddits: List[str] = ["stocks", "investing", "wallstreetbets"],
    limit: int = 25
) -> List[Dict]:
    """Fetch posts from Reddit and normalize into schema."""
    posts: List[Dict] = []
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.search(query, sort="new", time_filter="day", limit=limit):
            text = f"{submission.title} {submission.selftext}".strip()
            if text:
                posts.append({
                    "text": text,
                    "score": 0  # placeholder for sentiment score
                })
    return posts


if __name__ == "__main__":
    reddit = init_reddit_client()
    posts = fetch_reddit_posts(reddit, query="NVIDIA", limit=10)
    for p in posts:
        print(p)
