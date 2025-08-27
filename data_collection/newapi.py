"""
newsapi.py

Fetches articles from NewsAPI.org and returns them in the normalized schema:

- title (str)        REQUIRED
- description (str)  OPTIONAL ("" if missing)
- source (str)       REQUIRED (plain string)
- published_at (str) REQUIRED ISO8601 UTC (with trailing 'Z')
- url (str)          REQUIRED
"""

from newsapi import NewsApiClient
from datetime import datetime, timedelta, timezone
import os
from typing import List, Dict

def _normalize_article(article: Dict) -> Dict:
    """Convert NewsAPI article into required schema."""
    title = article.get("title")
    url = article.get("url")
    source = article.get("source", {}).get("name")
    published_at = article.get("publishedAt")

    # Normalize description
    description = article.get("description") or ""

    # Normalize publishedAt to ISO8601 UTC Z
    if published_at:
        try:
            dt_obj = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            dt_obj = dt_obj.astimezone(timezone.utc).replace(microsecond=0)
            published_at = dt_obj.isoformat().replace("+00:00", "Z")
        except Exception:
            published_at = None

    if not all([title, url, source, published_at]):
        return None

    return {
        "title": title.strip(),
        "description": description.strip(),
        "source": source.strip(),
        "published_at": published_at,
        "url": url.strip(),
    }

def fetch_news(
    api_key: str,
    query: str = "NVIDIA",
    days_back: int = 1,
    total_articles: int = 50,
    language: str = "en",
    sort_by: str = "publishedAt"
) -> List[Dict]:
    """
    Fetch articles from NewsAPI and return them in the normalized schema.
    """
    newsapi = NewsApiClient(api_key=api_key)

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    all_articles = []
    page = 1
    while len(all_articles) < total_articles:
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by=sort_by,
            page_size=min(100, total_articles - len(all_articles)),
            page=page,
        )
        raw_articles = response.get("articles", [])
        if not raw_articles:
            break

        for a in raw_articles:
            norm = _normalize_article(a)
            if norm:
                all_articles.append(norm)

        page += 1

    return all_articles[:total_articles]


if __name__ == "__main__":
    # Example run (API key from env var)
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("Set NEWSAPI_KEY environment variable or pass api_key directly")

    articles = fetch_news(api_key, query="NVIDIA", total_articles=5)
    for art in articles:
        print("📰", art["published_at"], "-", art["title"])
