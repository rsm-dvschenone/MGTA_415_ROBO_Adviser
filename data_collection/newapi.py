api_key = '0099e62b0d23445e8be391a4e96b4810'

# newsapi.py
#
# Fetches articles from NewsAPI.org and returns them in the normalized schema:
#
# OUTPUT SHAPE (articles): list[dict] with:
# - title (str)        REQUIRED
# - description (str)  OPTIONAL ("" if missing)
# - source (str)       REQUIRED (plain string)
# - published_at (str) REQUIRED ISO8601 UTC (with trailing 'Z')
# - url (str)          REQUIRED
#
# Dependencies:
# - pip install newsapi-python
# - Set NEWSAPI_KEY (for the collect() convenience wrapper)

from newsapi import NewsApiClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import os


def _normalize_article(article: Dict) -> Optional[Dict]:
    """Convert a NewsAPI article dict into the required schema."""
    title = article.get("title")
    url = article.get("url")
    source = (article.get("source") or {}).get("name")
    published_at = article.get("publishedAt")

    # Optional description
    description = article.get("description") or ""

    # Normalize publishedAt to ISO8601 UTC Z
    if published_at:
        try:
            dt_obj = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            dt_obj = dt_obj.astimezone(timezone.utc).replace(microsecond=0)
            published_at = dt_obj.isoformat().replace("+00:00", "Z")
        except Exception:
            published_at = None

    # Validate required fields
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
    sort_by: str = "publishedAt",
) -> List[Dict]:
    """
    Fetch articles from NewsAPI and return them in the normalized schema.
    """
    newsapi = NewsApiClient(api_key=api_key)

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    results: List[Dict] = []
    page = 1

    while len(results) < total_articles:
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by=sort_by,
            page_size=min(100, total_articles - len(results)),
            page=page,
        )
        raw_articles = response.get("articles", [])
        if not raw_articles:
            break

        for a in raw_articles:
            norm = _normalize_article(a)
            if norm:
                results.append(norm)

        page += 1

    return results[:total_articles]


def collect(
    query: str = "NVIDIA",
    days_back: int = 1,
    total_articles: int = 50,
    language: str = "en",
    sort_by: str = "publishedAt",
) -> List[Dict]:
    """
    Convenience wrapper that reads NEWSAPI_KEY from the environment.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("Missing NEWSAPI_KEY environment variable.")
    return fetch_news(
        api_key=api_key,
        query=query,
        days_back=days_back,
        total_articles=total_articles,
        language=language,
        sort_by=sort_by,
    )


if __name__ == "__main__":
    # Example run
    arts = collect(query="NVIDIA", total_articles=5)
    for a in arts:
        print(a["published_at"], "-", a["title"])
