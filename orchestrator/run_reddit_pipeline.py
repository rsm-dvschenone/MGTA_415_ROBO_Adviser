
"""
run_reddit_pipeline.py
----------------------
End-to-end test: collect Reddit posts using data_collection.reddit,
then analyze sentiment using data_analysis.reddit_post_analysis.
"""
from pprint import pprint
from data_collection.reddit import get_reddit_posts
from data_analysis.reddit_post_analysis import analyze_reddit_sentiment

def main():
    # Adjust these parameters for your test
    posts = get_reddit_posts(
        subreddits="stocks,investing",
        query="nvidia",
        limit=60,
        time_filter="week",
        sort="relevance",
        min_score=1,
    )
    res = analyze_reddit_sentiment(posts, filter_keywords=None, use_weight=True)
    print("\n=== Reddit Sentiment Summary ===")
    pprint(res["summary"])

if __name__ == "__main__":
    main()
