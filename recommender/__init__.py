"""A music recommender for one person: you. docs/4-the-recommender.md explains how it works.

From a notebook or script started in the project folder:

    from recommender.profile import build_profile, load_profile
    from recommender.recommend import recommend_songs, similar_songs

    recommend_songs(top_n=10)
    similar_songs("a song title from your history")
"""
