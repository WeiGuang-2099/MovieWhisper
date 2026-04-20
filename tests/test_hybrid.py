import pandas as pd
import pytest
from src.hybrid import HybridRecommender


@pytest.fixture
def sample_data():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "movie_id": [1, 2, 3, 1, 2, 4, 1, 3, 4],
        "rating": [5, 4, 3, 5, 4, 2, 2, 1, 5],
    })
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "title": ["Toy Story", "GoldenEye", "Get Shorty", "Cop Land"],
        "genre_Action": [0, 1, 1, 1],
        "genre_Comedy": [1, 0, 1, 0],
        "genre_Drama": [1, 0, 0, 1],
    })
    return ratings, movies


def test_recommend_returns_list(sample_data):
    ratings, movies = sample_data
    rec = HybridRecommender(ratings, movies)
    rec.fit()
    results = rec.recommend(user_id=1, top_k=5)
    assert isinstance(results, list)


def test_recommend_has_explanation(sample_data):
    ratings, movies = sample_data
    rec = HybridRecommender(ratings, movies)
    rec.fit()
    results = rec.recommend(user_id=1, top_k=5)
    for r in results:
        assert "movie_id" in r
        assert "score" in r
        assert "source" in r
        assert "reason" in r
        assert r["source"] in ["collaborative", "content", "hybrid"]


def test_recommend_excludes_rated(sample_data):
    ratings, movies = sample_data
    rec = HybridRecommender(ratings, movies)
    rec.fit()
    results = rec.recommend(user_id=1, top_k=5)
    rated_ids = set(ratings[ratings["user_id"] == 1]["movie_id"].tolist())
    for r in results:
        assert r["movie_id"] not in rated_ids
