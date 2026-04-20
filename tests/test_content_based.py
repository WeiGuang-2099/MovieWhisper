import pandas as pd
import pytest
from src.content_based import ContentBasedFilter


@pytest.fixture
def sample_data():
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "title": ["Toy Story", "GoldenEye", "Get Shorty", "Cop Land"],
        "genre_Action": [0, 1, 1, 1],
        "genre_Comedy": [1, 0, 1, 0],
        "genre_Drama": [1, 0, 0, 1],
    })
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1],
        "movie_id": [1, 2, 3],
        "rating": [5, 2, 4],
    })
    return movies, ratings


def test_fit_creates_feature_matrix(sample_data):
    movies, _ = sample_data
    cb = ContentBasedFilter(movies)
    cb.fit()
    assert cb.feature_matrix is not None


def test_recommend_returns_list(sample_data):
    movies, ratings = sample_data
    cb = ContentBasedFilter(movies)
    cb.fit()
    recs = cb.recommend(user_ratings=ratings[ratings["user_id"] == 1], top_k=5)
    assert isinstance(recs, list)


def test_recommend_excludes_rated(sample_data):
    movies, ratings = sample_data
    cb = ContentBasedFilter(movies)
    cb.fit()
    recs = cb.recommend(user_ratings=ratings[ratings["user_id"] == 1], top_k=5)
    rated_ids = set(ratings[ratings["user_id"] == 1]["movie_id"].tolist())
    rec_ids = [r["movie_id"] for r in recs]
    for rid in rec_ids:
        assert rid not in rated_ids


def test_recommend_has_score_and_similar(sample_data):
    movies, ratings = sample_data
    cb = ContentBasedFilter(movies)
    cb.fit()
    recs = cb.recommend(user_ratings=ratings[ratings["user_id"] == 1], top_k=5)
    for rec in recs:
        assert "movie_id" in rec
        assert "score" in rec
        assert "similar_to" in rec
