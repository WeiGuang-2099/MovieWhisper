import pandas as pd
import pytest
from src.user_profile import UserProfileBuilder


@pytest.fixture
def sample_data():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2],
        "movie_id": [1, 2, 3, 1, 3],
        "rating": [5, 3, 4, 2, 5],
    })
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["Toy Story", "GoldenEye", "Four Rooms"],
        "genre_Action": [0, 1, 0],
        "genre_Comedy": [1, 0, 1],
        "genre_Drama": [0, 0, 0],
    })
    return ratings, movies


def test_build_genre_preferences(sample_data):
    ratings, movies = sample_data
    builder = UserProfileBuilder(ratings, movies)
    prefs = builder.build_genre_preferences(user_id=1)
    assert isinstance(prefs, dict)
    assert "genre_Action" in prefs
    assert "genre_Comedy" in prefs


def test_genre_preference_weighted_by_rating(sample_data):
    ratings, movies = sample_data
    builder = UserProfileBuilder(ratings, movies)
    prefs = builder.build_genre_preferences(user_id=1)
    assert prefs["genre_Comedy"] > prefs["genre_Drama"]


def test_get_rated_movies(sample_data):
    ratings, movies = sample_data
    builder = UserProfileBuilder(ratings, movies)
    rated = builder.get_rated_movies(user_id=1)
    assert len(rated) == 3
    assert 1 in rated
    assert 2 in rated
    assert 3 in rated


def test_get_rated_movies_other_user(sample_data):
    ratings, movies = sample_data
    builder = UserProfileBuilder(ratings, movies)
    rated = builder.get_rated_movies(user_id=2)
    assert len(rated) == 2


def test_get_rating_stats(sample_data):
    ratings, movies = sample_data
    builder = UserProfileBuilder(ratings, movies)
    stats = builder.get_rating_stats(user_id=1)
    assert stats["count"] == 3
    assert stats["mean"] == 4.0
