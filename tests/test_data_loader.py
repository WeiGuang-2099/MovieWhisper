import pandas as pd
import pytest
from src.data_loader import load_ratings, load_movies, load_users, load_genres


def test_load_ratings_returns_dataframe():
    df = load_ratings("data/movielens/u.data")
    assert isinstance(df, pd.DataFrame)


def test_load_ratings_columns():
    df = load_ratings("data/movielens/u.data")
    assert list(df.columns) == ["user_id", "movie_id", "rating", "timestamp"]


def test_load_ratings_count():
    df = load_ratings("data/movielens/u.data")
    assert len(df) == 100000


def test_load_ratings_range():
    df = load_ratings("data/movielens/u.data")
    assert df["rating"].min() >= 1
    assert df["rating"].max() <= 5


def test_load_movies_returns_dataframe():
    df = load_movies("data/movielens/u.item")
    assert isinstance(df, pd.DataFrame)


def test_load_movies_has_title():
    df = load_movies("data/movielens/u.item")
    assert "title" in df.columns


def test_load_movies_has_genres():
    df = load_movies("data/movielens/u.item")
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    assert len(genre_cols) > 0


def test_load_movies_count():
    df = load_movies("data/movielens/u.item")
    assert len(df) == 1682


def test_load_users_returns_dataframe():
    df = load_users("data/movielens/u.user")
    assert isinstance(df, pd.DataFrame)


def test_load_users_columns():
    df = load_users("data/movielens/u.user")
    assert "user_id" in df.columns
    assert "age" in df.columns


def test_load_genres_returns_list():
    genres = load_genres("data/movielens/u.genre")
    assert isinstance(genres, list)
    assert "Action" in genres
    assert len(genres) == 19
