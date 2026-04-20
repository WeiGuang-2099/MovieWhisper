import pandas as pd
import pytest
from src.collaborative import CollaborativeFilter


@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "movie_id": [1, 2, 3, 1, 2, 4, 1, 3, 4],
        "rating": [5, 4, 3, 5, 4, 2, 2, 1, 5],
    })


def test_fit_creates_matrix(sample_ratings):
    cf = CollaborativeFilter()
    cf.fit(sample_ratings)
    assert cf.user_item_matrix is not None


def test_find_similar_users(sample_ratings):
    cf = CollaborativeFilter()
    cf.fit(sample_ratings)
    similar = cf.find_similar_users(user_id=1, top_k=2)
    assert len(similar) <= 2
    assert isinstance(similar, list)
    if len(similar) >= 2:
        user_ids = [s[0] for s in similar]
        assert 2 in user_ids


def test_recommend(sample_ratings):
    cf = CollaborativeFilter()
    cf.fit(sample_ratings)
    recs = cf.recommend(user_id=1, top_k=5)
    assert isinstance(recs, list)
    rec_ids = [r["movie_id"] for r in recs]
    assert 1 not in rec_ids
    assert 2 not in rec_ids
    assert 3 not in rec_ids


def test_recommend_returns_score(sample_ratings):
    cf = CollaborativeFilter()
    cf.fit(sample_ratings)
    recs = cf.recommend(user_id=1, top_k=5)
    for rec in recs:
        assert "movie_id" in rec
        assert "score" in rec
        assert rec["score"] > 0
