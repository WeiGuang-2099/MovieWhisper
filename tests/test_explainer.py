import pytest
from src.explainer import Explainer


def test_generate_text_explanation_hybrid():
    exp = Explainer()
    rec = {
        "title": "Star Wars",
        "source": "hybrid",
        "reason": "5 位与你口味相似的用户推荐; 与你喜欢的《Toy Story》风格相似",
        "score": 0.85,
        "cf_score": 0.8,
        "cb_score": 0.9,
        "similar_users_count": 5,
        "similar_to_title": "Toy Story",
        "genres": "Sci-Fi/Action",
    }
    text = exp.generate_text(rec)
    assert "Star Wars" in text
    assert len(text) > 20


def test_generate_text_explanation_collaborative():
    exp = Explainer()
    rec = {
        "title": "Star Wars",
        "source": "collaborative",
        "reason": "5 位与你口味相似的用户推荐",
        "score": 0.85,
        "cf_score": 0.85,
        "cb_score": 0,
        "similar_users_count": 5,
        "similar_to_title": "",
        "genres": "Sci-Fi",
    }
    text = exp.generate_text(rec)
    assert "口味相似" in text


def test_generate_source_label():
    exp = Explainer()
    assert exp.generate_source_label("hybrid") == "hybrid"
    assert exp.generate_source_label("collaborative") == "collaborative"
    assert exp.generate_source_label("content") == "content"
