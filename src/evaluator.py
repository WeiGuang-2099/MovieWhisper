"""Evaluation metrics for recommender systems.

Provides Precision@K, Recall@K, NDCG@K, RMSE, and MAE using
leave-one-out cross-validation on MovieLens 100K.
"""

import pandas as pd
import numpy as np
from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter
from src.hybrid import HybridRecommender


def leave_one_out_split(ratings: pd.DataFrame, seed: int = 42):
    """Split ratings: for each user, hold out their highest-rated movie as test.

    Returns (train_df, test_df).
    """
    rng = np.random.RandomState(seed)

    test_rows = []
    train_rows = []

    for uid, group in ratings.groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        # Pick one random rating as test
        test_idx = rng.choice(group.index)
        test_rows.append(group.loc[[test_idx]])
        train_rows.append(group.drop(test_idx))

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    return train_df, test_df


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for mid in top_k if mid in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for mid in top_k if mid in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if len(relevant) == 0:
        return 0.0
    dcg = 0.0
    for i, mid in enumerate(recommended[:k]):
        if mid in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all relevant items at top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def rmse(predicted: dict[int, float], actual: dict[int, float]) -> float:
    """Root Mean Squared Error between predicted and actual ratings."""
    common = set(predicted.keys()) & set(actual.keys())
    if not common:
        return float("inf")
    errors = [(predicted[mid] - actual[mid]) ** 2 for mid in common]
    return np.sqrt(np.mean(errors))


def mae(predicted: dict[int, float], actual: dict[int, float]) -> float:
    """Mean Absolute Error between predicted and actual ratings."""
    common = set(predicted.keys()) & set(actual.keys())
    if not common:
        return float("inf")
    errors = [abs(predicted[mid] - actual[mid]) for mid in common]
    return np.mean(errors)


def evaluate_recommender(ratings: pd.DataFrame, movies: pd.DataFrame, k: int = 10):
    """Run full evaluation: leave-one-out split, compute all metrics.

    Returns dict with metrics for collaborative, content-based, and hybrid.
    """
    train_df, test_df = leave_one_out_split(ratings)

    # Build test ground truth: user_id -> set of held-out movie_ids
    test_ground_truth = {}
    test_ratings = {}
    for _, row in test_df.iterrows():
        uid = int(row["user_id"])
        mid = int(row["movie_id"])
        test_ground_truth.setdefault(uid, set()).add(mid)
        test_ratings.setdefault(uid, {})[mid] = row["rating"]

    results = {}

    # Evaluate each recommender type
    for name, recommender in [
        ("collaborative", _make_cf(train_df, movies)),
        ("content", _make_cb(train_df, movies)),
        ("hybrid", _make_hybrid(train_df, movies)),
    ]:
        precisions, recalls, ndcgs, hit_rates = [], [], [], []

        test_users = list(test_ground_truth.keys())

        for uid in test_users[:100]:  # Sample 100 users for speed
            relevant = test_ground_truth[uid]

            # Get recommendations (use 3x K for broader coverage in ranking)
            recs = recommender.recommend(user_id=uid, top_k=k)
            rec_ids = [r["movie_id"] for r in recs]

            # Ranking metrics
            precisions.append(precision_at_k(rec_ids, relevant, k))
            recalls.append(recall_at_k(rec_ids, relevant, k))
            ndcgs.append(ndcg_at_k(rec_ids, relevant, k))
            # Hit Rate: did the held-out movie appear in top-K at all?
            hit_rates.append(1.0 if relevant & set(rec_ids) else 0.0)

        # Rating prediction: use CF model directly for RMSE/MAE
        pred_all, actual_all = _compute_rating_predictions(
            recommender.cf, train_df, test_df
        )

        results[name] = {
            "precision@k": round(np.mean(precisions), 4),
            "recall@k": round(np.mean(recalls), 4),
            "ndcg@k": round(np.mean(ndcgs), 4),
            "hit_rate": round(np.mean(hit_rates), 4),
            "rmse": round(rmse(pred_all, actual_all), 4),
            "mae": round(mae(pred_all, actual_all), 4),
        }

    return results


def _compute_rating_predictions(cf, train_df, test_df):
    """Compute predicted vs actual ratings using the CF model's similar users."""
    predicted = {}
    actual = {}

    for _, row in test_df.iterrows():
        uid = int(row["user_id"])
        mid = int(row["movie_id"])
        actual_rating = row["rating"]

        similar = cf.find_similar_users(uid, top_k=20)
        if not similar:
            continue

        # Predict rating: similarity-weighted average of similar users' ratings
        weighted_sum = 0
        sim_sum = 0
        for sim_uid, sim_score in similar:
            if mid in cf.raw_matrix.columns:
                sim_rating = cf.raw_matrix.loc[sim_uid, mid] if sim_uid in cf.raw_matrix.index else 0
                if sim_rating > 0:
                    weighted_sum += sim_score * sim_rating
                    sim_sum += sim_score

        if sim_sum > 0:
            predicted[mid + uid * 10000] = round(weighted_sum / sim_sum, 2)
            actual[mid + uid * 10000] = actual_rating

    return predicted, actual


def _make_cf(train_df, movies):
    """Create and fit a collaborative filter."""
    rec = HybridRecommender(train_df, movies)
    rec.fit()
    return rec


def _make_cb(train_df, movies):
    """Create and fit a content-based filter wrapped as a recommender."""
    rec = HybridRecommender(train_df, movies, cf_weight=0.0)
    rec.fit()
    return rec


def _make_hybrid(train_df, movies):
    """Create and fit a hybrid recommender."""
    rec = HybridRecommender(train_df, movies)
    rec.fit()
    return rec
