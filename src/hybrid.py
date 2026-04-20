import pandas as pd
from src.collaborative import CollaborativeFilter
from src.content_based import ContentBasedFilter


class HybridRecommender:
    """Hybrid recommender combining collaborative and content-based filtering.

    Fusion strategy: weighted average (50/50 by default).
    Each recommendation includes explainability metadata.
    """

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, cf_weight: float = 0.5):
        self.ratings = ratings
        self.movies = movies
        self.cf_weight = cf_weight
        self.cb_weight = 1.0 - cf_weight
        self.cf = CollaborativeFilter()
        self.cb = ContentBasedFilter(movies)

    def fit(self):
        """Fit both collaborative and content-based models."""
        self.cf.fit(self.ratings)
        self.cb.fit()

    def recommend(self, user_id: int, top_k: int = 10) -> list[dict]:
        """Generate hybrid recommendations with explanations."""
        cf_recs = self.cf.recommend(user_id, top_k=top_k * 3)
        cf_scores = {r["movie_id"]: r["score"] for r in cf_recs}

        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        cb_recs = self.cb.recommend(user_ratings, top_k=top_k * 3)
        cb_scores = {r["movie_id"]: r["score"] for r in cb_recs}
        cb_similar = {r["movie_id"]: r["similar_to"] for r in cb_recs}

        similar_users = self.cf.find_similar_users(user_id, top_k=20)

        all_movie_ids = set(cf_scores.keys()) | set(cb_scores.keys())
        rated_ids = set(
            self.ratings[self.ratings["user_id"] == user_id]["movie_id"].tolist()
        )

        if cf_scores:
            cf_max = max(cf_scores.values())
            cf_min = min(cf_scores.values())
            cf_range = cf_max - cf_min if cf_max > cf_min else 1
        else:
            cf_range = 1

        if cb_scores:
            cb_max = max(cb_scores.values())
            cb_min = min(cb_scores.values())
            cb_range = cb_max - cb_min if cb_max > cb_min else 1
        else:
            cb_range = 1

        recommendations = []
        for mid in all_movie_ids:
            if mid in rated_ids:
                continue

            cf_score = (cf_scores.get(mid, 0) - cf_min) / cf_range if mid in cf_scores else 0
            cb_score = (cb_scores.get(mid, 0) - cb_min) / cb_range if mid in cb_scores else 0

            hybrid_score = self.cf_weight * cf_score + self.cb_weight * cb_score

            has_cf = mid in cf_scores
            has_cb = mid in cb_scores
            if has_cf and has_cb:
                source = "hybrid"
            elif has_cf:
                source = "collaborative"
            else:
                source = "content"

            title_row = self.movies[self.movies["movie_id"] == mid]
            title = title_row["title"].iloc[0] if len(title_row) > 0 else f"Movie {mid}"

            genre_cols = [c for c in self.movies.columns if c.startswith("genre_")]
            if len(title_row) > 0:
                genres = [c.replace("genre_", "") for c in genre_cols if title_row[c].iloc[0] == 1]
                genre_str = "/".join(genres) if genres else "Unknown"
            else:
                genre_str = "Unknown"

            reason_parts = []
            if has_cf:
                n_similar = len(similar_users)
                reason_parts.append(f"{n_similar} 位与你口味相似的用户推荐了这部电影")
            if has_cb:
                similar_title = cb_similar.get(mid, "")
                if similar_title:
                    reason_parts.append(f"与你喜欢的《{similar_title}》风格相似")
                reason_parts.append(f"类型: {genre_str}")

            recommendations.append({
                "movie_id": mid,
                "title": title,
                "genres": genre_str,
                "score": round(hybrid_score, 4),
                "cf_score": round(cf_score, 4),
                "cb_score": round(cb_score, 4),
                "source": source,
                "reason": "; ".join(reason_parts),
                "similar_users_count": len(similar_users),
                "similar_to_title": cb_similar.get(mid, ""),
            })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]
