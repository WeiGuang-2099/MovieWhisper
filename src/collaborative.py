import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilter:
    """User-based collaborative filtering recommender."""

    def __init__(self):
        self.user_item_matrix = None
        self.user_ids = None
        self.movie_ids = None

    def fit(self, ratings: pd.DataFrame):
        """Build user-item rating matrix from ratings DataFrame.

        Uses mean-centering: subtract each user's mean rating so that
        unrated items stay at 0, and similarity reflects taste deviation
        rather than raw score levels.
        """
        raw_matrix = ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            fill_value=0,
        )
        # Mean-center: for each user, subtract their mean from rated items
        # Unrated items remain 0 (neutral, won't affect cosine similarity)
        user_means = raw_matrix.replace(0, np.nan).mean(axis=1)
        self.user_means = user_means
        centered = raw_matrix.copy()
        for uid in raw_matrix.index:
            rated_mask = raw_matrix.loc[uid] != 0
            centered.loc[uid, rated_mask] = raw_matrix.loc[uid, rated_mask] - user_means[uid]

        self.user_item_matrix = centered
        self.raw_matrix = raw_matrix
        self.user_ids = self.user_item_matrix.index.tolist()
        self.movie_ids = self.user_item_matrix.columns.tolist()

    def find_similar_users(self, user_id: int, top_k: int = 5) -> list[tuple[int, float]]:
        """Find the top_k most similar users to the given user."""
        if user_id not in self.user_ids:
            return []

        matrix = self.user_item_matrix.values
        user_idx = self.user_ids.index(user_id)
        user_vec = matrix[user_idx].reshape(1, -1)

        similarities = cosine_similarity(user_vec, matrix)[0]
        similarities[user_idx] = 0

        top_indices = np.argsort(similarities)[::-1][:top_k]

        result = []
        for idx in top_indices:
            sim_score = similarities[idx]
            if sim_score > 0:
                result.append((self.user_ids[idx], round(float(sim_score), 4)))
        return result

    def recommend(self, user_id: int, top_k: int = 10) -> list[dict]:
        """Recommend movies for the given user."""
        similar_users = self.find_similar_users(user_id, top_k=20)
        if not similar_users:
            return []

        user_rated = set(
            self.raw_matrix.columns[
                self.raw_matrix.loc[user_id] > 0
            ].tolist()
        )

        # Use raw (original) ratings for predicted score, weighted by similarity
        movie_scores = {}
        movie_contributors = {}
        for sim_user_id, sim_score in similar_users:
            sim_user_ratings = self.raw_matrix.loc[sim_user_id]
            for movie_id in self.movie_ids:
                rating = sim_user_ratings[movie_id]
                if rating > 0 and movie_id not in user_rated:
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = 0
                        movie_contributors[movie_id] = 0
                    movie_scores[movie_id] += sim_score * rating
                    movie_contributors[movie_id] += sim_score

        recommendations = []
        for movie_id, score in movie_scores.items():
            if movie_contributors[movie_id] > 0:
                normalized = round(score / movie_contributors[movie_id], 2)
                recommendations.append({
                    "movie_id": int(movie_id),
                    "score": normalized,
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]
