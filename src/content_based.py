import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    """Content-based recommender using movie genre features."""

    def __init__(self, movies: pd.DataFrame):
        self.movies = movies
        self.genre_cols = [c for c in movies.columns if c.startswith("genre_")]
        self.feature_matrix = None
        self.movie_id_to_idx = None

    def fit(self):
        """Build genre feature matrix and precompute movie-movie similarity."""
        features = self.movies[self.genre_cols].values
        self.feature_matrix = features
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.movies["movie_id"].tolist())
        }
        # Precompute all pairwise movie similarities (1682x1682, fast once)
        self.movie_similarity_matrix = cosine_similarity(features)

    def recommend(
        self, user_ratings: pd.DataFrame, top_k: int = 10
    ) -> list[dict]:
        """Recommend movies similar to user's highly-rated movies."""
        if self.feature_matrix is None:
            raise RuntimeError("Call fit() before recommend()")

        rated_movie_ids = user_ratings["movie_id"].tolist()
        ratings_values = user_ratings["rating"].values

        user_profile = np.zeros(self.feature_matrix.shape[1])
        total_weight = 0
        for mid, rating in zip(rated_movie_ids, ratings_values):
            if mid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[mid]
                user_profile += self.feature_matrix[idx] * rating
                total_weight += rating

        if total_weight > 0:
            user_profile = user_profile / total_weight

        user_profile_2d = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile_2d, self.feature_matrix)[0]

        rated_set = set(rated_movie_ids)
        rated_indices = [self.movie_id_to_idx[mid] for mid in rated_movie_ids if mid in self.movie_id_to_idx]

        recommendations = []
        movie_ids = self.movies["movie_id"].tolist()
        for idx, sim_score in enumerate(similarities):
            mid = movie_ids[idx]
            if mid not in rated_set and sim_score > 0:
                # Use precomputed matrix to find most similar rated movie
                if rated_indices:
                    pair_sims = self.movie_similarity_matrix[idx, rated_indices]
                    best_rated_idx = rated_indices[np.argmax(pair_sims)]
                    similar_title = self.movies.iloc[best_rated_idx]["title"]
                else:
                    similar_title = ""

                recommendations.append({
                    "movie_id": int(mid),
                    "score": round(float(sim_score), 4),
                    "similar_to": similar_title,
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]
