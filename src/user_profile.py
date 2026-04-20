import pandas as pd


class UserProfileBuilder:
    """Build user preference profiles from rating and movie data."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings
        self.movies = movies
        self.genre_cols = [c for c in movies.columns if c.startswith("genre_")]

    def build_genre_preferences(self, user_id: int) -> dict[str, float]:
        """Calculate average rating per genre for a user."""
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        merged = user_ratings.merge(self.movies, on="movie_id")

        prefs = {}
        for genre in self.genre_cols:
            genre_movies = merged[merged[genre] == 1]
            if len(genre_movies) > 0:
                prefs[genre] = round(genre_movies["rating"].mean(), 2)
            else:
                prefs[genre] = 0.0
        return prefs

    def get_rated_movies(self, user_id: int) -> set[int]:
        """Return set of movie IDs rated by the user."""
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        return set(user_ratings["movie_id"].tolist())

    def get_rating_stats(self, user_id: int) -> dict:
        """Return rating statistics for a user."""
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        return {
            "count": len(user_ratings),
            "mean": round(user_ratings["rating"].mean(), 2),
        }
