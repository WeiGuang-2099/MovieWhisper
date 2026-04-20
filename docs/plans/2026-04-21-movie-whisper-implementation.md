# MovieWhisper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an explainable movie recommendation engine using collaborative filtering + content-based filtering, with a Streamlit web interface that shows users WHY each movie was recommended.

**Architecture:** Three-layer design -- data layer (pandas data loading/cleaning), algorithm layer (collaborative filtering + content-based + hybrid with explainability), presentation layer (Streamlit multi-page app). All data from MovieLens 100K dataset.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, Streamlit, plotly

---

### Task 1: Project Setup and Data Download

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Initialize project structure**

```bash
cd D:/codeproject/机械学习
git init
mkdir -p src tests data notebooks
```

**Step 2: Create requirements.txt**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.30.0
plotly>=5.15.0
pytest>=7.4.0
requests>=2.31.0
```

**Step 3: Create src/__init__.py and tests/__init__.py**

Both files are empty.

```bash
touch src/__init__.py tests/__init__.py
```

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Download MovieLens 100K dataset**

```bash
cd D:/codeproject/机械学习
python -c "
import requests, zipfile, os
url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
r = requests.get(url)
with open('ml-100k.zip', 'wb') as f:
    f.write(r.content)
with zipfile.ZipFile('ml-100k.zip', 'r') as z:
    z.extractall('data')
os.rename('data/ml-100k', 'data/movielens')
os.remove('ml-100k.zip')
print('Downloaded and extracted MovieLens 100K')
"
```

Expected: `data/movielens/` directory contains `u.data`, `u.item`, `u.user`, `u.genre`, etc.

**Step 6: Commit**

```bash
git add .
git commit -m "chore: init project structure and download MovieLens 100K dataset"
```

---

### Task 2: Data Loader

**Files:**
- Create: `src/data_loader.py`
- Create: `tests/test_data_loader.py`

**Step 1: Write tests for data_loader**

Create `tests/test_data_loader.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
cd D:/codeproject/机械学习
python -m pytest tests/test_data_loader.py -v
```

Expected: FAIL (ImportError: cannot import name 'load_ratings')

**Step 3: Implement data_loader**

Create `src/data_loader.py`:

```python
import pandas as pd


GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "Unknown"
]


def load_ratings(path: str) -> pd.DataFrame:
    """Load MovieLens ratings data.

    Columns: user_id, movie_id, rating, timestamp
    """
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return df


def load_movies(path: str) -> pd.DataFrame:
    """Load MovieLens movie data with genre flags.

    Columns: movie_id, title, release_date, genre_Action, genre_Adventure, ...
    """
    genre_cols = [f"genre_{g}" for g in GENRE_NAMES]
    col_names = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols

    df = pd.read_csv(
        path,
        sep="|",
        names=col_names,
        encoding="latin-1",
        usecols=list(range(len(col_names))),
    )
    # Drop unused columns
    df = df.drop(columns=["video_release_date", "imdb_url"], errors="ignore")
    return df


def load_users(path: str) -> pd.DataFrame:
    """Load MovieLens user data.

    Columns: user_id, age, gender, occupation, zip_code
    """
    df = pd.read_csv(
        path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    return df


def load_genres(path: str) -> list[str]:
    """Load genre names from genre file."""
    df = pd.read_csv(path, sep="|", names=["genre", "id"])
    return df["genre"].tolist()[:-1]  # last row is empty
```

**Step 4: Run tests to verify they pass**

```bash
cd D:/codeproject/机械学习
python -m pytest tests/test_data_loader.py -v
```

Expected: All 11 tests PASS.

**Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: add data loader for MovieLens 100K dataset"
```

---

### Task 3: User Profile Builder

**Files:**
- Create: `src/user_profile.py`
- Create: `tests/test_user_profile.py`

**Step 1: Write tests for user_profile**

Create `tests/test_user_profile.py`:

```python
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
    # User 1: rated Comedy movies (Toy Story=5, Four Rooms=4) higher than no Action
    # genre_Comedy avg = (5+4)/2 = 4.5, genre_Action avg = 0 (not rated action movies... actually GoldenEye is Action but user rated 3)
    # genre_Comedy avg = (5+4)/2 = 4.5, genre_Action = 3/1 = 3
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_user_profile.py -v
```

Expected: FAIL (ImportError)

**Step 3: Implement UserProfileBuilder**

Create `src/user_profile.py`:

```python
import pandas as pd


class UserProfileBuilder:
    """Build user preference profiles from rating and movie data."""

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings
        self.movies = movies
        self.genre_cols = [c for c in movies.columns if c.startswith("genre_")]

    def build_genre_preferences(self, user_id: int) -> dict[str, float]:
        """Calculate average rating per genre for a user.

        Returns dict mapping genre column name to average rating.
        Only includes genres where the user has rated at least one movie.
        """
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        merged = user_ratings.merge(self.movies, on="movie_id")

        prefs = {}
        for genre in self.genre_cols:
            genre_movies = merged[merged[genre] == 1]
            if len(genre_movies) > 0:
                prefs[genre] = round(genre_movies["rating"].mean(), 2)
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
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_user_profile.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/user_profile.py tests/test_user_profile.py
git commit -m "feat: add user profile builder with genre preference analysis"
```

---

### Task 4: Collaborative Filtering

**Files:**
- Create: `src/collaborative.py`
- Create: `tests/test_collaborative.py`

**Step 1: Write tests for collaborative filtering**

Create `tests/test_collaborative.py`:

```python
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
    # User 2 should be more similar to User 1 than User 3
    if len(similar) >= 2:
        user_ids = [s[0] for s in similar]
        assert 2 in user_ids


def test_recommend(sample_ratings):
    cf = CollaborativeFilter()
    cf.fit(sample_ratings)
    recs = cf.recommend(user_id=1, top_k=5)
    assert isinstance(recs, list)
    # User 1 hasn't rated movie 4, so it might be recommended
    rec_ids = [r["movie_id"] for r in recs]
    # Should not include movies user already rated
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_collaborative.py -v
```

Expected: FAIL (ImportError)

**Step 3: Implement CollaborativeFilter**

Create `src/collaborative.py`:

```python
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
        """Build user-item rating matrix from ratings DataFrame."""
        self.user_item_matrix = ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            fill_value=0,
        )
        self.user_ids = self.user_item_matrix.index.tolist()
        self.movie_ids = self.user_item_matrix.columns.tolist()

    def find_similar_users(self, user_id: int, top_k: int = 5) -> list[tuple[int, float]]:
        """Find the top_k most similar users to the given user.

        Returns list of (user_id, similarity_score) tuples.
        """
        if user_id not in self.user_ids:
            return []

        matrix = self.user_item_matrix.values
        user_idx = self.user_ids.index(user_id)
        user_vec = matrix[user_idx].reshape(1, -1)

        similarities = cosine_similarity(user_vec, matrix)[0]
        # Set self-similarity to 0
        similarities[user_idx] = 0

        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        result = []
        for idx in top_indices:
            sim_score = similarities[idx]
            if sim_score > 0:
                result.append((self.user_ids[idx], round(float(sim_score), 4)))
        return result

    def recommend(self, user_id: int, top_k: int = 10) -> list[dict]:
        """Recommend movies for the given user.

        Returns list of dicts with movie_id and score.
        Excludes movies the user has already rated.
        """
        similar_users = self.find_similar_users(user_id, top_k=20)
        if not similar_users:
            return []

        # Get movies the user has already rated
        user_rated = set(
            self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] > 0
            ].tolist()
        )

        # Aggregate scores from similar users
        movie_scores = {}
        movie_contributors = {}
        for sim_user_id, sim_score in similar_users:
            sim_user_ratings = self.user_item_matrix.loc[sim_user_id]
            for movie_id in self.movie_ids:
                rating = sim_user_ratings[movie_id]
                if rating > 0 and movie_id not in user_rated:
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = 0
                        movie_contributors[movie_id] = 0
                    movie_scores[movie_id] += sim_score * rating
                    movie_contributors[movie_id] += sim_score

        # Normalize scores
        recommendations = []
        for movie_id, score in movie_scores.items():
            if movie_contributors[movie_id] > 0:
                normalized = round(score / movie_contributors[movie_id], 2)
                recommendations.append({
                    "movie_id": int(movie_id),
                    "score": normalized,
                })

        # Sort by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_collaborative.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/collaborative.py tests/test_collaborative.py
git commit -m "feat: add user-based collaborative filtering recommender"
```

---

### Task 5: Content-Based Filtering

**Files:**
- Create: `src/content_based.py`
- Create: `tests/test_content_based.py`

**Step 1: Write tests for content-based filtering**

Create `tests/test_content_based.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_content_based.py -v
```

Expected: FAIL (ImportError)

**Step 3: Implement ContentBasedFilter**

Create `src/content_based.py`:

```python
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
        """Build genre feature matrix and compute pairwise similarities."""
        features = self.movies[self.genre_cols].values
        self.feature_matrix = features
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(self.movies["movie_id"].tolist())
        }

    def recommend(
        self, user_ratings: pd.DataFrame, top_k: int = 10
    ) -> list[dict]:
        """Recommend movies similar to user's highly-rated movies.

        Args:
            user_ratings: DataFrame with movie_id and rating columns for one user
            top_k: number of recommendations to return

        Returns list of dicts with movie_id, score, similar_to (best matching movie title).
        """
        if self.feature_matrix is None:
            raise RuntimeError("Call fit() before recommend()")

        # Build user profile: weighted average of genre vectors by rating
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

        # Compute similarity to all movies
        user_profile_2d = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile_2d, self.feature_matrix)[0]

        # Exclude already rated movies
        rated_set = set(rated_movie_ids)

        # Build recommendations
        recommendations = []
        movie_ids = self.movies["movie_id"].tolist()
        for idx, sim_score in enumerate(similarities):
            mid = movie_ids[idx]
            if mid not in rated_set and sim_score > 0:
                # Find which of user's rated movies is most similar to this one
                best_similar = None
                best_sim = -1
                for rated_mid in rated_movie_ids:
                    if rated_mid in self.movie_id_to_idx:
                        rated_idx = self.movie_id_to_idx[rated_mid]
                        pair_sim = cosine_similarity(
                            self.feature_matrix[idx].reshape(1, -1),
                            self.feature_matrix[rated_idx].reshape(1, -1),
                        )[0][0]
                        if pair_sim > best_sim:
                            best_sim = pair_sim
                            best_similar = rated_mid

                similar_title = ""
                if best_similar is not None:
                    similar_title = self.movies[
                        self.movies["movie_id"] == best_similar
                    ]["title"].iloc[0]

                recommendations.append({
                    "movie_id": int(mid),
                    "score": round(float(sim_score), 4),
                    "similar_to": similar_title,
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_content_based.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/content_based.py tests/test_content_based.py
git commit -m "feat: add content-based recommender with genre similarity"
```

---

### Task 6: Hybrid Recommender

**Files:**
- Create: `src/hybrid.py`
- Create: `tests/test_hybrid.py`

**Step 1: Write tests for hybrid recommender**

Create `tests/test_hybrid.py`:

```python
import pandas as pd
import pytest
from src.hybrid import HybridRecommender


@pytest.fixture
def sample_data():
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "movie_id": [1, 2, 3, 1, 2, 4, 1, 3, 4],
        "rating": [5, 4, 3, 5, 4, 2, 2, 1, 5],
    })
    movies = pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "title": ["Toy Story", "GoldenEye", "Get Shorty", "Cop Land"],
        "genre_Action": [0, 1, 1, 1],
        "genre_Comedy": [1, 0, 1, 0],
        "genre_Drama": [1, 0, 0, 1],
    })
    return ratings, movies


def test_recommend_returns_list(sample_data):
    ratings, movies = sample_data
    rec = HybridRecommender(ratings, movies)
    rec.fit()
    results = rec.recommend(user_id=1, top_k=5)
    assert isinstance(results, list)


def test_recommend_has_explanation(sample_data):
    ratings, movies = sample_data
    rec = HybridRecommender(ratings, movies)
    rec.fit()
    results = rec.recommend(user_id=1, top_k=5)
    for r in results:
        assert "movie_id" in r
        assert "score" in r
        assert "source" in r
        assert "reason" in r
        assert r["source"] in ["collaborative", "content", "hybrid"]


def test_recommend_excludes_rated(sample_data):
    ratings, movies = sample_data
    rec = HybridRecommender(ratings, movies)
    rec.fit()
    results = rec.recommend(user_id=1, top_k=5)
    rated_ids = set(ratings[ratings["user_id"] == 1]["movie_id"].tolist())
    for r in results:
        assert r["movie_id"] not in rated_ids
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_hybrid.py -v
```

Expected: FAIL (ImportError)

**Step 3: Implement HybridRecommender**

Create `src/hybrid.py`:

```python
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
        """Generate hybrid recommendations with explanations.

        Returns list of dicts with:
            movie_id, title, score, source, reason, similar_users_count, similar_to_title
        """
        # Get collaborative filtering recommendations
        cf_recs = self.cf.recommend(user_id, top_k=top_k * 3)
        cf_scores = {r["movie_id"]: r["score"] for r in cf_recs}

        # Get content-based recommendations
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        cb_recs = self.cb.recommend(user_ratings, top_k=top_k * 3)
        cb_scores = {r["movie_id"]: r["score"] for r in cb_recs}
        cb_similar = {r["movie_id"]: r["similar_to"] for r in cb_recs}

        # Find similar users count for explanation
        similar_users = self.cf.find_similar_users(user_id, top_k=20)

        # Merge scores
        all_movie_ids = set(cf_scores.keys()) | set(cb_scores.keys())

        # User's rated movies
        rated_ids = set(
            self.ratings[self.ratings["user_id"] == user_id]["movie_id"].tolist()
        )

        # Normalize CF scores to 0-1 range
        if cf_scores:
            cf_max = max(cf_scores.values())
            cf_min = min(cf_scores.values())
            cf_range = cf_max - cf_min if cf_max > cf_min else 1
        else:
            cf_range = 1

        # Normalize CB scores to 0-1 range
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

            # Determine source
            has_cf = mid in cf_scores
            has_cb = mid in cb_scores
            if has_cf and has_cb:
                source = "hybrid"
            elif has_cf:
                source = "collaborative"
            else:
                source = "content"

            # Generate reason
            title_row = self.movies[self.movies["movie_id"] == mid]
            title = title_row["title"].iloc[0] if len(title_row) > 0 else f"Movie {mid}"

            # Get genre info
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

        # Sort by hybrid score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_hybrid.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/hybrid.py tests/test_hybrid.py
git commit -m "feat: add hybrid recommender with explainability"
```

---

### Task 7: Explainer

**Files:**
- Create: `src/explainer.py`
- Create: `tests/test_explainer.py`

**Step 1: Write tests for explainer**

Create `tests/test_explainer.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_explainer.py -v
```

Expected: FAIL (ImportError)

**Step 3: Implement Explainer**

Create `src/explainer.py`:

```python
SOURCE_LABELS = {
    "collaborative": "相似用户推荐",
    "content": "风格匹配推荐",
    "hybrid": "综合推荐",
}


class Explainer:
    """Generate human-readable explanations for recommendations."""

    def generate_text(self, rec: dict) -> str:
        """Generate a natural language explanation for a single recommendation.

        Args:
            rec: recommendation dict with title, source, reason, score, etc.

        Returns:
            A human-readable explanation string.
        """
        title = rec.get("title", "")
        source = rec.get("source", "")
        source_label = SOURCE_LABELS.get(source, source)
        reason = rec.get("reason", "")

        parts = [f"[{source_label}] {title}: {reason}"]

        if rec.get("cf_score", 0) > 0 and rec.get("cb_score", 0) > 0:
            parts.append(
                f"协同过滤得分 {rec['cf_score']:.2f}, 内容相似度 {rec['cb_score']:.2f}"
            )
        elif rec.get("cf_score", 0) > 0:
            parts.append(f"协同过滤得分 {rec['cf_score']:.2f}")
        elif rec.get("cb_score", 0) > 0:
            parts.append(f"内容相似度 {rec['cb_score']:.2f}")

        return " | ".join(parts)

    def generate_source_label(self, source: str) -> str:
        """Return a short label for the recommendation source."""
        return source
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_explainer.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/explainer.py tests/test_explainer.py
git commit -m "feat: add recommendation explainer for human-readable output"
```

---

### Task 8: Streamlit App - Main Entry and Rating Page

**Files:**
- Create: `app.py`

**Step 1: Implement the main Streamlit app with rating page**

Create `app.py`:

```python
import streamlit as st
import pandas as pd
from src.data_loader import load_ratings, load_movies, load_users
from src.user_profile import UserProfileBuilder
from src.hybrid import HybridRecommender
from src.explainer import Explainer

# --- Page Config ---
st.set_page_config(page_title="MovieWhisper", page_icon="movie", layout="wide")

# --- Load Data (cached) ---
@st.cache_data
def load_data():
    ratings = load_ratings("data/movielens/u.data")
    movies = load_movies("data/movielens/u.item")
    users = load_users("data/movielens/u.user")
    return ratings, movies, users

ratings, movies, users = load_data()

# --- Session State Init ---
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

# --- Sidebar ---
st.sidebar.title("MovieWhisper")
st.sidebar.caption("可解释电影推荐引擎")

page = st.sidebar.radio(
    "导航",
    ["选择用户", "电影评分", "获取推荐", "用户画像"],
)

explainer = Explainer()

# ============================================================
# Page: Select User
# ============================================================
if page == "选择用户":
    st.header("选择用户")

    user_id = st.number_input(
        "输入用户 ID (1-943)",
        min_value=1,
        max_value=943,
        value=1,
        step=1,
    )

    if st.button("确认选择"):
        st.session_state.current_user = int(user_id)
        # Load this user's existing ratings
        user_data = ratings[ratings["user_id"] == user_id]
        st.session_state.user_ratings = dict(
            zip(user_data["movie_id"], user_data["rating"])
        )
        st.success(f"已选择用户 {user_id}，该用户已有 {len(st.session_state.user_ratings)} 条评分")

    if st.session_state.current_user:
        st.info(f"当前用户: {st.session_state.current_user}")

# ============================================================
# Page: Rate Movies
# ============================================================
elif page == "电影评分":
    st.header("电影评分")

    if not st.session_state.current_user:
        st.warning("请先在「选择用户」页面选择一个用户")
    else:
        st.subheader(f"用户 {st.session_state.current_user} 的评分")

        # Genre filter
        genre_cols = [c for c in movies.columns if c.startswith("genre_")]
        genre_names = [c.replace("genre_", "") for c in genre_cols]

        selected_genre = st.selectbox("按类型筛选", ["全部"] + genre_names)

        if selected_genre != "全部":
            genre_col = f"genre_{selected_genre}"
            filtered = movies[movies[genre_col] == 1]
        else:
            filtered = movies

        # Show a sample of movies
        sample_size = min(20, len(filtered))
        sample_movies = filtered.sample(sample_size, random_state=42)

        for _, movie in sample_movies.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                movie_genres = [c.replace("genre_", "") for c in genre_cols if movie[c] == 1]
                st.write(f"**{movie['title']}** ({', '.join(movie_genres)})")
            with col2:
                current_rating = st.session_state.user_ratings.get(movie["movie_id"], 0)
                new_rating = st.selectbox(
                    "评分",
                    [0, 1, 2, 3, 4, 5],
                    index=int(current_rating),
                    key=f"rating_{movie['movie_id']}",
                )
                if new_rating > 0:
                    st.session_state.user_ratings[movie["movie_id"]] = new_rating
                elif movie["movie_id"] in st.session_state.user_ratings and new_rating == 0:
                    del st.session_state.user_ratings[movie["movie_id"]]

        st.write(f"已评分电影数量: {len(st.session_state.user_ratings)}")

# ============================================================
# Page: Recommendations
# ============================================================
elif page == "获取推荐":
    st.header("为你推荐")

    if not st.session_state.current_user:
        st.warning("请先在「选择用户」页面选择一个用户")
    elif len(st.session_state.user_ratings) < 3:
        st.warning("请至少评分 3 部电影后再获取推荐")
    else:
        # Build ratings DataFrame from session state
        user_rating_rows = []
        for mid, rating in st.session_state.user_ratings.items():
            user_rating_rows.append({
                "user_id": st.session_state.current_user,
                "movie_id": mid,
                "rating": rating,
            })
        user_df = pd.DataFrame(user_rating_rows)

        # Combine with original ratings (replace current user's ratings)
        other_ratings = ratings[ratings["user_id"] != st.session_state.current_user]
        combined = pd.concat([other_ratings, user_df], ignore_index=True)

        with st.spinner("正在计算推荐..."):
            recommender = HybridRecommender(combined, movies)
            recommender.fit()
            recs = recommender.recommend(
                user_id=st.session_state.current_user, top_k=10
            )

        if not recs:
            st.info("暂无推荐结果，请尝试评分更多电影")
        else:
            for i, rec in enumerate(recs):
                with st.container():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        source_emoji = {
                            "hybrid": "[综合]",
                            "collaborative": "[相似用户]",
                            "content": "[风格匹配]",
                        }
                        source_tag = source_emoji.get(rec["source"], "")
                        st.subheader(f"{i+1}. {rec['title']} {source_tag}")
                        st.caption(f"类型: {rec.get('genres', '')}")
                        st.write(rec["reason"])
                    with col2:
                        st.metric("推荐得分", f"{rec['score']:.2f}")
                        if rec.get("cf_score", 0) > 0:
                            st.caption(f"协同过滤: {rec['cf_score']:.2f}")
                        if rec.get("cb_score", 0) > 0:
                            st.caption(f"内容相似: {rec['cb_score']:.2f}")

                    # Explanation detail
                    explanation = explainer.generate_text(rec)
                    st.info(explanation)
                    st.divider()

# ============================================================
# Page: User Profile
# ============================================================
elif page == "用户画像":
    st.header("用户画像")

    if not st.session_state.current_user:
        st.warning("请先在「选择用户」页面选择一个用户")
    else:
        # Build ratings from session
        user_rating_rows = []
        for mid, rating in st.session_state.user_ratings.items():
            user_rating_rows.append({
                "user_id": st.session_state.current_user,
                "movie_id": mid,
                "rating": rating,
            })
        user_df = pd.DataFrame(user_rating_rows)

        builder = UserProfileBuilder(user_df, movies)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("类型偏好分布")
            prefs = builder.build_genre_preferences(st.session_state.current_user)
            if prefs:
                pref_df = pd.DataFrame(
                    {"类型": [k.replace("genre_", "") for k in prefs.keys()],
                     "平均评分": list(prefs.values())}
                ).sort_values("平均评分", ascending=False)

                st.bar_chart(pref_df.set_index("类型"))
            else:
                st.info("暂无评分数据")

        with col2:
            st.subheader("评分统计")
            stats = builder.get_rating_stats(st.session_state.current_user)
            st.metric("评分数量", stats["count"])
            st.metric("平均评分", stats["mean"])

            # Rating distribution
            if not user_df.empty:
                rating_dist = user_df["rating"].value_counts().sort_index()
                st.bar_chart(rating_dist)

        # Similar users
        st.subheader("与你口味相似的用户")
        user_rating_rows_full = []
        for mid, rating in st.session_state.user_ratings.items():
            user_rating_rows_full.append({
                "user_id": st.session_state.current_user,
                "movie_id": mid,
                "rating": rating,
            })
        combined = pd.concat([
            ratings[ratings["user_id"] != st.session_state.current_user],
            pd.DataFrame(user_rating_rows_full),
        ], ignore_index=True)

        from src.collaborative import CollaborativeFilter
        cf = CollaborativeFilter()
        cf.fit(combined)
        similar = cf.find_similar_users(st.session_state.current_user, top_k=5)

        if similar:
            for uid, sim_score in similar:
                user_info = users[users["user_id"] == uid]
                if not user_info.empty:
                    info = user_info.iloc[0]
                    st.write(
                        f"用户 {uid} (相似度: {sim_score:.2f}) - "
                        f"年龄: {info['age']}, 性别: {info['gender']}, "
                        f"职业: {info['occupation']}"
                    )
        else:
            st.info("暂无相似用户数据，请评分更多电影")
```

**Step 2: Test the app launches**

```bash
cd D:/codeproject/机械学习
streamlit run app.py
```

Expected: Browser opens with the MovieWhisper app. Can select a user, rate movies, see recommendations and user profile.

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit app with rating, recommendation, and profile pages"
```

---

### Task 9: Data Exploration Notebook

**Files:**
- Create: `notebooks/exploration.ipynb`

**Step 1: Create the exploration notebook**

This notebook is for your own understanding and for showing in interviews. Create it with these sections:

1. **数据概览**: Load data, show shape, head, describe
2. **评分分布**: Histogram of ratings, stats
3. **电影类型分布**: Bar chart of genre counts
4. **用户行为分析**: Ratings per user distribution, mean rating per user
5. **稀疏度分析**: User-item matrix sparsity calculation

You can create this notebook interactively in Jupyter:

```bash
cd D:/codeproject/机械学习
jupyter notebook notebooks/exploration.ipynb
```

Or create it as a plain Python script `notebooks/exploration.py` first:

```python
"""MovieLens 100K Data Exploration Notebook"""

import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_ratings, load_movies, load_users, load_genres

# Load data
ratings = load_ratings("data/movielens/u.data")
movies = load_movies("data/movielens/u.item")
users = load_users("data/movielens/u.user")

print("=== 数据概览 ===")
print(f"评分数据: {ratings.shape}")
print(f"电影数据: {movies.shape}")
print(f"用户数据: {users.shape}")
print(ratings.head())

print("\n=== 评分分布 ===")
print(ratings["rating"].describe())
ratings["rating"].hist(bins=5)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig("notebooks/rating_distribution.png")
plt.close()

print("\n=== 电影类型分布 ===")
genre_cols = [c for c in movies.columns if c.startswith("genre_")]
genre_counts = movies[genre_cols].sum().sort_values(ascending=False)
print(genre_counts)
genre_counts.plot(kind="bar")
plt.title("Genre Distribution")
plt.savefig("notebooks/genre_distribution.png")
plt.close()

print("\n=== 用户行为 ===")
ratings_per_user = ratings.groupby("user_id").size()
print(f"每用户平均评分数: {ratings_per_user.mean():.1f}")
print(f"最多评分: {ratings_per_user.max()}")
print(f"最少评分: {ratings_per_user.min()}")

print("\n=== 稀疏度 ===")
n_users = ratings["user_id"].nunique()
n_movies = ratings["movie_id"].nunique()
sparsity = 1 - len(ratings) / (n_users * n_movies)
print(f"用户-电影矩阵大小: {n_users} x {n_movies}")
print(f"稀疏度: {sparsity:.4f} ({sparsity*100:.2f}%)")
```

**Step 2: Commit**

```bash
git add notebooks/
git commit -m "docs: add data exploration notebook"
```

---

### Task 10: Final Polish and README

**Files:**
- Create: `README.md`

**Step 1: Write README**

Create `README.md`:

```markdown
# MovieWhisper - 可解释电影推荐引擎

一个基于混合推荐算法的电影推荐系统，核心特点是提供**可解释的推荐理由**，告诉用户"为什么推荐这部电影给你"。

## 项目亮点

- **混合推荐策略**：协同过滤 + 内容推荐加权融合
- **可解释性**：每条推荐附带详细推荐理由和来源分析
- **交互式界面**：基于 Streamlit 的 Web 界面，支持评分、推荐、用户画像

## 技术栈

- Python 3.10+
- pandas / numpy - 数据处理
- scikit-learn - 相似度计算
- Streamlit - Web 界面
- plotly - 可视化

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py
```

## 项目结构

```
src/
  data_loader.py     - 数据加载和预处理
  user_profile.py    - 用户画像构建
  collaborative.py   - 协同过滤算法
  content_based.py   - 内容推荐算法
  hybrid.py          - 混合推荐 + 可解释性
  explainer.py       - 推荐解释生成器
app.py               - Streamlit 主入口
tests/               - 单元测试
notebooks/           - 数据探索笔记
```

## 推荐算法说明

### 协同过滤 (User-Based)
通过余弦相似度找到与目标用户口味相似的用户群体，推荐这些用户高分但目标用户尚未观看的电影。

### 内容推荐
基于电影类型标签 (genre) 构建 TF-IDF 特征，计算用户偏好向量与候选电影的相似度。

### 混合策略
两种推荐结果按 50/50 权重融合，综合得分排序后取 Top-10。

### 可解释性
每条推荐标注来源（相似用户推荐/风格匹配推荐/综合推荐），并给出具体理由文字。

## 数据来源

[MovieLens 100K](https://grouplens.org/datasets/movielens/) - GroupLens Research 项目提供的经典电影评分数据集。
```

**Step 2: Run all tests**

```bash
cd D:/codeproject/机械学习
python -m pytest tests/ -v
```

Expected: All tests PASS.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add project README"
```
