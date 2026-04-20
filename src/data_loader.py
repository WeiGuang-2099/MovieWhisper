import pandas as pd


GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western", "Unknown"
]


def load_ratings(path: str) -> pd.DataFrame:
    """Load MovieLens ratings data."""
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return df


def load_movies(path: str) -> pd.DataFrame:
    """Load MovieLens movie data with genre flags."""
    genre_cols = [f"genre_{g}" for g in GENRE_NAMES]
    col_names = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_cols

    df = pd.read_csv(
        path,
        sep="|",
        names=col_names,
        encoding="latin-1",
        usecols=list(range(len(col_names))),
    )
    df = df.drop(columns=["video_release_date", "imdb_url"], errors="ignore")
    return df


def load_users(path: str) -> pd.DataFrame:
    """Load MovieLens user data."""
    df = pd.read_csv(
        path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    return df


def load_genres(path: str) -> list[str]:
    """Load genre names from genre file."""
    df = pd.read_csv(path, sep="|", names=["genre", "id"])
    return df["genre"].dropna().tolist()
