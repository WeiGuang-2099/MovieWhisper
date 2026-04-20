import streamlit as st
import pandas as pd
from src.data_loader import load_ratings, load_movies, load_users
from src.user_profile import UserProfileBuilder
from src.hybrid import HybridRecommender
from src.explainer import Explainer

st.set_page_config(page_title="MovieWhisper", page_icon="movie", layout="wide")


@st.cache_data
def load_data():
    ratings = load_ratings("data/movielens/u.data")
    movies = load_movies("data/movielens/u.item")
    users = load_users("data/movielens/u.user")
    return ratings, movies, users


@st.cache_data
def get_recommendations(user_id, ratings_hash, _user_ratings, _all_ratings, _movies):
    """Cache recommendations by user_id + ratings snapshot.
    Only recomputes when ratings change. First call is slow, subsequent calls are instant."""
    user_rating_rows = [
        {"user_id": user_id, "movie_id": mid, "rating": rating}
        for mid, rating in _user_ratings.items()
    ]
    other_ratings = _all_ratings[_all_ratings["user_id"] != user_id]
    combined = pd.concat([other_ratings, pd.DataFrame(user_rating_rows)], ignore_index=True)

    recommender = HybridRecommender(combined, _movies)
    recommender.fit()
    return recommender.recommend(user_id=user_id, top_k=10)


@st.cache_data
def get_similar_users(user_id, ratings_hash, _user_ratings, _all_ratings):
    """Cache similar users computation."""
    from src.collaborative import CollaborativeFilter
    user_rating_rows = [
        {"user_id": user_id, "movie_id": mid, "rating": rating}
        for mid, rating in _user_ratings.items()
    ]
    other_ratings = _all_ratings[_all_ratings["user_id"] != user_id]
    combined = pd.concat([other_ratings, pd.DataFrame(user_rating_rows)], ignore_index=True)

    cf = CollaborativeFilter()
    cf.fit(combined)
    return cf.find_similar_users(user_id, top_k=5)


def ratings_hash(user_ratings):
    """Create a hashable key from user ratings dict for caching."""
    return frozenset(user_ratings.items())


ratings, movies, users = load_data()

if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}

st.sidebar.title("MovieWhisper")
st.sidebar.caption("可解释电影推荐引擎")

page = st.sidebar.radio(
    "导航",
    ["选择用户", "电影评分", "获取推荐", "用户画像"],
)

explainer = Explainer()

# Page: Select User
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
        user_data = ratings[ratings["user_id"] == user_id]
        st.session_state.user_ratings = dict(
            zip(user_data["movie_id"], user_data["rating"])
        )
        st.success(f"已选择用户 {user_id}，该用户已有 {len(st.session_state.user_ratings)} 条评分")
    if st.session_state.current_user:
        st.info(f"当前用户: {st.session_state.current_user}")

# Page: Rate Movies
elif page == "电影评分":
    st.header("电影评分")
    if not st.session_state.current_user:
        st.warning("请先在「选择用户」页面选择一个用户")
    else:
        st.subheader(f"用户 {st.session_state.current_user} 的评分")
        genre_cols = [c for c in movies.columns if c.startswith("genre_")]
        genre_names = [c.replace("genre_", "") for c in genre_cols]
        selected_genre = st.selectbox("按类型筛选", ["全部"] + genre_names)

        if selected_genre != "全部":
            genre_col = f"genre_{selected_genre}"
            filtered = movies[movies[genre_col] == 1]
        else:
            filtered = movies

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

# Page: Recommendations
elif page == "获取推荐":
    st.header("为你推荐")
    if not st.session_state.current_user:
        st.warning("请先在「选择用户」页面选择一个用户")
    elif len(st.session_state.user_ratings) < 3:
        st.warning("请至少评分 3 部电影后再获取推荐")
    else:
        r_hash = ratings_hash(st.session_state.user_ratings)
        with st.spinner("正在计算推荐..."):
            recs = get_recommendations(
                st.session_state.current_user,
                r_hash,
                st.session_state.user_ratings,
                ratings,
                movies,
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
                    explanation = explainer.generate_text(rec)
                    st.info(explanation)
                    st.divider()

# Page: User Profile
elif page == "用户画像":
    st.header("用户画像")
    if not st.session_state.current_user:
        st.warning("请先在「选择用户」页面选择一个用户")
    else:
        user_rating_rows = [
            {"user_id": st.session_state.current_user, "movie_id": mid, "rating": rating}
            for mid, rating in st.session_state.user_ratings.items()
        ]
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
            if not user_df.empty:
                rating_dist = user_df["rating"].value_counts().sort_index()
                st.bar_chart(rating_dist)

        st.subheader("与你口味相似的用户")
        r_hash = ratings_hash(st.session_state.user_ratings)
        similar = get_similar_users(
            st.session_state.current_user,
            r_hash,
            st.session_state.user_ratings,
            ratings,
        )

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
