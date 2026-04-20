"""MovieLens 100K Data Exploration"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_ratings, load_movies, load_users

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
