import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# Compute similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert to DataFrame
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_movies(user_id, top_n=5):
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Remove itself
    similar_users = similar_users.drop(user_id)

    # Get top similar users
    top_users = similar_users.head(10).index

    # Get movies rated by similar users
    similar_users_ratings = ratings[ratings['userId'].isin(top_users)]

    # Get movies already watched by this user
    watched_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    # Remove already watched movies
    similar_users_ratings = similar_users_ratings[
        ~similar_users_ratings['movieId'].isin(watched_movies)
    ]

    # Average ratings
    movie_scores = similar_users_ratings.groupby('movieId')['rating'].mean()

    # Sort movies
    recommended_movie_ids = movie_scores.sort_values(ascending=False).head(top_n).index

    # Return movie details
    return movies[movies['movieId'].isin(recommended_movie_ids)][['movieId', 'title', 'genres']]


def recommend_similar_movies(movie_id, top_n=5):
    # Create movie-user matrix
    movie_user_matrix = ratings.pivot_table(
        index='movieId',
        columns='userId',
        values='rating'
    ).fillna(0)

    # Calculate similarity between movies
    movie_similarity = cosine_similarity(movie_user_matrix)

    # Convert to DataFrame
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    # Get similar movies
    similar_movies = movie_similarity_df[movie_id].sort_values(ascending=False)

    # Remove the same movie
    similar_movies = similar_movies.drop(movie_id)

    # Get top movies
    similar_movie_ids = similar_movies.head(top_n).index

    return movies[movies['movieId'].isin(similar_movie_ids)][['movieId', 'title', 'genres']]


if __name__ == "__main__":
    user_id = 1
    movie_id = 1

    print("Recommended movies for user:", user_id)
    print(recommend_movies(user_id))

    print("\nMovies similar to movie ID:", movie_id)
    print(recommend_similar_movies(movie_id))