import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load movies
movies = pd.read_csv("data/movies.csv")

# Combine title + genres
movies['content'] = movies['title'] + " " + movies['genres']

# Generate embeddings
embeddings = model.encode(
    movies['content'].tolist(),
    show_progress_bar=True
)

def recommend_by_embedding(movie_title, top_n=5):

    # Find movie index
    movie_index = movies[movies['title'] == movie_title].index[0]

    # Get similarity scores
    similarities = cosine_similarity(
        [embeddings[movie_index]],
        embeddings
    )[0]
    similar_indices = similarities.argsort()[::-1][1:top_n+1]

    return movies.iloc[similar_indices][['title', 'genres']]
if __name__ == "__main__":

    movie_name = "Toy Story (1995)"
    print("Movies similar to:", movie_name)
    print(recommend_by_embedding(movie_name))