import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "movie-recommender"

existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:

    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Pinecone index ready!")

index = pc.Index(index_name)


print("Loading embedding model...")

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

print("Model loaded!")

movies = pd.read_csv("data/movies.csv")

movies['content'] = (
    movies['title']
    + " "
    + movies['genres']
)

print("Generating embeddings...")

embeddings = model.encode(
    movies['content'].tolist(),
    show_progress_bar=True
)

print("Embeddings generated!")



vectors = []

for i, row in movies.iterrows():

    vectors.append(
        (
            str(row['movieId']),
            embeddings[i].tolist(),
            {
                "title": row['title'],
                "genres": row['genres']
            }
        )
    )

print("Uploading vectors to Pinecone...")

batch_size = 100

for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"Uploaded batch {i // batch_size + 1}")

print("Vectors uploaded successfully!")



query_movie = "Toy Story (1995)"


query_embedding = model.encode(query_movie).tolist()


results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

print("\nMovies similar to:", query_movie)

for match in results['matches']:

    print(match['metadata']['title'])

def search_similar_movies_pinecone(movie_title, top_n=5):
    query_embedding = model.encode(movie_title).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_n + 1,
        include_metadata=True
    )

    movies_found = []

    for match in results["matches"]:
        title = match["metadata"]["title"]

        if title != movie_title:
            movies_found.append({
                "title": title,
                "genres": match["metadata"]["genres"],
                "score": match["score"]
            })

        if len(movies_found) == top_n:
            break

    return movies_found
