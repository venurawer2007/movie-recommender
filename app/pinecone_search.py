import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("movie-recommender")
model = SentenceTransformer("all-MiniLM-L6-v2")
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