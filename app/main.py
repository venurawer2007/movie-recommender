from fastapi import FastAPI
from app.recommender import recommend_movies, recommend_similar_movies

app = FastAPI(title="Real-Time Movie Recommendation System")


@app.get("/")
def home():
    return {"message": "Movie Recommendation API is running"}


@app.get("/recommend/user/{user_id}")
def get_user_recommendations(user_id: int):
    results = recommend_movies(user_id)
    return results.to_dict(orient="records")


@app.get("/recommend/movie/{movie_id}")
def get_movie_recommendations(movie_id: int):
    results = recommend_similar_movies(movie_id)
    return results.to_dict(orient="records")