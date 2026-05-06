import streamlit as st
import sys
import os
import pandas as pd

# Add root folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from app.recommender import recommend_movies, recommend_similar_movies

# Load movies dataset
movies_df = pd.read_csv("data/movies.csv")

# Page title
st.title("🎬 Real-Time Movie Recommendation System")

st.write("Choose recommendation type:")

# Dropdown
option = st.selectbox(
    "Recommendation Type",
    ["User Recommendations", "Movie Similarity"]
)

# User recommendations
if option == "User Recommendations":

    user_id = st.number_input(
        "Enter User ID",
        min_value=1,
        value=1,
        step=1
    )

    if st.button("Get Recommendations"):

        recommendations = recommend_movies(user_id)

        st.subheader("Recommended Movies")

        st.dataframe(recommendations)

# Movie similarity
elif option == "Movie Similarity":

    # Movie title dropdown
    movie_title = st.selectbox(
        "Select a Movie",
        movies_df['title'].sort_values()
    )

    # Get movie ID
    movie_id = movies_df[
        movies_df['title'] == movie_title
    ]['movieId'].values[0]

    if st.button("Find Similar Movies"):

        recommendations = recommend_similar_movies(movie_id)

        st.subheader("Similar Movies")

        st.dataframe(recommendations)