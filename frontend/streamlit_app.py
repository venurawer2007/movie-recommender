import streamlit as st
import sys
import os
import pandas as pd

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Anton&family=Rajdhani:wght@400;700&display=swap');

/* Main app background */
.stApp {
    background-color: #050816;
    color: white;
}

/* Default text */
html, body, [class*="css"]  {
    font-family: 'Rajdhani', sans-serif;
    color: white;
}

/* Main title */
.main-title {
    font-family: 'Anton', sans-serif;
    text-align: center;
    font-size: 65px;
    letter-spacing: 4px;
    margin-bottom: 20px;
    text-transform: uppercase;
    color: white;
}

/* Section headers */
h2, h3 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    color: white;
}

/* Labels */
label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 20px !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 22px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 10px !important;
    border: 2px solid white !important;
    background-color: transparent !important;
    color: white !important;
    padding: 10px 25px !important;
}

/* Dropdowns */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1b1f2e;
    border-radius: 10px;
}

/* Tables */
[data-testid="stDataFrame"] {
    font-family: 'Rajdhani', sans-serif !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class='main-title'>
REAL-TIME MOVIE RECOMMENDATION SYSTEM
</h1>
""", unsafe_allow_html=True)


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from app.pinecone_search import search_similar_movies_pinecone
from app.pinecone_search import search_similar_movies_pinecone
movies_df = pd.read_csv("data/movies.csv")



st.write("Choose recommendation type:")

# Dropdown
option = st.selectbox(
    "Recommendation Type",
    ["User Recommendations", "Movie Similarity" , "Pinecone Semantic Search"]
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
elif option == "Pinecone Semantic Search":

    movie_title = st.selectbox(
        "Select a Movie",
        movies_df['title'].sort_values()
    )

    if st.button("Find Semantic Recommendations"):

        recommendations = search_similar_movies_pinecone(movie_title)

        st.subheader("Pinecone Semantic Recommendations")

        st.dataframe(recommendations)