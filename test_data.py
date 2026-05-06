import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Movies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())

print("\nTotal movies:", len(movies))
print("Total ratings:", len(ratings))