import pandas as pd

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")

movies = movies[
    [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew'
    ]
]

print(movies.head())
print("\nMissing Values:")
print(movies.isnull().sum())
movies.dropna(inplace=True)

print("\nAfter Removing Missing Values:")
print(movies.isnull().sum())
print("\nGenres of first movie:")
print(movies['genres'][0])
import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# Load Dataset
# ===========================

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge both datasets
movies = movies.merge(credits, on="title")

# ===========================
# Select Required Columns
# ===========================

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Remove missing values
movies.dropna(inplace=True)

# ===========================
# Functions
# ===========================

# Convert genres and keywords
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Get first 3 actors
def convert3(text):
    L = []
    counter = 0

    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break

    return L

# Get Director
def fetch_director(text):
    L = []

    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break

    return L

# ===========================
# Apply Functions
# ===========================

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(convert3)

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# ===========================
# Remove Spaces
# ===========================

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])

movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])

movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# ===========================
# Create Tags
# ===========================

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Keep only required columns

new_df = movies[['movie_id','title','tags']]

# Convert list into string

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

# Convert to lowercase

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

print(new_df.head())

# ===========================
# Text Vectorization
# ===========================

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

print(vectors.shape)

# ===========================
# Cosine Similarity
# ===========================

similarity = cosine_similarity(vectors)

print(similarity.shape)

# ===========================
# Recommendation Function
# ===========================

def recommend(movie):

    movie = movie.lower()

    if movie not in new_df['title'].str.lower().values:
        print("Movie not found!")
        return

    movie_index = new_df[new_df['title'].str.lower()==movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    print("\nRecommended Movies:\n")

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# ===========================
# Test
# ===========================

recommend("Avatar")
import pickle

# Save processed dataframe
pickle.dump(new_df, open("movies.pkl", "wb"))

# Save similarity matrix
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("Files saved successfully!")