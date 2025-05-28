import pandas as pd
import ast
import os
import zipfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Extract data.zip if not already extracted
if not os.path.exists('data'):
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

# Load datasets
movies_file = 'data/tmdb_5000_movies.csv'
credits_file = 'data/tmdb_5000_credits.csv'

movies = pd.read_csv(movies_file)
credits = pd.read_csv(credits_file)

# Merge on title
movies = movies.merge(credits, on='title')

# Data preprocessing
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if pd.notnull(x) else [])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if pd.notnull(x) else [])
movies['overview'] = movies['overview'].fillna('')

# Create tags column for recommendations
movies['tags'] = movies['overview'] + ' ' + \
                 movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['crew'].apply(lambda x: ' '.join(x))

# Vectorization and similarity computation
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vector)

# Recommendation by title + optional genre
def recommend(movie, genre_filter=None):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
    except IndexError:
        return []

    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:20]

    recommended = []
    for i in movie_list:
        movie_data = movies.iloc[i[0]]
        if genre_filter and not any(g in movie_data['genres'] for g in genre_filter):
            continue
        recommended.append((movie_data.title, ', '.join(movie_data.crew)))
        if len(recommended) == 5:
            break
    return recommended

# Surprise me feature
def surprise_me(genre):
    filtered = movies[movies['genres'].apply(lambda x: genre in x)]
    selected = filtered.sample(n=5)
    return [(row.title, ', '.join(row.crew)) for _, row in selected.iterrows()]

# Search by actor or director
def search_by_person(name):
    matched = movies[movies['cast'].apply(lambda x: name in x) | movies['crew'].apply(lambda x: name in x)]
    return [(row.title, ', '.join(row.crew)) for _, row in matched.head(5).iterrows()]

# Utility functions
def get_all_movies():
    return sorted(movies['title'].unique())

def get_all_genres():
    return sorted(set([genre for sublist in movies['genres'] for genre in sublist]))
