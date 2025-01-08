import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the datasets on the 'title' column
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Preprocessing the data
import ast

def convert(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if pd.notnull(x) else [])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if pd.notnull(x) else [])

movies['overview'] = movies['overview'].fillna('')
movies['tags'] = movies['overview'] + movies['genres'].apply(lambda x: ' '.join(x)) + movies['keywords'].apply(lambda x: ' '.join(x)) + movies['cast'].apply(lambda x: ' '.join(x)) + movies['crew'].apply(lambda x: ' '.join(x))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vector)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movie_list:
        print(movies.iloc[i[0]].title)

# Example Usage
recommend('The Dark Knight')
recommend('Inception')
recommend('Interstellar')
