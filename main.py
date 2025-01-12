import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import zipfile

# Extract data.zip if not already extracted
if not os.path.exists('data'):
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

# Load datasets
movies_file = 'data/tmdb_5000_movies.csv'
credits_file = 'data/tmdb_5000_credits.csv'

movies = pd.read_csv(movies_file)
credits = pd.read_csv(credits_file)

# Merge datasets
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
movies['tags'] = movies['overview'] + movies['genres'].apply(lambda x: ' '.join(x)) + movies['keywords'].apply(lambda x: ' '.join(x)) + movies['cast'].apply(lambda x: ' '.join(x)) + movies['crew'].apply(lambda x: ' '.join(x))

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vector)

def recommend(movie, genre_filter=None, language_filter=None, year_range=None):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:20]
    
    recommended_movies = []
    for i in movie_list:
        movie_data = movies.iloc[i[0]]
        if genre_filter and not any(genre in movie_data['genres'] for genre in genre_filter):
            continue
        if language_filter and movie_data['original_language'] not in language_filter:
            continue
        if year_range:
            release_year = int(movie_data['release_date'][:4]) if pd.notnull(movie_data['release_date']) else 0
            if release_year < year_range[0] or release_year > year_range[1]:
                continue
        recommended_movies.append(movie_data.title)
        if len(recommended_movies) == 5:
            break
    return recommended_movies

# Example usage
if __name__ == "__main__":
    print("Recommendations for 'The Dark Knight':", recommend('The Dark Knight', genre_filter=['Action'], year_range=(2000, 2020)))
    print("Recommendations for 'Inception':", recommend('Inception', language_filter=['en'], year_range=(2010, 2020)))
    print("Recommendations for 'Interstellar':", recommend('Interstellar'))
