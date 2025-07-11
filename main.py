import pandas as pd
import ast
import os
import zipfile
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

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

# Helper to safely parse stringified JSON
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

# Process columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if pd.notnull(x) else [])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if pd.notnull(x) else [])
movies['overview'] = movies['overview'].fillna('')
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['year'] = movies['release_date'].dt.year.fillna(0).astype(int)

# Create tags
movies['tags'] = movies['overview'] + ' ' + \
                 movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['crew'].apply(lambda x: ' '.join(x))

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vector)

# Recommend based on movie, genre, year
def recommend(movie, genre_filter=None, year_range=None):
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
        if year_range and not (year_range[0] <= movie_data['year'] <= year_range[1]):
            continue
        recommended.append((movie_data.title, ', '.join(movie_data.crew)))
        if len(recommended) == 5:
            break
    return recommended

# Surprise Me
def surprise_me(genre):
    filtered = movies[movies['genres'].apply(lambda x: genre in x)]
    selected = filtered.sample(n=5)
    return [(row.title, ', '.join(row.crew)) for _, row in selected.iterrows()]

# Search by actor/director
def search_by_person(name):
    matched = movies[movies['cast'].apply(lambda x: name in x) | movies['crew'].apply(lambda x: name in x)]
    return [(row.title, ', '.join(row.crew)) for _, row in matched.head(5).iterrows()]

# Mood-based recommendation
def recommend_by_mood(sentence):
    words = [word for word in sentence.lower().split() if word not in ENGLISH_STOP_WORDS]
    if not words:
        return []
    matches = []
    for idx, row in movies.iterrows():
        tag_words = row['tags'].lower()
        if any(word in tag_words for word in words):
            matches.append((row.title, ', '.join(row.crew)))
        if len(matches) == 10:
            break
    return matches

# Utility
def get_all_movies():
    return sorted(movies['title'].unique())

def get_all_genres():
    return sorted(set([genre for sublist in movies['genres'] for genre in sublist]))

__all__ = [
    'recommend',
    'surprise_me',
    'search_by_person',
    'recommend_by_mood',
    'get_all_movies',
    'get_all_genres',
    'movies'
]
