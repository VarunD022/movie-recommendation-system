import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import zipfile

# Extract data.zip if not already extracted
if not os.path.exists('data'):
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall('data')
        st.write("Extracted data successfully.")

# Load datasets
movies_file = 'data/tmdb_5000_movies.csv'
credits_file = 'data/tmdb_5000_credits.csv'

try:
    movies = pd.read_csv(movies_file)
    credits = pd.read_csv(credits_file)
except FileNotFoundError:
    st.error("CSV files not found. Ensure 'data.zip' contains the required files.")
    st.stop()

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
similarity = cosine_similarity(vector)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Streamlit UI
st.markdown('<p style="font-family: \'Courier New\', monospace; font-size: 12px; text-align: center;">Created by Varun</p>', unsafe_allow_html=True)
st.title('🎬 Movie Recommendation System')
selected_movie = st.selectbox('Select a movie:', movies['title'].values)

if st.button('Recommend'):
    try:
        recommendations = recommend(selected_movie)
        st.write('**Top Recommendations:**')
        for movie in recommendations:
            st.write(f"- {movie}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
