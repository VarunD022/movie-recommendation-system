import streamlit as st 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load and preprocess data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')

# Data Preprocessing
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
    recommendations = recommend(selected_movie)
    st.write('**Top Recommendations:**')
    for movie in recommendations:
        st.write(f"- {movie}")
