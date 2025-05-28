import streamlit as st
import main

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")

# Section 1: Movie-based recommendation
st.header("📌 Recommend Movies by Title")
selected_movie = st.selectbox("Choose a movie", main.get_all_movies())
selected_genres = st.multiselect("Filter by genre (optional)", main.get_all_genres())

if st.button("Recommend"):
    recommendations = main.recommend(selected_movie, selected_genres)
    if recommendations:
        st.subheader("🎥 Recommended Movies:")
        for title, director in recommendations:
            st.markdown(f"**{title}** — 🎬 Director(s): {director}")
    else:
        st.warning("No recommendations found with selected filters.")

# Section 2: Surprise Me
st.header("🎁 Surprise Me")
random_genre = st.selectbox("Choose a genre", main.get_all_genres())
if st.button("Surprise Me"):
    surprise_list = main.surprise_me(random_genre)
    st.subheader("🎲 Random Recommendations:")
    for title, director in surprise_list:
        st.markdown(f"**{title}** — 🎬 Director(s): {director}")

# Section 3: Search by Actor or Director
st.header("🔍 Search by Actor or Director")
person = st.text_input("Enter actor or director name")
if person:
    search_results = main.search_by_person(person)
    if search_results:
        st.subheader(f"🎭 Top Movies for '{person}':")
        for title, director in search_results:
            st.markdown(f"**{title}** — 🎬 Director(s): {director}")
    else:
        st.warning("No movies found for the given name.")
