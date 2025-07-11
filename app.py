import streamlit as st
import main

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")

# Section 1: Movie-based recommendation
st.header("📌 Recommend Movies by Title")
selected_movie = st.selectbox("Choose a movie", main.get_all_movies())
selected_genres = st.multiselect("Filter by genre (optional)", main.get_all_genres())
selected_years = st.slider("Filter by release year", 1995, 2020, (2010, 2020))

if st.button("Recommend"):
    recommendations = main.recommend(selected_movie, selected_genres, selected_years)
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

# Section 4: Mood-Based Recommendation
st.header("🧠 Let Us Know Your Mood [AI]")
mood_input = st.text_input("Type how you're feeling or what you want to watch")
if st.button("Find Movies"):
    mood_recommendations = main.recommend_by_mood(mood_input)
    if mood_recommendations:
        st.subheader("🎯 Based on your mood, we suggest:")
        for title, director in mood_recommendations:
            st.markdown(f"**{title}** — 🎬 Director(s): {director}")
    else:
        st.warning("No matching movies found for your mood.")
