import streamlit as st
import pickle
import time

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1 {
    color: #ff4b4b;
    text-align: center;
    font-size: 55px;
}

h3 {
    color: #444444;
    text-align: center;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size:18px;
    border:none;
}

.stButton>button:hover {
    background-color: #ff2b2b;
    color:white;
}

.movie-card{
    background:#ffffff;
    padding:20px;
    border-radius:15px;
    margin-bottom:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.15);
    font-size:22px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:60px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Data
# -------------------------
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# -------------------------
# Recommendation Function
# -------------------------
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:

    st.title("🎬 About Project")

    st.write("""
This Movie Recommendation System is developed using:

✅ Python

✅ Pandas

✅ Scikit-Learn

✅ CountVectorizer

✅ Cosine Similarity

✅ Streamlit
""")

    st.divider()

    st.success("AI Internship Project")

# -------------------------
# Main Heading
# -------------------------
st.title("🎬 Movie Recommendation System")

st.markdown(
    "<h3>Discover Movies Similar To Your Favourite One</h3>",
    unsafe_allow_html=True
)

st.write("")

# -------------------------
# Movie Selection
# -------------------------
selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movies['title'].values
)

# -------------------------
# Recommendation Button
# -------------------------
if st.button("🚀 Recommend Movies"):

    with st.spinner("Finding similar movies..."):
        time.sleep(1)

    recommendations = recommend(selected_movie)

    st.success("Top 5 Recommended Movies")

    st.write("")

    for movie in recommendations:

        st.markdown(
            f"""
            <div class="movie-card">
            🎬 <b>{movie}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.balloons()

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class='footer'>

Made with ❤️ using Streamlit

</div>
""", unsafe_allow_html=True)