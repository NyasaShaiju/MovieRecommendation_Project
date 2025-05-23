import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    movies = movies.sample(n=500, random_state=42).reset_index(drop=True)

    movies.drop_duplicates(inplace=True)
    movies['genres'] = movies['genres'].fillna('')
    return movies, ratings

movies, ratings = load_data()
def get_genre_similarity_matrix():
     # Assuming genre_matrix is created via CountVectorizer
    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform(movies['genres'])  # shape: [n_movies x n_genres]
    
    # Use sparse format
    genre_matrix_sparse = csr_matrix(genre_matrix)

    # Compute similarity
    return cosine_similarity(genre_matrix_sparse)

genre_sim = get_genre_similarity_matrix()

def recommend_by_genre(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return ["Movie not found."]
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(genre_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return movies['title'].iloc[top_indices].tolist()
def get_collaborative_matrix():
    merged = pd.merge(ratings, movies, on='movieId')
    user_movie_matrix = merged.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    return cosine_similarity(user_movie_matrix.T), user_movie_matrix.columns

collab_sim, movie_titles = get_collaborative_matrix()
collab_df = pd.DataFrame(collab_sim, index=movie_titles, columns=movie_titles)

def recommend_collaborative(movie_title, top_n=5):
    if movie_title not in collab_df.columns:
        return ["Movie not found."]
    similar_scores = collab_df[movie_title].sort_values(ascending=False)
    return similar_scores.iloc[1:top_n+1].index.tolist()
st.title("🎬 Movie Recommendation System")

# Dropdown for movie selection
movie_input = st.selectbox("Select a movie you like", sorted(set(movies['title']).intersection(set(collab_df.columns))))

# Button to trigger recommendation
if st.button("Get Recommendations"):
    st.subheader("🔁 Collaborative Filtering Recommendations(People also like these movies)")
    for i, movie in enumerate(recommend_collaborative(movie_input), 1):
        st.write(f"{i}. {movie}")
    
    st.subheader("🎯 Content-Based Filtering Recommendations(Similar movies)")
    for i, movie in enumerate(recommend_by_genre(movie_input), 1):
        st.write(f"{i}. {movie}")
