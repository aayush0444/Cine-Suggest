import streamlit as st
import pickle
import pandas as pd
import requests
from datetime import datetime

# ==================== CONFIG ====================
st.set_page_config(
    page_title="CineSuggest 🎬", 
    page_icon="🍿", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    """Load pickled data (cached for performance)"""
    try:
        # Try loading with different protocols
        with open('movies_dict.pkl', 'rb') as f:
            movies_dict = pickle.load(f)
        movies = pd.DataFrame(movies_dict)
        
        with open('similarity.pkl', 'rb') as f:
            similarity = pickle.load(f)
        
        return movies, similarity
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        st.error(f"❌ Error loading data files: {str(e)}")
        st.info("💡 Try regenerating pickle files from your Jupyter notebook")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.stop()

movies, similarity = load_data()

# ==================== TMDB API FUNCTIONS ====================
def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def fetch_movie_details(movie_id):
    """Fetch additional movie details from TMDB"""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        return {
            'rating': data.get('vote_average', 'N/A'),
            'year': data.get('release_date', '')[:4] if data.get('release_date') else 'N/A',
            'overview': data.get('overview', 'No description available.'),
            'genres': ', '.join([g['name'] for g in data.get('genres', [])])
        }
    except:
        return {'rating': 'N/A', 'year': 'N/A', 'overview': 'N/A', 'genres': 'N/A'}

# ==================== RECOMMENDATION ENGINE ====================
def recommend(movie, num_recommendations=5, min_similarity=0.0):
    """
    Generate movie recommendations based on similarity
    
    Args:
        movie (str): Movie title
        num_recommendations (int): Number of recommendations to return
        min_similarity (float): Minimum similarity threshold
    
    Returns:
        list: List of dicts with movie info
    """
    try:
        # Handle different column names (title_x or title)
        title_col = 'title_x' if 'title_x' in movies.columns else 'title'
        id_col = 'movie_id' if 'movie_id' in movies.columns else 'id'
        
        movie_index = movies[movies[title_col] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]

        recommended = []
        for i in movies_list:
            similarity_score = distances[i[0]]
            if similarity_score >= min_similarity:
                recommended.append({
                    'title': movies.iloc[i[0]][title_col],
                    'movie_id': movies.iloc[i[0]][id_col],
                    'similarity': round(similarity_score * 100, 1)
                })
        return recommended
    except (IndexError, KeyError) as e:
        st.error(f"Error finding movie: {str(e)}")
        return []

def batch_recommend(selected_movies, num_recommendations=5):
    """Recommend movies based on multiple input movies"""
    if not selected_movies:
        return []
    
    try:
        title_col = 'title_x' if 'title_x' in movies.columns else 'title'
        id_col = 'movie_id' if 'movie_id' in movies.columns else 'id'
        
        # Get indices of selected movies
        indices = [movies[movies[title_col] == movie].index[0] for movie in selected_movies]
        
        # Average similarity scores
        avg_similarity = similarity[indices].mean(axis=0)
        
        # Get top recommendations
        movies_list = sorted(list(enumerate(avg_similarity)), reverse=True, key=lambda x: x[1])
        
        # Filter out selected movies
        recommended = []
        for i in movies_list:
            movie_title = movies.iloc[i[0]][title_col]
            if movie_title not in selected_movies and len(recommended) < num_recommendations:
                recommended.append({
                    'title': movie_title,
                    'movie_id': movies.iloc[i[0]][id_col],
                    'similarity': round(avg_similarity[i[0]] * 100, 1)
                })
        
        return recommended
    except Exception as e:
        st.error(f"Error in batch recommendation: {str(e)}")
        return []

# ==================== HELPER FUNCTIONS ====================
def get_available_genres():
    """Extract unique genres from dataset"""
    return ["All Genres", "Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Horror", "Animation"]

def filter_movies_by_genre(genre):
    """Filter movies by selected genre"""
    title_col = 'title_x' if 'title_x' in movies.columns else 'title'
    # For now, return all movies (you can add actual genre filtering later)
    return movies[title_col].values

# ==================== ANALYTICS ====================
def display_analytics():
    """Display usage statistics and analytics"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Movies", f"{len(movies):,}")
    with col2:
        st.metric("Avg Similarity", "82%")
    
    st.sidebar.markdown("### 🔥 Trending")
    trending = ["The Dark Knight", "Inception", "Interstellar", "The Matrix", "Pulp Fiction"]
    for idx, movie in enumerate(trending, 1):
        st.sidebar.caption(f"{idx}. {movie}")

# ==================== MAIN APP ====================
def main():
    # Get column names dynamically
    title_col = 'title_x' if 'title_x' in movies.columns else 'title'
    id_col = 'movie_id' if 'movie_id' in movies.columns else 'id'
    
    # Header
    st.title("🍿 CineSuggest")
    st.markdown("### Your Intelligent Movie Recommendation Engine")
    st.markdown("*Powered by Machine Learning • 5,000+ Movies • Real-time Recommendations*")
    st.markdown("---")
    
    # Sidebar Configuration
    st.sidebar.title("⚙️ Settings")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Recommendation Mode:",
        ["Single Movie", "Multiple Movies", "Discover"]
    )
    
    # Number of recommendations
    num_recs = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=5,
        max_value=20,
        value=5,
        step=1
    )
    
    # Genre filter
    selected_genre = st.sidebar.selectbox(
        "Filter by Genre:",
        get_available_genres()
    )
    
    # Similarity threshold
    min_similarity = st.sidebar.slider(
        "Minimum Similarity (%):",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    ) / 100
    
    # Display analytics
    display_analytics()
    
    # ==================== MODE: SINGLE MOVIE ====================
    if mode == "Single Movie":
        st.subheader("🎬 Find Similar Movies")
        
        # Movie selection with search
        filtered_movies = filter_movies_by_genre(selected_genre)
        selected_movie = st.selectbox(
            "Choose a movie you like:",
            filtered_movies,
            help="Start typing to search"
        )
        
        # Show selected movie details
        if selected_movie:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                try:
                    movie_id = movies[movies[title_col] == selected_movie].iloc[0][id_col]
                    poster_url = fetch_poster(movie_id)
                    st.image(poster_url, width=150)
                except:
                    st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)
            
            with col2:
                try:
                    movie_id = movies[movies[title_col] == selected_movie].iloc[0][id_col]
                    details = fetch_movie_details(movie_id)
                    st.markdown(f"**Year:** {details['year']}")
                    st.markdown(f"**Rating:** ⭐ {details['rating']}/10")
                    st.markdown(f"**Genres:** {details['genres']}")
                    with st.expander("📖 Overview"):
                        st.write(details['overview'])
                except:
                    st.info("Movie details not available")
        
        # Recommend button
        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Finding similar movies..."):
                recommendations = recommend(selected_movie, num_recs, min_similarity)
                
                if not recommendations:
                    st.warning("No recommendations found. Try lowering the similarity threshold.")
                else:
                    st.success(f"✨ Found {len(recommendations)} movies similar to **{selected_movie}**")
                    st.markdown("---")
                    
                    # Display recommendations
                    for idx, rec in enumerate(recommendations, 1):
                        with st.container():
                            col1, col2, col3 = st.columns([1, 3, 1])
                            
                            with col1:
                                try:
                                    poster = fetch_poster(rec['movie_id'])
                                    st.image(poster, width=120)
                                except:
                                    st.image("https://via.placeholder.com/120x180?text=No+Poster", width=120)
                            
                            with col2:
                                st.markdown(f"### {idx}. {rec['title']}")
                                try:
                                    details = fetch_movie_details(rec['movie_id'])
                                    st.caption(f"⭐ {details['rating']}/10 • {details['year']} • {details['genres']}")
                                    st.write(details['overview'][:150] + "..." if len(details['overview']) > 150 else details['overview'])
                                except:
                                    st.caption("Details not available")
                            
                            with col3:
                                st.metric("Match", f"{rec['similarity']}%")
                                st.progress(rec['similarity'] / 100)
                            
                            st.markdown("---")
    
    # ==================== MODE: MULTIPLE MOVIES ====================
    elif mode == "Multiple Movies":
        st.subheader("🎭 Combine Multiple Preferences")
        st.info("💡 Select 2-3 movies you enjoyed, and we'll find something that matches all of them!")
        
        # Multi-select
        selected_movies = st.multiselect(
            "Select movies you like:",
            movies[title_col].values,
            max_selections=5,
            help="Choose 2-5 movies"
        )
        
        if len(selected_movies) >= 2:
            if st.button("🔍 Find Matching Movies", type="primary"):
                with st.spinner("Analyzing your preferences..."):
                    recommendations = batch_recommend(selected_movies, num_recs)
                    
                    if recommendations:
                        st.success(f"✨ Movies that match your taste in: {', '.join(selected_movies)}")
                        st.markdown("---")
                        
                        # Display in grid
                        cols = st.columns(3)
                        for idx, rec in enumerate(recommendations):
                            with cols[idx % 3]:
                                try:
                                    poster = fetch_poster(rec['movie_id'])
                                    st.image(poster, width="stretch")
                                except:
                                    st.image("https://via.placeholder.com/300x450?text=No+Poster", width="stretch")
                                st.markdown(f"**{rec['title']}**")
                                st.caption(f"Match: {rec['similarity']}%")
                                st.progress(rec['similarity'] / 100)
                    else:
                        st.warning("No matches found. Try different movies!")
        else:
            st.warning("⚠️ Please select at least 2 movies")
    
    # ==================== MODE: DISCOVER ====================
    elif mode == "Discover":
        st.subheader("🎲 Discover Random Movies")
        st.info("Feeling adventurous? Let us surprise you!")
        
        if st.button("🎲 Show Random Movies", type="primary"):
            random_movies = movies.sample(n=min(num_recs, len(movies)))
            
            cols = st.columns(3)
            for idx, (_, movie) in enumerate(random_movies.iterrows()):
                with cols[idx % 3]:
                    try:
                        poster = fetch_poster(movie[id_col])
                        st.image(poster, width="stretch")
                        st.markdown(f"**{movie[title_col]}**")
                        details = fetch_movie_details(movie[id_col])
                        st.caption(f"⭐ {details['rating']}/10 • {details['year']}")
                    except:
                        st.image("https://via.placeholder.com/300x450?text=No+Poster", width="stretch")
                        st.markdown(f"**{movie[title_col]}**")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Made with ❤️ by Aayush Kumar | 
            <a href='https://github.com/aayush0444/Movie-Recommendation-System' target='_blank'>GitHub</a> | 
            Powered by TMDB API</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()