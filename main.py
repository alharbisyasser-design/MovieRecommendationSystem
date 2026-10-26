"""Movie Recommendation System — Streamlit coursework application."""

import hashlib
import hmac
import os
import secrets
import sqlite3

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
DATABASE = "mrs.db"


def run_sql(statement, values=(), result=None):
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        cursor = connection.execute(statement, values)
        output = cursor.fetchall() if result == "all" else cursor.fetchone() if result == "one" else cursor.lastrowid
        connection.commit()
        return output
    finally:
        connection.close()


def initialize():
    first_launch = not os.path.exists(DATABASE)
    for statement in (
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL, preferences TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS movies (id INTEGER PRIMARY KEY, title TEXT UNIQUE NOT NULL, genre TEXT NOT NULL, description TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS ratings (user_id INTEGER NOT NULL, movie_id INTEGER NOT NULL, score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5), rated_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(user_id,movie_id), FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE)",
        "CREATE TABLE IF NOT EXISTS watches (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, movie_id INTEGER NOT NULL, watched_at TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE, FOREIGN KEY(movie_id) REFERENCES movies(id) ON DELETE CASCADE)",
    ):
        run_sql(statement)
    if first_launch:
        for title, genre, description in (
            ("Inception", "Science Fiction", "Dreams and mind bending mystery"),
            ("Interstellar", "Science Fiction", "Space exploration and family"),
            ("The Dark Knight", "Action", "Superhero crime thriller"),
            ("The Matrix", "Science Fiction", "Virtual reality and action"),
            ("Titanic", "Romance", "Historical love and disaster"),
            ("Toy Story", "Animation", "Friendship and animated adventure"),
            ("Coco", "Animation", "Music family and memory"),
            ("The Lion King", "Animation", "Animals family and leadership"),
        ):
            run_sql("INSERT INTO movies(title,genre,description) VALUES(?,?,?)", (title, genre, description))


def encrypt_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 250000).hex()


def movie_catalog():
    return [dict(row) for row in run_sql("SELECT * FROM movies ORDER BY title", result="all")]


def get_recommendations(account):
    catalog = movie_catalog()
    if not catalog:
        return []
    activities = run_sql(
        "SELECT m.genre, m.description FROM movies m JOIN ratings r ON r.movie_id=m.id WHERE r.user_id=? AND r.score>=4",
        (account["id"],), "all"
    )
    profile = " ".join([account["preferences"]] + [f"{item['genre']} {item['description']}" for item in activities]).strip()
    watched = run_sql("SELECT DISTINCT movie_id FROM watches WHERE user_id=?", (account["id"],), "all")
    excluded = {item["movie_id"] for item in watched}
    texts = [f"{item['genre']} {item['description']}" for item in catalog]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [profile or "film"])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    eligible = [(float(similarities[index]), item) for index, item in enumerate(catalog) if item["id"] not in excluded]
    eligible.sort(key=lambda pair: (-pair[0], pair[1]["title"]))
    return [item for _, item in eligible[:5]]


def tabular(rows, columns=None):
    if rows:
        st.dataframe(pd.DataFrame([dict(row) for row in rows]), use_container_width=True, hide_index=True)
    else:
        st.info("No records yet")


def access_panel():
    st.subheader("Account access")
    option = st.radio("Choose", ["Login", "Register", "Administrator"], horizontal=True)
    if option == "Login":
        with st.form("sign_in"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            account = run_sql("SELECT * FROM users WHERE username=?", (username.strip(),), "one")
            if account and hmac.compare_digest(encrypt_password(password, account["salt"]), account["password_hash"]):
                st.session_state.user_id = account["id"]
                st.rerun()
            st.error("Invalid credentials")
    elif option == "Register":
        genres = sorted({item["genre"] for item in movie_catalog()})
        with st.form("sign_up"):
            username = st.text_input("New username")
            password = st.text_input("New password", type="password")
            selections = st.multiselect("Favorite genres", genres)
            submitted = st.form_submit_button("Create account")
        if submitted:
            if not username.strip() or not password or not selections:
                st.warning("Complete every field")
            else:
                salt = secrets.token_hex(16)
                try:
                    user_id = run_sql(
                        "INSERT INTO users(username,salt,password_hash,preferences) VALUES(?,?,?,?)",
                        (username.strip(), salt, encrypt_password(password, salt), ", ".join(selections))
                    )
                    st.session_state.user_id = user_id
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Username already exists")
    else:
        with st.form("admin_access"):
            key = st.text_input("Unique administrator key", type="password")
            submitted = st.form_submit_button("Unlock")
        if submitted:
            configured = os.environ.get("MRS_ADMIN_KEY", "")
            if configured and hmac.compare_digest(key, configured):
                st.session_state.admin = True
                st.rerun()
            st.error("Invalid key or administrator key not configured")


def explore(account):
    st.header("Search and rate movies")
    catalog = movie_catalog()
    search = st.text_input("Movie search")
    matches = [item for item in catalog if search.lower() in item["title"].lower()]
    if not matches:
        st.info("No matching movies")
        return
    chosen = st.selectbox("Select movie", matches, format_func=lambda item: item["title"])
    st.write(f"Genre: {chosen['genre']}")
    st.write(chosen["description"])
    with st.form("rate_form"):
        score = st.select_slider("Your rating", options=[1, 2, 3, 4, 5], value=3)
        submitted = st.form_submit_button("Save rating")
    if submitted:
        run_sql(
            "INSERT INTO ratings(user_id,movie_id,score) VALUES(?,?,?) ON CONFLICT(user_id,movie_id) DO UPDATE SET score=excluded.score,rated_at=CURRENT_TIMESTAMP",
            (account["id"], chosen["id"], score)
        )
        st.success("Rating recorded")
    if st.button("Mark as watched"):
        run_sql("INSERT INTO watches(user_id,movie_id) VALUES(?,?)", (account["id"], chosen["id"]))
        st.success("Viewing history updated")
    st.subheader("Recommended for you")
    choices = get_recommendations(account)
    st.write(", ".join(item["title"] for item in choices) if choices else "No unseen movies available")


def dashboard(account):
    st.header("User dashboard")
    st.subheader("Top recommended movies")
    selections = get_recommendations(account)
    st.write(", ".join(item["title"] for item in selections) if selections else "No unseen movies available")
    st.subheader("Trending movies")
    trending = run_sql(
        "SELECT m.title,COUNT(w.id) AS views FROM movies m LEFT JOIN watches w ON w.movie_id=m.id GROUP BY m.id ORDER BY views DESC,m.title LIMIT 5",
        result="all"
    )
    if any(item["views"] for item in trending):
        st.bar_chart(pd.DataFrame([dict(item) for item in trending]).set_index("title"))
    else:
        st.info("No viewing trends yet")
    st.subheader("Popular genres")
    genres = run_sql(
        "SELECT m.genre, COUNT(w.id) AS views FROM movies m JOIN watches w ON w.movie_id=m.id GROUP BY m.genre ORDER BY views DESC",
        result="all"
    )
    if genres:
        st.bar_chart(pd.DataFrame([dict(item) for item in genres]).set_index("genre"))
    else:
        st.info("No genre activity yet")
    st.subheader("Watch history")
    tabular(run_sql(
        "SELECT m.title,w.watched_at FROM watches w JOIN movies m ON m.id=w.movie_id WHERE w.user_id=? ORDER BY w.watched_at DESC",
        (account["id"],), "all"
    ))
    st.subheader("Rating logs")
    tabular(run_sql(
        "SELECT m.title,r.score,r.rated_at FROM ratings r JOIN movies m ON m.id=r.movie_id WHERE r.user_id=? ORDER BY r.rated_at DESC",
        (account["id"],), "all"
    ))
    st.subheader("Top rated movies")
    averages = run_sql(
        "SELECT m.title, ROUND(AVG(r.score),2) AS average FROM ratings r JOIN movies m ON m.id=r.movie_id GROUP BY m.id ORDER BY average DESC LIMIT 5",
        result="all"
    )
    if averages:
        st.bar_chart(pd.DataFrame([dict(item) for item in averages]).set_index("title"))
    else:
        st.info("No ratings yet")


def admin_console():
    st.header("Administrator console")
    operation = st.radio("Manage movies", ["Add", "Edit", "Remove"], horizontal=True)
    catalog = movie_catalog()
    if operation == "Add":
        with st.form("new_movie"):
            title = st.text_input("Title")
            genre = st.text_input("Genre")
            description = st.text_area("Description")
            submitted = st.form_submit_button("Add movie")
        if submitted:
            if not all([title.strip(), genre.strip(), description.strip()]):
                st.warning("Complete every field")
            else:
                try:
                    run_sql("INSERT INTO movies(title,genre,description) VALUES(?,?,?)", (title.strip(), genre.strip(), description.strip()))
                    st.success("Movie added")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Title already exists")
    elif catalog:
        selected = st.selectbox("Movie", catalog, format_func=lambda item: item["title"])
        if operation == "Edit":
            with st.form("edit_movie"):
                title = st.text_input("Title", selected["title"])
                genre = st.text_input("Genre", selected["genre"])
                description = st.text_area("Description", selected["description"])
                submitted = st.form_submit_button("Save changes")
            if submitted:
                if not all([title.strip(), genre.strip(), description.strip()]):
                    st.warning("Complete every field")
                else:
                    try:
                        run_sql("UPDATE movies SET title=?,genre=?,description=? WHERE id=?", (title.strip(), genre.strip(), description.strip(), selected["id"]))
                        st.success("Movie updated")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Title already exists")
        else:
            confirmation = st.checkbox("Confirm permanent removal")
            if st.button("Delete movie", disabled=not confirmation):
                run_sql("DELETE FROM movies WHERE id=?", (selected["id"],))
                st.success("Movie removed")
                st.rerun()
    st.subheader("Most watched movies")
    data = run_sql(
        "SELECT m.title,COUNT(w.id) AS views FROM movies m LEFT JOIN watches w ON w.movie_id=m.id GROUP BY m.id ORDER BY views DESC",
        result="all"
    )
    tabular(data)
    if any(item["views"] for item in data):
        st.bar_chart(pd.DataFrame([dict(item) for item in data]).set_index("title"))


initialize()
st.title("🎬 Movie Recommendation System")
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "admin" not in st.session_state:
    st.session_state.admin = False
if st.session_state.user_id is None and not st.session_state.admin:
    access_panel()
else:
    with st.sidebar:
        pages = (["Search and rate", "Dashboard"] if st.session_state.user_id is not None else [])
        if st.session_state.admin:
            pages.append("Administrator")
        page = st.radio("Navigation", pages)
        if st.button("Log out"):
            st.session_state.user_id = None
            st.session_state.admin = False
            st.rerun()
    if page == "Administrator":
        admin_console()
    else:
        current = run_sql("SELECT * FROM users WHERE id=?", (st.session_state.user_id,), "one")
        if current:
            explore(current) if page == "Search and rate" else dashboard(current)
        else:
            st.error("Account unavailable")

