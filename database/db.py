import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    url = os.getenv("DATABASE_URL") or DATABASE_URL
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Check your .env file.")
    return psycopg2.connect(url, cursor_factory=RealDictCursor)


def init_db():
    """Create tables if they don't exist. Call once on startup."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS repos (
                        id SERIAL PRIMARY KEY,
                        repo_name TEXT UNIQUE NOT NULL,
                        github_url TEXT NOT NULL,
                        total_chunks INTEGER DEFAULT 0,
                        indexed_at TIMESTAMP DEFAULT NOW()
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS file_hashes (
                        id SERIAL PRIMARY KEY,
                        repo_name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        hash TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(repo_name, file_path)
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id SERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT,
                        repo TEXT,
                        asked_at TIMESTAMP DEFAULT NOW()
                    );
                """)
            conn.commit()
        print("Database initialized.")
    except Exception as e:
        print(f"Database init warning (safe to ignore if tables exist): {e}")


# --- Repo metadata ---

def upsert_repo(repo_name, github_url, total_chunks):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO repos (repo_name, github_url, total_chunks, indexed_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (repo_name) DO UPDATE
                SET github_url = EXCLUDED.github_url,
                    total_chunks = EXCLUDED.total_chunks,
                    indexed_at = EXCLUDED.indexed_at;
            """, (repo_name, github_url, total_chunks, datetime.utcnow()))
        conn.commit()


def get_all_repos():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT repo_name FROM repos ORDER BY indexed_at DESC;")
            return [row["repo_name"] for row in cur.fetchall()]


def repo_exists(repo_name):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM repos WHERE repo_name = %s;", (repo_name,))
            return cur.fetchone() is not None


def delete_repo_metadata(repo_name):
    """Delete all metadata for a repo (used when deleting from Qdrant)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Delete from repos table
            cur.execute("DELETE FROM repos WHERE repo_name = %s;", (repo_name,))
            # Delete from file_hashes table
            cur.execute("DELETE FROM file_hashes WHERE repo_name = %s;", (repo_name,))
        conn.commit()
    print(f"Deleted metadata for '{repo_name}' from PostgreSQL")


# --- File hashes ---

def get_hashes(repo_name) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_path, hash FROM file_hashes WHERE repo_name = %s;",
                (repo_name,)
            )
            return {row["file_path"]: row["hash"] for row in cur.fetchall()}


def save_hashes(repo_name, hashes: dict):
    with get_conn() as conn:
        with conn.cursor() as cur:
            for file_path, hash_val in hashes.items():
                cur.execute("""
                    INSERT INTO file_hashes (repo_name, file_path, hash, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (repo_name, file_path) DO UPDATE
                    SET hash = EXCLUDED.hash,
                        updated_at = EXCLUDED.updated_at;
                """, (repo_name, file_path, hash_val, datetime.utcnow()))
        conn.commit()


# --- Query history ---

def save_query(question, answer, repo):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_history (question, answer, repo, asked_at)
                VALUES (%s, %s, %s, %s);
            """, (question, answer, repo, datetime.utcnow()))
        conn.commit()


def get_query_history(limit=50):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT question, answer, repo, asked_at
                FROM query_history
                ORDER BY asked_at DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()