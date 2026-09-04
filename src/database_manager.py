import os
import sqlite3
import yaml
from datetime import datetime

class DBManager:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.db_path = config["database"]["db_path"]
        
        # Ensure data folder exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

    def _get_connection(self):
        """Helper to get a database connection with auto-closed cursors."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enables column access by name
        return conn

    def initialize_db(self):
        """Creates the database schema and indexes if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create posts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT UNIQUE,
                platform TEXT NOT NULL,
                username TEXT NOT NULL,
                followers INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                topic TEXT NOT NULL,
                likes INTEGER DEFAULT 0,
                retweets INTEGER DEFAULT 0,
                sentiment_label TEXT,
                sentiment_score REAL,
                model_used TEXT,
                processed INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes to speed up dashboard queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed ON posts(processed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic ON posts(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON posts(sentiment_label)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON posts(timestamp)")
        
        conn.commit()
        conn.close()
        print("SQLite Database initialized successfully.")

    def insert_raw_posts(self, posts):
        """Inserts a list of dictionaries representing raw, unprocessed posts.
        
        Each post dict should contain: platform, username, followers, timestamp, text, topic, likes, retweets.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        inserted_count = 0
        for post in posts:
            try:
                # Use INSERT OR IGNORE to prevent duplicate tweets by tweet_id
                cursor.execute("""
                    INSERT OR IGNORE INTO posts (
                        tweet_id, platform, username, followers, timestamp, text, topic, likes, retweets, processed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    post.get("tweet_id"),
                    post["platform"],
                    post["username"],
                    post.get("followers", 0),
                    post["timestamp"],
                    post["text"],
                    post["topic"],
                    post.get("likes", 0),
                    post.get("retweets", 0)
                ))
                if cursor.rowcount > 0:
                    inserted_count += 1
            except sqlite3.Error as e:
                print(f"Database insertion error: {e}")
                
        conn.commit()
        conn.close()
        return inserted_count

    def get_unprocessed_posts(self, limit=50):
        """Fetches the oldest unprocessed posts to run prediction inference on."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, text, topic FROM posts 
            WHERE processed = 0 
            ORDER BY timestamp ASC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_post_sentiment(self, post_id, label, score, model_used):
        """Updates a post's sentiment metadata and flags it as processed."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE posts 
            SET sentiment_label = ?, 
                sentiment_score = ?, 
                model_used = ?, 
                processed = 1 
            WHERE id = ?
        """, (label, score, model_used, post_id))
        
        conn.commit()
        conn.close()

    def get_processed_data(self, topic=None, limit=1000):
        """Retrieves processed posts for the dashboard frontend."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if topic:
            cursor.execute("""
                SELECT * FROM posts 
                WHERE processed = 1 AND topic = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (topic, limit))
        else:
            cursor.execute("""
                SELECT * FROM posts 
                WHERE processed = 1 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_kpi_summary(self, topic=None):
        """Computes key dashboard stats (Total, positive percentage, average sentiment score)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query_base = "SELECT COUNT(*) as total, AVG(sentiment_score) as avg_score FROM posts WHERE processed = 1"
        pos_query_base = "SELECT COUNT(*) as pos_count FROM posts WHERE processed = 1 AND sentiment_label = 'Positive'"
        
        if topic:
            cursor.execute(query_base + " AND topic = ?", (topic,))
            stats = dict(cursor.fetchone())
            cursor.execute(pos_query_base + " AND topic = ?", (topic,))
            pos_stats = dict(cursor.fetchone())
        else:
            cursor.execute(query_base)
            stats = dict(cursor.fetchone())
            cursor.execute(pos_query_base)
            pos_stats = dict(cursor.fetchone())
            
        conn.close()
        
        total = stats.get("total", 0)
        avg_score = stats.get("avg_score") or 0.0
        pos_count = pos_stats.get("pos_count", 0)
        
        pos_pct = (pos_count / total * 100) if total > 0 else 0.0
        
        return {
            "total_posts": total,
            "avg_score": round(avg_score, 2),
            "pos_pct": round(pos_pct, 1)
        }

    def clear_database(self):
        """Truncates all posts in the database, resetting state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM posts")
        conn.commit()
        conn.close()
        print("Database wiped successfully.")
