"""Local SQLite storage for enrolled identities and face embeddings."""

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from app.config import config


class DatabaseManager:
    """Manages SQLite storage and caching for identities and biometric embeddings."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else config.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache_valid = False
        self._cached_embeddings: Optional[np.ndarray] = None
        self._cached_person_ids: List[int] = []
        self._cached_person_names: Dict[int, str] = {}
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create schema tables and indices if they do not exist."""
        with self._lock, self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    source_image_name TEXT,
                    source TEXT DEFAULT 'upload',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_embeddings_person_id 
                ON face_embeddings(person_id);
                """
            )
            # Automatic schema migration for existing databases without 'source' column
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(face_embeddings);")
            columns = [col["name"] for col in cursor.fetchall()]
            if "source" not in columns:
                conn.execute("ALTER TABLE face_embeddings ADD COLUMN source TEXT DEFAULT 'upload';")
            conn.commit()

    def _invalidate_cache(self) -> None:
        """Mark memory cache as stale."""
        self._cache_valid = False
        self._cached_embeddings = None
        self._cached_person_ids = []
        self._cached_person_names = {}

    def add_person(self, name: str) -> int:
        """Add a person or return the existing person ID if already enrolled.

        Args:
            name: Person's name.

        Returns:
            int: Person ID.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM persons WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return int(row["id"])
            cursor.execute("INSERT INTO persons (name) VALUES (?)", (name,))
            conn.commit()
            self._invalidate_cache()
            return int(cursor.lastrowid)

    def get_person_by_name(self, name: str) -> Optional[dict]:
        """Look up person record by name."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, created_at FROM persons WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def add_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
        source_image_name: Optional[str] = None,
        source: Optional[str] = "upload",
    ) -> int:
        """Save a face embedding vector for a given person.

        Args:
            person_id: ID of the enrolled person.
            embedding: 1D or 2D numpy array of shape (128,).
            source_image_name: Optional original file name.
            source: Source of enrollment ('upload' or 'webcam').

        Returns:
            int: Inserted embedding ID.
        """
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        if vec.ndim != 1:
            raise ValueError(f"Expected 1D embedding vector, got shape {vec.shape}")

        blob = vec.tobytes()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO face_embeddings (person_id, embedding, source_image_name, source)
                VALUES (?, ?, ?, ?)
                """,
                (person_id, blob, source_image_name, source or "upload"),
            )
            conn.commit()
            self._invalidate_cache()
            return int(cursor.lastrowid)

    def get_embeddings_for_person(self, person_id: int) -> List[dict]:
        """Retrieve all embedding metadata records for a given person.

        Args:
            person_id: ID of the enrolled person.

        Returns:
            List of dicts with id, person_id, source_image_name, source, created_at.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, person_id, source_image_name, source, created_at
                FROM face_embeddings
                WHERE person_id = ?
                ORDER BY id ASC
                """,
                (person_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_embeddings_matrix(
        self,
    ) -> Tuple[np.ndarray, List[int], Dict[int, str]]:
        """Retrieve all stored embeddings as a contiguous matrix alongside person metadata.

        Returns:
            Tuple:
                - embeddings_matrix: np.ndarray of shape (N, 128) float32, or empty (0, 128)
                - person_ids: List of person IDs corresponding to each row
                - person_names: Dict mapping person_id to person name
        """
        with self._lock:
            if self._cache_valid and self._cached_embeddings is not None:
                return (
                    self._cached_embeddings,
                    self._cached_person_ids.copy(),
                    self._cached_person_names.copy(),
                )

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT e.id, e.person_id, e.embedding, p.name
                    FROM face_embeddings e
                    JOIN persons p ON e.person_id = p.id
                    ORDER BY e.id ASC
                    """
                )
                rows = cursor.fetchall()

            if not rows:
                self._cached_embeddings = np.empty((0, config.embedding_dimension), dtype=np.float32)
                self._cached_person_ids = []
                self._cached_person_names = {}
                self._cache_valid = True
                return (
                    self._cached_embeddings,
                    self._cached_person_ids.copy(),
                    self._cached_person_names.copy(),
                )

            vectors = []
            person_ids = []
            person_names = {}

            for row in rows:
                vec = np.frombuffer(row["embedding"], dtype=np.float32)
                vectors.append(vec)
                p_id = int(row["person_id"])
                person_ids.append(p_id)
                person_names[p_id] = str(row["name"])

            matrix = np.vstack(vectors).astype(np.float32)
            self._cached_embeddings = matrix
            self._cached_person_ids = person_ids
            self._cached_person_names = person_names
            self._cache_valid = True

            return (
                self._cached_embeddings,
                self._cached_person_ids.copy(),
                self._cached_person_names.copy(),
            )

    def get_persons_with_counts(self) -> List[dict]:
        """List all enrolled identities along with embedding counts and timestamps."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    p.id, 
                    p.name, 
                    p.created_at,
                    COUNT(e.id) AS embedding_count
                FROM persons p
                LEFT JOIN face_embeddings e ON p.id = e.person_id
                GROUP BY p.id
                ORDER BY p.name ASC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_person(self, person_id: int) -> bool:
        """Delete an enrolled person and cascade delete all associated embeddings.

        Args:
            person_id: ID of the person to remove.

        Returns:
            bool: True if identity was deleted, False if person did not exist.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            if deleted:
                self._invalidate_cache()
            return deleted

    def delete_embedding(self, embedding_id: int) -> bool:
        """Delete a single embedding vector."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM face_embeddings WHERE id = ?", (embedding_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            if deleted:
                self._invalidate_cache()
            return deleted

    def count_identities(self) -> int:
        """Return total count of distinct enrolled identities."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM persons")
            return int(cursor.fetchone()["total"])

    def count_embeddings(self) -> int:
        """Return total count of stored face embeddings."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM face_embeddings")
            return int(cursor.fetchone()["total"])

    def clear_all(self) -> None:
        """Remove all persons and embeddings from the database."""
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM persons")
            conn.commit()
            self._invalidate_cache()
