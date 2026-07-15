import sqlite3
import os
import uuid
from datetime import datetime
import numpy as np


class DBWriter:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def _get_collection_id(self):
        self.cursor.execute("SELECT id FROM collections LIMIT 1")
        result = self.cursor.fetchone()
        if result:
            return result[0]
        return None

    def _get_or_create_segment(self, collection_id):
        self.cursor.execute("SELECT id FROM segments WHERE collection = ? LIMIT 1", (collection_id,))
        result = self.cursor.fetchone()
        if result:
            return result[0]

        new_segment_id = str(uuid.uuid4())
        self.cursor.execute(
            "INSERT INTO segments (id, type, scope, collection) VALUES (?, 'hnsw', 'default', ?)",
            (new_segment_id, collection_id)
        )
        self.conn.commit()
        return new_segment_id

    def _get_next_seq_id(self, segment_id):
        self.cursor.execute("SELECT seq_id FROM max_seq_id WHERE segment_id = ?", (segment_id,))
        result = self.cursor.fetchone()
        if result:
            next_seq_id = result[0] + 1
        else:
            next_seq_id = 1
        self.cursor.execute(
            "INSERT OR REPLACE INTO max_seq_id (segment_id, seq_id) VALUES (?, ?)",
            (segment_id, next_seq_id)
        )
        return next_seq_id

    def _seq_id_to_blob(self, seq_id):
        return seq_id.to_bytes(8, byteorder='big')

    def _embedding_to_blob(self, embedding):
        if isinstance(embedding, np.ndarray):
            return embedding.tobytes()
        elif isinstance(embedding, list):
            return np.array(embedding, dtype=np.float32).tobytes()
        return embedding

    def add_documents(self, texts, metadatas, embeddings_model):
        self.connect()
        try:
            collection_id = self._get_collection_id()
            if not collection_id:
                collection_id = str(uuid.uuid4())
                self.cursor.execute(
                    "INSERT INTO collections (id, name, dimension, database_id, config_json_str, schema_str) VALUES (?, 'default', 384, 'default', '{}', '{}')",
                    (collection_id,)
                )
                self.cursor.execute("INSERT INTO databases (id, name, tenant_id) VALUES ('default', 'default', 'default')")
                self.cursor.execute("INSERT INTO tenants (id) VALUES ('default')")
                self.conn.commit()

            segment_id = self._get_or_create_segment(collection_id)

            embeddings = embeddings_model.embed_documents(texts)

            for idx, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                embedding_id = str(uuid.uuid4())
                seq_id = self._get_next_seq_id(segment_id)
                seq_id_blob = self._seq_id_to_blob(seq_id)
                embedding_blob = self._embedding_to_blob(embedding)
                created_at = datetime.now().isoformat()

                self.cursor.execute(
                    "INSERT INTO embeddings (segment_id, embedding_id, seq_id, created_at) VALUES (?, ?, ?, ?)",
                    (segment_id, embedding_id, seq_id_blob, created_at)
                )

                row_id = self.cursor.lastrowid

                self.cursor.execute(
                    "INSERT INTO embedding_fulltext_search_content (id, c0) VALUES (?, ?)",
                    (row_id, text)
                )

                for key, value in metadata.items():
                    if isinstance(value, str):
                        self.cursor.execute(
                            "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                            (row_id, key, value)
                        )
                    elif isinstance(value, int):
                        self.cursor.execute(
                            "INSERT INTO embedding_metadata (id, key, int_value) VALUES (?, ?, ?)",
                            (row_id, key, value)
                        )
                    elif isinstance(value, float):
                        self.cursor.execute(
                            "INSERT INTO embedding_metadata (id, key, float_value) VALUES (?, ?, ?)",
                            (row_id, key, value)
                        )
                    elif isinstance(value, bool):
                        self.cursor.execute(
                            "INSERT INTO embedding_metadata (id, key, bool_value) VALUES (?, ?, ?)",
                            (row_id, key, 1 if value else 0)
                        )

                if (idx + 1) % 100 == 0:
                    self.conn.commit()

            self.conn.commit()
            return len(texts)
        finally:
            self.close()
    
    def delete_documents_by_entry_id(self, entry_ids):
        self.connect()
        try:
            deleted_count = 0
            
            for entry_id in entry_ids:
                self.cursor.execute(
                    "SELECT id FROM embedding_metadata WHERE key = 'id' AND string_value = ?",
                    (entry_id,)
                )
                result = self.cursor.fetchone()
                
                if result:
                    row_id = result[0]
                    
                    self.cursor.execute("DELETE FROM embedding_metadata WHERE id = ?", (row_id,))
                    self.cursor.execute("DELETE FROM embedding_fulltext_search_content WHERE id = ?", (row_id,))
                    self.cursor.execute("DELETE FROM embeddings WHERE rowid = ?", (row_id,))
                    
                    deleted_count += 1
            
            self.conn.commit()
            return deleted_count
        finally:
            self.close()
