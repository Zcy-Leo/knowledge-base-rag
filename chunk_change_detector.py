import hashlib
import os
import sqlite3
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    position: int
    content: str
    status: str = "new"

class ChunkChangeDetector:
    def __init__(self, db_path: str = None):
        if db_path is None:
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "my_local_database", "livevectorlake.db")
        else:
            self.db_path = db_path
        self._ensure_tables()
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def _ensure_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                valid_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                valid_to TIMESTAMP,
                version_number INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                change_type TEXT NOT NULL DEFAULT 'insert',
                UNIQUE(chunk_id, version_number)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doc_hash_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                chunk_hashes TEXT NOT NULL,
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version_number INTEGER NOT NULL DEFAULT 1,
                UNIQUE(doc_id, version_number)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunk_versions_chunk_id ON chunk_versions(chunk_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunk_versions_doc_id ON chunk_versions(doc_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunk_versions_valid ON chunk_versions(valid_from, valid_to)
        ''')
        
        conn.commit()
        conn.close()
    
    def _compute_chunk_hash(self, content: str) -> str:
        normalized = content.strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _chunk_document(self, content: str, doc_id: str) -> List[Chunk]:
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            chunk_hash = self._compute_chunk_hash(para)
            chunks.append(Chunk(
                chunk_id=chunk_hash,
                doc_id=doc_id,
                position=i,
                content=para,
                status="new"
            ))
        return chunks
    
    def get_stored_hashes(self, doc_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT chunk_hashes, version_number FROM doc_hash_store WHERE doc_id = ?
        ''', (doc_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            import json
            return {
                'hashes': json.loads(result[0]),
                'version': result[1]
            }
        return None
    
    def detect_changes(self, content: str, doc_id: str) -> Tuple[List[Chunk], Dict]:
        new_chunks = self._chunk_document(content, doc_id)
        stored_data = self.get_stored_hashes(doc_id)
        
        if not stored_data:
            for chunk in new_chunks:
                chunk.status = "new"
            return new_chunks, {'new': len(new_chunks), 'modified': 0, 'deleted': 0, 'unchanged': 0}
        
        stored_hashes = stored_data['hashes']
        new_hashes = {c.chunk_id: c for c in new_chunks}
        
        new_count = 0
        modified_count = 0
        deleted_count = 0
        unchanged_count = 0
        
        for i, stored_hash in enumerate(stored_hashes):
            if stored_hash in new_hashes:
                new_chunks[new_hashes[stored_hash].position].status = "unchanged"
                unchanged_count += 1
            else:
                deleted_count += 1
        
        for chunk in new_chunks:
            if chunk.status == "new":
                prev_hash_at_pos = stored_hashes[chunk.position] if chunk.position < len(stored_hashes) else None
                if prev_hash_at_pos and prev_hash_at_pos != chunk.chunk_id:
                    chunk.status = "modified"
                    modified_count += 1
                else:
                    new_count += 1
        
        return new_chunks, {
            'new': new_count,
            'modified': modified_count,
            'deleted': deleted_count,
            'unchanged': unchanged_count,
            'total': len(new_chunks),
            'previous_version': stored_data['version']
        }
    
    def save_hashes(self, doc_id: str, chunks: List[Chunk]):
        import json
        hashes = [c.chunk_id for c in chunks]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT MAX(version_number) FROM doc_hash_store WHERE doc_id = ?
        ''', (doc_id,))
        result = cursor.fetchone()
        new_version = result[0] + 1 if result[0] else 1
        
        cursor.execute('''
            INSERT INTO doc_hash_store (doc_id, chunk_hashes, last_updated, version_number)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
        ''', (doc_id, json.dumps(hashes), new_version))
        
        conn.commit()
        conn.close()
    
    def save_chunk_version(self, chunk: Chunk, embedding: Optional[bytes] = None, doc_version: int = 1):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if chunk.status in ["modified", "new"]:
            cursor.execute('''
                UPDATE chunk_versions 
                SET valid_to = CURRENT_TIMESTAMP, status = 'superseded'
                WHERE doc_id = ? AND position = ? AND status = 'active'
            ''', (chunk.doc_id, chunk.position))
        
        cursor.execute('''
            INSERT INTO chunk_versions 
            (chunk_id, doc_id, position, content, embedding, valid_from, valid_to, 
             version_number, status, change_type)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL, ?, 'active', ?)
        ''', (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.position,
            chunk.content,
            embedding,
            doc_version,
            chunk.status
        ))
        
        conn.commit()
        conn.close()
    
    def get_chunk_at_time(self, chunk_id: str, timestamp: str) -> Optional[Chunk]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, doc_id, position, content, valid_from, valid_to, version_number, status
            FROM chunk_versions 
            WHERE chunk_id = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
            ORDER BY version_number DESC
            LIMIT 1
        ''', (chunk_id, timestamp, timestamp))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return Chunk(
                chunk_id=result[0],
                doc_id=result[1],
                position=result[2],
                content=result[3],
                status=result[7]
            )
        return None
    
    def get_doc_versions(self, doc_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version_number, last_updated FROM doc_hash_store WHERE doc_id = ? ORDER BY version_number
        ''', (doc_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'version': r[0], 'timestamp': r[1]} for r in results]
    
    def get_active_chunks(self, doc_id: str) -> List[Chunk]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT chunk_id, doc_id, position, content, version_number
            FROM chunk_versions 
            WHERE doc_id = ? AND status = 'active'
            ORDER BY position
        ''', (doc_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [Chunk(
            chunk_id=r[0],
            doc_id=r[1],
            position=r[2],
            content=r[3],
            status="active"
        ) for r in results]
    
    def cleanup_old_versions(self, doc_id: str, keep_versions: int = 5):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT version_number FROM doc_hash_store WHERE doc_id = ? ORDER BY version_number
        ''', (doc_id,))
        versions = [r[0] for r in cursor.fetchall()]
        
        if len(versions) > keep_versions:
            versions_to_delete = versions[:-keep_versions]
            for v in versions_to_delete:
                cursor.execute('''
                    DELETE FROM chunk_versions WHERE doc_id = ? AND version_number = ?
                ''', (doc_id, v))
            
            cursor.execute('''
                DELETE FROM doc_hash_store WHERE doc_id = ? AND version_number IN ({})
            '''.format(','.join('?' * len(versions_to_delete))), [doc_id] + versions_to_delete)
        
        conn.commit()
        conn.close()