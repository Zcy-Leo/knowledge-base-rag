import os
import time
import sqlite3
import hashlib
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class EntryChange:
    entry_id: str
    status: str
    content_hash: str

class IncrementalIngestor:
    def __init__(self, db_path: str = None):
        if db_path is None:
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
            CREATE TABLE IF NOT EXISTS entry_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                valid_from TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                valid_to TIMESTAMP,
                version_number INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                UNIQUE(doc_id, entry_id, version_number)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doc_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL UNIQUE,
                last_version INTEGER NOT NULL DEFAULT 1,
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                entry_count INTEGER NOT NULL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.strip().encode('utf-8')).hexdigest()
    
    def _get_stored_doc_info(self, doc_id: str) -> Optional[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT last_version, entry_count FROM doc_metadata WHERE doc_id = ?
        ''', (doc_id,))
        result = cursor.fetchone()
        
        legacy_version = 0
        try:
            cursor.execute('''
                SELECT MAX(version_number) FROM doc_hash_store WHERE doc_id = ?
            ''', (doc_id,))
            legacy_result = cursor.fetchone()
            if legacy_result and legacy_result[0]:
                legacy_version = legacy_result[0]
        except:
            pass
        
        if not result:
            if legacy_version > 0:
                conn.close()
                return {
                    'version': legacy_version,
                    'entry_count': 0,
                    'entries': {},
                    'from_legacy': True
                }
            conn.close()
            return None
        
        stored_version = result[0]
        entry_count = result[1]
        
        if legacy_version > stored_version:
            stored_version = legacy_version
            cursor.execute('''
                UPDATE doc_metadata SET last_version = ? WHERE doc_id = ?
            ''', (stored_version, doc_id))
            conn.commit()
        
        cursor.execute('''
            SELECT entry_id, content_hash FROM entry_versions 
            WHERE doc_id = ? AND status = 'active'
        ''', (doc_id,))
        
        stored_entries = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return {
            'version': stored_version,
            'entry_count': entry_count,
            'entries': stored_entries
        }
    
    def detect_changes(self, entries: List[Dict], doc_id: str) -> Dict:
        new_hashes = set()
        new_entries_map = {}
        for entry in entries:
            content = entry.get('text', '')
            entry_id = entry.get('id', '')
            content_hash = self._compute_content_hash(content)
            new_hashes.add(content_hash)
            new_entries_map[entry_id] = content_hash
        
        stored_info = self._get_stored_doc_info(doc_id)
        
        if not stored_info:
            new_count = len(new_entries_map)
            return {
                'new': new_count,
                'modified': 0,
                'deleted': 0,
                'unchanged': 0,
                'total': new_count,
                'previous_version': 0,
                'new_version': 1,
                'new_entries': list(new_entries_map.keys()),
                'modified_entries': [],
                'deleted_entries': [],
                'unchanged_entries': []
            }
        
        stored_hashes = set(stored_info['entries'].values())
        previous_version = stored_info['version']
        
        unchanged_count = len(new_hashes & stored_hashes)
        new_count = len(new_hashes - stored_hashes)
        deleted_count = len(stored_hashes - new_hashes)
        modified_count = 0
        
        new_entries_list = []
        modified_entries_list = []
        deleted_entries_list = []
        unchanged_entries_list = []
        
        for entry_id, content_hash in new_entries_map.items():
            if content_hash in stored_hashes:
                unchanged_entries_list.append(entry_id)
            else:
                new_entries_list.append(entry_id)
        
        if new_count == 0 and modified_count == 0 and deleted_count == 0:
            new_version = previous_version
        else:
            new_version = previous_version + 1
        
        return {
            'new': new_count,
            'modified': modified_count,
            'deleted': deleted_count,
            'unchanged': unchanged_count,
            'total': len(new_entries_map),
            'previous_version': previous_version,
            'new_version': new_version,
            'new_entries': new_entries_list,
            'modified_entries': modified_entries_list,
            'deleted_entries': deleted_entries_list,
            'unchanged_entries': unchanged_entries_list
        }
    
    def persist_changes(self, entries: List[Dict], doc_id: str, changes: Dict):
        new_version = changes['new_version']
        previous_version = changes['previous_version']
        
        if new_version == previous_version:
            # Even if no content changed, update the timestamp
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE doc_metadata 
                SET last_updated = CURRENT_TIMESTAMP
                WHERE doc_id = ?
            ''', (doc_id,))
            conn.commit()
            conn.close()
            return
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for entry_id in changes.get('deleted_entries', []):
            cursor.execute('''
                UPDATE entry_versions 
                SET valid_to = CURRENT_TIMESTAMP, status = 'deleted'
                WHERE doc_id = ? AND entry_id = ? AND status = 'active'
            ''', (doc_id, entry_id))
        
        for entry_id in changes.get('modified_entries', []):
            cursor.execute('''
                UPDATE entry_versions 
                SET valid_to = CURRENT_TIMESTAMP, status = 'superseded'
                WHERE doc_id = ? AND entry_id = ? AND status = 'active'
            ''', (doc_id, entry_id))
        
        for entry in entries:
            entry_id = entry.get('id', '')
            content = entry.get('text', '')
            content_hash = self._compute_content_hash(content)
            
            if entry_id in changes.get('new_entries', []) or entry_id in changes.get('modified_entries', []):
                cursor.execute('''
                    INSERT INTO entry_versions 
                    (doc_id, entry_id, content_hash, version_number, status)
                    VALUES (?, ?, ?, ?, 'active')
                ''', (doc_id, entry_id, content_hash, new_version))
        
        cursor.execute('''
            INSERT OR REPLACE INTO doc_metadata 
            (doc_id, last_version, last_updated, entry_count)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
        ''', (doc_id, new_version, len(entries)))
        
        conn.commit()
        conn.close()
    
    def ingest_entries(self, entries: List[Dict], doc_id: str) -> Dict:
        start_time = time.time()
        
        changes = self.detect_changes(entries, doc_id)
        
        self.persist_changes(entries, doc_id, changes)
        
        elapsed_time = time.time() - start_time
        
        return {
            **changes,
            'embedding_time': elapsed_time,
            'embeddings_computed': changes['new'] + changes['modified'],
            'embeddings_reused': changes['unchanged'],
            'version_number': changes['new_version']
        }
    
    def get_doc_version_history(self, doc_id: str) -> List[Dict]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT last_version, last_updated FROM doc_metadata WHERE doc_id = ?
        ''', (doc_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{'version': r[0], 'timestamp': r[1]} for r in results]
    
    def get_active_entries(self, doc_id: str) -> List[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT entry_id FROM entry_versions 
            WHERE doc_id = ? AND status = 'active'
        ''', (doc_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [r[0] for r in results]
    
    def get_system_stats(self) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM entry_versions WHERE status = 'active'")
        active_entries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM entry_versions")
        total_versions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM entry_versions")
        unique_docs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM doc_metadata")
        tracked_docs = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'active_entries': active_entries,
            'total_versions': total_versions,
            'unique_documents': unique_docs,
            'tracked_documents': tracked_docs
        }
