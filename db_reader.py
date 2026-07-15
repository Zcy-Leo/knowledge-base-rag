import sqlite3
import os

class DBReader:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def _connect(self):
        return sqlite3.connect(self.db_path)
    
    def count_documents(self):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0
    
    def get_all_metadatas(self):
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, key, string_value, int_value, float_value FROM embedding_metadata")
        rows = cursor.fetchall()
        
        metadata_dict = {}
        for doc_id, key, string_val, int_val, float_val in rows:
            if doc_id not in metadata_dict:
                metadata_dict[doc_id] = {}
            
            if string_val is not None:
                metadata_dict[doc_id][key] = string_val
            elif int_val is not None:
                metadata_dict[doc_id][key] = int_val
            elif float_val is not None:
                metadata_dict[doc_id][key] = float_val
        
        conn.close()
        return metadata_dict
    
    def get_all_documents(self):
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, c0 FROM embedding_fulltext_search_content")
        rows = cursor.fetchall()
        
        docs = {str(row[0]): row[1] for row in rows}
        
        conn.close()
        return docs
    
    def get_documents_with_metadatas(self):
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, c0 FROM embedding_fulltext_search_content")
        content_rows = cursor.fetchall()
        
        cursor.execute("SELECT id, key, string_value, int_value, float_value FROM embedding_metadata")
        meta_rows = cursor.fetchall()
        
        docs = {}
        for doc_id, content in content_rows:
            docs[str(doc_id)] = {"content": content, "metadata": {}}
        
        for doc_id, key, string_val, int_val, float_val in meta_rows:
            doc_id_str = str(doc_id)
            if doc_id_str in docs:
                if string_val is not None:
                    docs[doc_id_str]["metadata"][key] = string_val
                elif int_val is not None:
                    docs[doc_id_str]["metadata"][key] = int_val
                elif float_val is not None:
                    docs[doc_id_str]["metadata"][key] = float_val
        
        conn.close()
        return docs
    
    def get_metadata_by_id(self, doc_id):
        conn = self._connect()
        cursor = conn.cursor()
        
        try:
            doc_id_int = int(doc_id)
        except:
            doc_id_int = doc_id
        
        cursor.execute("SELECT key, string_value, int_value, float_value FROM embedding_metadata WHERE id = ?", (doc_id_int,))
        rows = cursor.fetchall()
        
        metadata = {}
        for key, string_val, int_val, float_val in rows:
            if string_val is not None:
                metadata[key] = string_val
            elif int_val is not None:
                metadata[key] = int_val
            elif float_val is not None:
                metadata[key] = float_val
        
        conn.close()
        return metadata
    
    def get_all_ids(self):
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM embeddings")
        rows = cursor.fetchall()
        
        conn.close()
        return [str(row[0]) for row in rows]