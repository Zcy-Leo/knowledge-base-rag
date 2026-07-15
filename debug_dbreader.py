import sys
sys.path.insert(0, '.')

from app_v2 import get_all_companies, get_all_topics, get_metadata_counts, DBReader, DB_PATH

print(f'DB_PATH: {DB_PATH}')

# Check if DB exists
import os
print(f'DB exists: {os.path.exists(DB_PATH)}')

# Test DBReader directly
try:
    reader = DBReader(DB_PATH)
    count = reader.count_documents()
    print(f'Documents in DB: {count}')
    
    metas = reader.get_all_metadatas()
    print(f'Metadata entries: {len(metas)}')
    
    # Check if any metadata has company/topic
    company_values = []
    topic_values = []
    for doc_id, m in metas.items():
        if isinstance(m, dict):
            c = m.get('company', '')
            t = m.get('topic', '')
            if c:
                company_values.append(c)
            if t:
                topic_values.append(t)
    
    print(f'Unique companies in metadata: {sorted(set(company_values))}')
    print(f'Unique topics in metadata: {sorted(set(topic_values))}')
    
except Exception as e:
    print(f'DBReader error: {e}')
    import traceback
    traceback.print_exc()
