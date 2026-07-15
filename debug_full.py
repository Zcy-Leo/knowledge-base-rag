import sys
sys.path.insert(0, '.')

import sqlite3
import os

# Test 1: Check database directly
print('=== Test 1: Database Check ===')
DB_PATH = 'my_local_database/chroma.sqlite3'
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT key, COUNT(*) FROM embedding_metadata GROUP BY key")
results = cursor.fetchall()
print('Metadata keys:', dict(results))

cursor.execute("SELECT string_value FROM embedding_metadata WHERE key='company'")
companies = [r[0] for r in cursor.fetchall() if r[0]]
print(f'Unique companies: {sorted(set(companies))}')

cursor.execute("SELECT string_value FROM embedding_metadata WHERE key='topic'")
topics = [r[0] for r in cursor.fetchall() if r[0]]
print(f'Unique topics: {sorted(set(topics))}')

conn.close()

# Test 2: Test get_all_companies and get_all_topics
print('\n=== Test 2: Function Check ===')
from app_v2 import get_all_companies, get_all_topics, get_metadata_counts

try:
    companies = get_all_companies()
    print(f'get_all_companies(): {companies}')
    print(f'Type: {type(companies)}, Length: {len(companies)}')
    
    topics = get_all_topics()
    print(f'get_all_topics(): {topics}')
    print(f'Type: {type(topics)}, Length: {len(topics)}')
    
    company_counts, topic_counts = get_metadata_counts()
    print(f'company_counts: {company_counts}')
    print(f'topic_counts: {topic_counts}')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()

# Test 3: Build dropdown options exactly like the app
print('\n=== Test 3: Build Options ===')
try:
    companies = get_all_companies()
    company_options = ["--- Select Company ---"]
    for c in companies:
        count = company_counts.get(c, 0)
        company_options.append(f"{c} ({count})")
    company_options.append("+ Add New Company")
    print(f'Company options: {company_options}')
    print(f'Length: {len(company_options)}')
    
    topics = get_all_topics()
    topic_options = ["--- Select Topic ---"]
    for t in topics:
        count = topic_counts.get(t, 0)
        topic_options.append(f"{t} ({count})")
    print(f'Topic options: {topic_options}')
    print(f'Length: {len(topic_options)}')
    
except Exception as e:
    print(f'ERROR: {e}')
