import sys
sys.path.insert(0, '.')

# Test the exact flow as the app
from app_v2 import get_all_companies, get_all_topics, get_metadata_counts

print('=== Testing exact flow ===')

try:
    companies = get_all_companies()
    print(f'get_all_companies() returned: {companies}')
    print(f'Type: {type(companies)}')
    print(f'Length: {len(companies)}')
except Exception as e:
    print(f'ERROR in get_all_companies(): {e}')

try:
    topics = get_all_topics()
    print(f'\nget_all_topics() returned: {topics}')
    print(f'Type: {type(topics)}')
    print(f'Length: {len(topics)}')
except Exception as e:
    print(f'ERROR in get_all_topics(): {e}')

try:
    company_counts, topic_counts = get_metadata_counts()
    print(f'\nget_metadata_counts():')
    print(f'  company_counts: {company_counts}')
    print(f'  topic_counts: {topic_counts}')
except Exception as e:
    print(f'ERROR in get_metadata_counts(): {e}')

# Test building options like the app does
print('\n=== Building dropdown options ===')
try:
    companies = get_all_companies()
    company_options = ["--- Select Company ---"]
    for c in companies:
        count = company_counts.get(c, 0) if 'company_counts' in dir() else 0
        company_options.append(f"{c} ({count})")
    company_options.append("+ Add New Company")
    print(f'Company options: {company_options}')
    
    topics = get_all_topics()
    topic_options = ["--- Select Topic ---"]
    for t in topics:
        count = topic_counts.get(t, 0) if 'topic_counts' in dir() else 0
        topic_options.append(f"{t} ({count})")
    print(f'Topic options: {topic_options}')
    
except Exception as e:
    print(f'ERROR building options: {e}')
