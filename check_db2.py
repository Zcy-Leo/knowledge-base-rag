import sqlite3

conn = sqlite3.connect('my_local_database/chroma.sqlite3')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('所有表:', tables)

print()
print('=== companies 表 ===')
try:
    cursor.execute('SELECT * FROM companies')
    companies = cursor.fetchall()
    print(f'公司数量: {len(companies)}')
    for c in companies:
        print(c)
except Exception as e:
    print(f'错误: {e}')

print()
print('=== topics 表 ===')
try:
    cursor.execute('SELECT * FROM topics')
    topics = cursor.fetchall()
    print(f'Topic数量: {len(topics)}')
    for t in topics:
        print(t)
except Exception as e:
    print(f'错误: {e}')

conn.close()
