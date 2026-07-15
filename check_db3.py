import sqlite3

conn = sqlite3.connect('my_local_database/chroma.sqlite3')
cursor = conn.cursor()

print('=== 检查 embedding_metadata 中的 company 和 topic ===')
cursor.execute("SELECT key, COUNT(*) FROM embedding_metadata GROUP BY key")
results = cursor.fetchall()
print('所有元数据键:')
for key, count in results:
    print(f'  {key}: {count}')

print()
print('=== company 值 ===')
cursor.execute("SELECT string_value, COUNT(*) FROM embedding_metadata WHERE key='company' GROUP BY string_value")
companies = cursor.fetchall()
if companies:
    for c, count in companies:
        print(f'  {c}: {count}')
else:
    print('  没有 company 元数据')

print()
print('=== topic 值 ===')
cursor.execute("SELECT string_value, COUNT(*) FROM embedding_metadata WHERE key='topic' GROUP BY string_value")
topics = cursor.fetchall()
if topics:
    for t, count in topics:
        print(f'  {t}: {count}')
else:
    print('  没有 topic 元数据')

conn.close()
